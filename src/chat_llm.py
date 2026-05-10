"""
chat_llm.py
RAG chat engine backed by LiteLLM (swap model with one env-var change).

Key improvements vs original
─────────────────────────────
1. LiteLLM replaces the direct Vertex AI SDK calls.
2. Intent detection: classifies the query as QA or SUMMARY before retrieval.
3. Summary path: fetches ALL chunks whose `title` matches the requested document
   instead of top-k similarity results.
4. QA path: standard top-k similarity search (with optional source filter).
5. Conversation memory maintained with a simple rolling list (no langchain dep).
6. Cleaner prompt templates with explicit language instruction.
"""

import os
import json
from difflib import SequenceMatcher
from typing import List, Tuple
from src.prompts import INTENT_SYSTEM, QA_SYSTEM, QA_CITATION_SYSTEM, SUMMARY_SYSTEM,MANUAL_SYSTEM

import litellm
import vertexai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ── Helpers ────────────────────────────────────────────────────────────────────

def _similarity(a: str, b: str) -> float:
    """Simple fuzzy similarity between two strings (0–1)."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _format_context(docs: List[Document]) -> str:
    parts = []
    for doc in docs:
        m = doc.metadata
        citation = f"[{m.get('title', '')}|{m.get('chapter', '')}|{m.get('page_range', '')}]"
        parts.append(f"{doc.page_content} {citation}.")
    return "\n---\n".join(parts)


def _build_messages(system: str, history: list, user_query: str) -> list:
    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_query})
    return messages


# ── Main class ─────────────────────────────────────────────────────────────────

class GeminiRAGSystem:
    def __init__(
        self,
        project_id: str,
        index_path: str = "vector_database",
        chat_model: str = "vertex_ai/gemini-2.0-flash",
        intent_model: str = "vertex_ai/gemini-2.0-flash",
        memory_k: int = 10,
        metadata_json: str = "vector_database/doc_metadata.json",
    ):
        """
        Parameters
        ----------
        project_id    : Google Cloud project ID.
        index_path    : Folder containing the saved FAISS index.
        chat_model    : LiteLLM model string for generating answers.
        intent_model  : LiteLLM model string for intent classification
                        (cheap/fast model is fine here).
        memory_k      : Number of conversation turns to keep in memory.
        metadata_json : Path to doc_metadata.json (needed for summary lookup).
        """
        os.environ["VERTEXAI_PROJECT"]  = project_id
        os.environ["VERTEXAI_LOCATION"] = "us-central1"

        vertexai.init(project=project_id)
        self.chat_model = chat_model
        self.intent_model = intent_model
        self.memory_k = memory_k
        self._history: list = []          # rolling list of {role, content} dicts

        # Load FAISS
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            project=project_id,
        )
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        self.vector_db = FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )

        # Load document titles from metadata for summary matching
        self._doc_titles: List = []  
        if os.path.exists(metadata_json):
            with open(metadata_json, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            for v in meta.values():
                title = v.get("title", "") 
                if title:
                    self._doc_titles.append(title)

    # ------------------------------------------------------------------
    # Intent detection
    # ------------------------------------------------------------------

    def _detect_intent(self, query: str) -> str:
        """Returns 'SUMMARY', 'INFORMATION', 'MANUAL', LLM, or 'QA'."""
        try:
            response_schema = {
                "type": "object",
                "properties": {
                    "Categories": {
                        "type": "string",
                        "enum": ["SUMMARY", "QA", "MANUAL", "INFORMATION","LLM"],
                        "description": "The classified intent category of the user query."
                    }
                },
                "required": ["Categories"],
                "additionalProperties": False
            }
            resp = litellm.completion(
                model=self.intent_model,
                messages=[
                    {"role": "system", "content": INTENT_SYSTEM},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                max_tokens=10,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "query_category", # Fixed typo in name
                        "strict": True,
                        "schema": response_schema
                    }
                },
            )
            intent_str = resp.choices[0].message.content.strip()
            
            # 1. Try to parse as JSON first
            try:
                # Use json.loads() for strings
                data = json.loads(intent_str)
                # Check for variations in case just to be safe
                cat = data.get("Categories") or data.get("categories") or data.get("CATEGORIES")
                
                if cat and cat.upper() in ["SUMMARY", "QA", "MANUAL", "INFORMATION"]:
                    return cat.upper()
            except json.JSONDecodeError:
                # If JSON parsing fails, we'll gracefully fall down to the string-matching logic
                pass 
            
            # 2. Fallback: robust string matching if JSON is malformed
            intent_lower = intent_str.lower()
            for word in ["SUMMARY", "QA", "MANUAL", "INFORMATION", "LLM"]:
                if word.lower() in intent_lower:
                    return word
            
            # 3. Ultimate safe default if nothing matches
            print("###################")
            print(f"Fallback triggered. Unmatched intent string: {intent_str}")
            print("###################")
            return "QA"
            
        except Exception as e:
            print("###################")
            print(f"Fatal error in intent detection: {e}")
            print("###################")
            return "QA"  # safe default

    # ------------------------------------------------------------------
    # Source / keyword pre-processing (preserve original behaviour)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_source_prefix(user_input: str) -> Tuple[str, str]:
        """
        Strip leading keyword (citazione / manuale) and return
        (cleaned_query, source_filter).
        """
        cleaned = user_input.strip()
        for kw in ("citazione", "manuale"):
            if cleaned.lower().startswith(kw):
                return cleaned[len(kw):].strip(), kw
        return cleaned, "general"

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def _qa_retrieval(self, query: str, source: str="general", k: int = 16) -> List[Document]:
        kwargs: dict = {"k": k}
        #if source != "general":
        kwargs["filter"] = {"source": source}
        return self.vector_db.similarity_search(query, **kwargs)

    def _summary_retrieval(self, query: str) -> Tuple[List[Document], str]:
        """
        Find the document whose title best matches the query, then return ALL
        chunks for that document.
        Returns (docs, matched_title).
        """
        if not self._doc_titles:
            # Fall back to top-k if no metadata available
            return self.vector_db.similarity_search(query, k=50), "unknown"

        # Score each known title against the query
        best_title, best_source, best_score = "", "citazione", 0.0
        for title in self._doc_titles:
            score = _similarity(query, title)
            if score > best_score:
                best_score, best_title = score, title

        # Fetch ALL chunks with that title (FAISS doesn't support "get all by
        # filter" natively, so we use a large-k similarity search + filter)
        docs = self.vector_db.similarity_search(
            best_title,
            k=500,
            filter={"title": best_title, "source": best_source},
        )

        # De-duplicate by page_content
        seen: set = set()
        unique: List[Document] = []
        for doc in docs:
            key = doc.page_content[:200]
            if key not in seen:
                seen.add(key)
                unique.append(doc)

        return unique, best_title

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _llm(self, system_prompt: str, query: str) -> str:
        messages = _build_messages(system_prompt, self._history, query)
        resp = litellm.completion(
            model=self.chat_model,
            messages=messages,
            temperature=0.2,
            max_tokens=2048,
        )
        return resp.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def _update_memory(self, query: str, answer: str) -> None:
        self._history.append({"role": "user", "content": query})
        self._history.append({"role": "assistant", "content": answer})
        # Keep only last k exchanges (2 messages per exchange)
        max_msgs = self.memory_k * 2
        if len(self._history) > max_msgs:
            self._history = self._history[-max_msgs:]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(self, user_query: str, k: int = 16) -> str:
        """
        Process a user query end-to-end:
        1. Parse optional source prefix (citazione / manuale).
        2. Detect intent: SUMMARY or QA.
        3. Retrieve relevant chunks accordingly.
        4. Generate and return the Italian-language answer.
        """
        query, source = self._parse_source_prefix(user_query)
        intent = self._detect_intent(query)

        if intent == "SUMMARY":
            docs, matched_title = self._summary_retrieval(query)

            if not docs:
                answer = (
                    "Non ho trovato un documento corrispondente al titolo indicato."
                )
            else:
                context = _format_context(docs)
                system = SUMMARY_SYSTEM.format(context=context)
                answer = self._llm(system, f"Riassumi il documento: {matched_title}")
        elif intent == "INFORMATION": 
            docs = self._qa_retrieval(query, source = "citazione", k=k)
            if not docs:
                return (
                    "Le risorse attualmente disponibili non forniscono una risposta "
                    "definitiva alla Sua domanda."
                )
            context = _format_context(docs)
            template = QA_CITATION_SYSTEM #if source == "citazione" else QA_SYSTEM
            system = template.format(context=context)
            answer = self._llm(system, query)
        elif intent == "MANUAL": 
            docs = self._qa_retrieval(query, source = "manuale", k=100)
            if not docs:
                return (
                    "Le risorse attualmente disponibili non forniscono una risposta "
                    "definitiva alla Sua domanda."
                )
            context = _format_context(docs)
            print(context)
            template = MANUAL_SYSTEM #if source == "citazione" else QA_SYSTEM
            system = template.format(context=context)
            answer = self._llm(system, query)
        elif intent == "LLM":
            # ── Direct LLM answer (no retrieval) ──────────────────────────
            try:
                resp = litellm.completion(
                    model=self.chat_model,
                    messages=_build_messages(
                        "Sei un assistente esperto. Rispondi in modo chiaro e preciso. "
                        "Alla fine della risposta, cita sempre le fonti utilizzate "
                        "(modello, data di addestramento o conoscenza generale).",
                        self._history,
                        query,
                    ),
                    temperature=0.0,
                    max_tokens=2048,
                )
                raw_answer = resp.choices[0].message.content.strip()
            except Exception as e:
                raw_answer = f"Errore durante la generazione della risposta: {e}"

            answer = (
                f"{raw_answer}\n\n"
                "---\n"
                "⚠️ *Nota: la risposta è stata generata direttamente dal LLM "
                "senza consultare la base documentale.*"
            )

        else:  # QA
            # from the dispensa
            docs = self._qa_retrieval(query, source, k=k)
            if not docs:
                return (
                    "Le risorse attualmente disponibili non forniscono una risposta "
                    "definitiva alla Sua domanda."
                )
            context = _format_context(docs)
            template = QA_SYSTEM #QA_CITATION_SYSTEM if source == "citazione" else QA_SYSTEM
            system = template.format(context=context)
            answer_1 = self._llm(system, query)
            # from citation
            docs = self._qa_retrieval(query, source = "citazione", k=k)
            if not docs:
                answer_2 = None
            else:
                context = _format_context(docs)
                template = QA_CITATION_SYSTEM #if source == "citazione" else QA_SYSTEM
                system = template.format(context=context)
                answer_2 = self._llm(system, query)
            if answer_2 == None:
                answer = answer_1
            else:
                answer = answer_1 + "\n" + answer_2
        self._update_memory(query, answer)
        return answer