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

import litellm
import vertexai
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# ── Prompts ────────────────────────────────────────────────────────────────────

INTENT_SYSTEM = """You are an intent classifier. Given a user query, decide whether
the user wants:
  A) A SUMMARY of an entire document (they mention a document/book title or ask for
     a summary/overview/riassunto/sommario), OR
  B) A specific QA answer to a factual question.

Reply with exactly one word: SUMMARY or QA."""

QA_SYSTEM = """You are a precise Information Retrieval Assistant.
Language rule: always answer in Italian, regardless of the query language.

The context below contains paragraphs each ending with a citation:
  [Book Title | Chapter Title | Page Range]

Rules:
- Answer ONLY from the provided context. No external knowledge.
- At the end list every citation that contributed ≥20 % of your answer.
- If the answer is not in the context, say so in Italian.

Format:
<answer text>

Fonti:
[Citation 1]
[Citation 2]   ← only if applicable

Context:
{context}"""

QA_CITATION_SYSTEM = """You are a precise Information Retrieval Assistant.
Language rule: always answer in Italian, regardless of the query language.

The context contains paragraphs each ending with a citation:
  [Book Title | Chapter Title | Page Range]

Rules:
- Answer ONLY from the provided context. No external knowledge.
- Include EVERY distinct citation that appears in the context (de-duplicated).
- If the answer is not in the context, say so in Italian.

Format:
<answer text>

Fonti:
[Citation 1]
[Citation 2]   ← only if applicable

Context:
{context}"""

SUMMARY_SYSTEM = """You are a Document Summarisation Expert.
Language rule: always write in Italian.

You are given all the chunks from a specific document.
Write a comprehensive, well-structured summary organised by chapter/section.
End with a brief list of key takeaways.

Context:
{context}"""


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
        chat_model: str = "vertex_ai/gemini-2.5-flash-preview-04-17",
        intent_model: str = "vertex_ai/gemini-2.0-flash-001",
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
        embeddings = VertexAIEmbeddings(
            model_name="text-embedding-004",
            project=project_id,
        )
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        self.vector_db = FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )

        # Load document titles from metadata for summary matching
        self._doc_titles: List[Tuple[str, str]] = []  # [(title, source), ...]
        if os.path.exists(metadata_json):
            with open(metadata_json, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            for v in meta.values():
                title = v.get("title", "")
                source = v.get("source", "general")
                if title:
                    self._doc_titles.append((title, source))

    # ------------------------------------------------------------------
    # Intent detection
    # ------------------------------------------------------------------

    def _detect_intent(self, query: str) -> str:
        """Returns 'SUMMARY' or 'QA'."""
        try:
            resp = litellm.completion(
                model=self.intent_model,
                messages=[
                    {"role": "system", "content": INTENT_SYSTEM},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                max_tokens=10,
            )
            intent = resp.choices[0].message.content.strip().upper()
            return "SUMMARY" if "SUMMARY" in intent else "QA"
        except Exception:
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

    def _qa_retrieval(self, query: str, source: str, k: int = 16) -> List[Document]:
        kwargs: dict = {"k": k}
        if source != "general":
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
        best_title, best_source, best_score = "", "general", 0.0
        for title, source in self._doc_titles:
            score = _similarity(query, title)
            if score > best_score:
                best_score, best_title, best_source = score, title, source

        # Fetch ALL chunks with that title (FAISS doesn't support "get all by
        # filter" natively, so we use a large-k similarity search + filter)
        docs = self.vector_db.similarity_search(
            best_title,
            k=500,
            filter={"title": best_title},
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

        else:  # QA
            docs = self._qa_retrieval(query, source, k=k)
            if not docs:
                return (
                    "Le risorse attualmente disponibili non forniscono una risposta "
                    "definitiva alla Sua domanda."
                )
            context = _format_context(docs)
            template = QA_CITATION_SYSTEM if source == "citazione" else QA_SYSTEM
            system = template.format(context=context)
            answer = self._llm(system, query)

        self._update_memory(query, answer)
        return answer