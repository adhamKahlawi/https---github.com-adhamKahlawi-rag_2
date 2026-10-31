"""
chat_llm.py
RAG chat engine backed by LiteLLM.
"""

import json
import os
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import litellm
import vertexai
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.prompts import (
    DIRECT_LLM_SYSTEM,
    INTENT_SYSTEM,
    MANUAL_SYSTEM,
    QA_CITATION_SYSTEM,
    QA_SYSTEM,
    QUERY_REWRITE_SYSTEM,
    SUMMARY_SYSTEM,
)
from src.core.citations import format_academic_response


INTENTS = {"SUMMARY", "QA", "MANUAL", "INFORMATION", "LLM"}
SUMMARY_MATCH_THRESHOLD = 0.45
# Relevance scores returned by FAISS are distance-normalized by LangChain.
# Low-scoring neighbours are useful for retrieval diagnostics, but must not
# become citation candidates.
CITATION_RELEVANCE_THRESHOLD = 0.35
NO_SOURCE_ANSWER = (
    "Le risorse attualmente disponibili non forniscono una risposta definitiva alla Sua domanda.\n\n"
    "[Fonti: nessuna fonte pertinente]"
)


def _normalize_for_embedding(text: str) -> str:
    """Normalize text immediately before embedding lookup."""
    return (text or "").strip().lower()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[\w']+", text.lower())


def _similarity(a: str, b: str) -> float:
    """Simple fuzzy similarity between two strings (0-1)."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _token_set_similarity(a: str, b: str) -> float:
    a_tokens = set(_tokenize(a))
    b_tokens = set(_tokenize(b))
    if not a_tokens or not b_tokens:
        return 0.0
    overlap = len(a_tokens & b_tokens)
    precision = overlap / len(a_tokens)
    recall = overlap / len(b_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _title_match_score(query: str, title: str) -> float:
    query_l = query.lower()
    title_l = title.lower()
    containment = 1.0 if title_l in query_l or query_l in title_l else 0.0
    return max(_similarity(query_l, title_l), _token_set_similarity(query_l, title_l), containment)


def _format_context(docs: List[Document]) -> str:
    parts = []
    seen = set()
    for doc in docs:
        m = doc.metadata
        key = (m.get("title", ""), m.get("chapter", ""), m.get("page_range", ""), doc.page_content[:300])
        if key in seen:
            continue
        seen.add(key)
        citation = f"[{m.get('title', '')} | {m.get('chapter', '')} | {m.get('page_range', '')}]"
        parts.append(f"{doc.page_content} {citation}.")
    return "\n---\n".join(parts)


def _build_messages(system: str, history: list, user_query: str, include_history: bool = True) -> list:
    messages = [{"role": "system", "content": system}]
    if include_history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_query})
    return messages


def _dedupe(items: List[str], limit: Optional[int] = None) -> List[str]:
    seen = set()
    deduped = []
    for item in items:
        normalized = re.sub(r"\s+", " ", item).strip(" []\t\r\n,.;")
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
        if limit and len(deduped) >= limit:
            break
    return deduped


class GeminiRAGSystem:
    def __init__(
        self,
        project_id: str,
        index_path: str = "vector_database",
        # Vertex AI model IDs must use the API slug, not the display name.
        # In particular, a display name such as ``Gemini 3.5 Flash-Lite`` is
        # interpreted by LiteLLM as a custom Vertex endpoint name.
        chat_model: Optional[str] = None,
        intent_model: Optional[str] = None,
        memory_k: int = 10,
        metadata_json: str = "vector_database/doc_metadata.json",
    ):
        os.environ["VERTEXAI_PROJECT"] = project_id
        os.environ["VERTEXAI_LOCATION"] = "us-central1"

        vertexai.init(project=project_id)
        default_model = os.getenv("VERTEX_CHAT_MODEL", "vertex_ai/gemini-2.5-flash-lite")
        self.chat_model = chat_model or default_model
        self.intent_model = intent_model or os.getenv("VERTEX_INTENT_MODEL", self.chat_model)
        self.memory_k = memory_k
        self._history: list = []

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            project=project_id,
        )
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        self.vector_db = FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )

        self._doc_catalog: List[Dict[str, str]] = []
        if os.path.exists(metadata_json):
            with open(metadata_json, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            for v in meta.values():
                title = v.get("title", "")
                if title:
                    self._doc_catalog.append(
                        {
                            "title": title,
                            "source": v.get("source", "general"),
                        }
                    )

    # ------------------------------------------------------------------
    # LLM utilities
    # ------------------------------------------------------------------

    def _json_completion(
        self,
        system: str,
        query: str,
        schema: dict,
        max_tokens: int,
        include_history: bool,
    ) -> dict:
        resp = litellm.completion(
            model=self.intent_model,
            messages=_build_messages(system, self._history, query, include_history=include_history),
            temperature=0.0,
            max_tokens=max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "rag_structured_output",
                    "strict": True,
                    "schema": schema,
                },
            },
        )
        content = resp.choices[0].message.content.strip()
        return json.loads(content)

    def _rewrite_query(self, query: str) -> str:
        if not self._history:
            return query.strip()

        schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        }
        try:
            data = self._json_completion(
                QUERY_REWRITE_SYSTEM,
                query,
                schema,
                max_tokens=160,
                include_history=True,
            )
            rewritten = data.get("query", "").strip()
            return rewritten or query.strip()
        except Exception as exc:
            print(f"Query rewrite fallback: {exc}")
            return query.strip()

    def _detect_intent(self, query: str) -> str:
        schema = {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": sorted(INTENTS),
                }
            },
            "required": ["intent"],
            "additionalProperties": False,
        }
        try:
            data = self._json_completion(
                INTENT_SYSTEM,
                query,
                schema,
                max_tokens=32,
                include_history=False,
            )
            intent = str(
                data.get("intent")
                or data.get("Intent")
                or data.get("Categories")
                or data.get("categories")
                or ""
            ).upper()
            return intent if intent in INTENTS else "QA"
        except Exception as exc:
            print(f"Intent detection fallback: {exc}")
            return "QA"

    def _llm(self, system_prompt: str, query: str, include_history: bool = False) -> str:
        resp = litellm.completion(
            model=self.chat_model,
            messages=_build_messages(system_prompt, self._history, query, include_history=include_history),
            temperature=0.2,
            max_tokens=2048,
        )
        return resp.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    # Source / keyword pre-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_source_prefix(user_input: str) -> Tuple[str, str]:
        cleaned = user_input.strip()
        for kw in ("citazione", "manuale"):
            if cleaned.lower().startswith(kw):
                return cleaned[len(kw):].strip(), kw
        return cleaned, "general"

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def _qa_retrieval(self, query: str, source: str = "general", k: int = 16) -> List[Document]:
        # Query the vector index with small, independent facets. This avoids a
        # single embedding dominating multi-concept questions (e.g. accessibilita
        # AND riusabilita) and avoids expensive graph-style traversals.
        facets = [query]
        terms = re.split(r"\s+(?:e|ed|and|oppure|o|sia|che|con|vs\.?|versus)\s+|[,;?]", query, flags=re.I)
        facets.extend(part.strip() for part in terms if len(part.strip().split()) >= 2)
        results: List[Document] = []
        seen = set()
        per_facet = max(4, min(10, k // max(1, len(facets))))
        for facet in facets[:6]:
            kwargs: dict = {"k": per_facet}
            if source and source != "general":
                kwargs["filter"] = {"source": source}
            try:
                scored = self.vector_db.similarity_search_with_relevance_scores(
                    _normalize_for_embedding(facet), **kwargs
                )
                found = [
                    doc for doc, relevance in scored
                    if relevance >= CITATION_RELEVANCE_THRESHOLD
                ]
            except Exception as exc:
                # Keep compatibility with vector-store implementations that
                # expose only similarity_search; citation post-processing still
                # prevents citations on explicit no-context answers.
                print(f"Scored retrieval fallback ({facet!r}): {exc}")
                try:
                    found = self.vector_db.similarity_search(
                        _normalize_for_embedding(facet), **kwargs
                    )
                except Exception as inner_exc:
                    print(f"Retrieval facet fallback ({facet!r}): {inner_exc}")
                    found = []
            for doc in found:
                m = doc.metadata
                key = (m.get("title", ""), m.get("chapter", ""), m.get("page_range", ""), doc.page_content[:300])
                if key not in seen:
                    seen.add(key)
                    results.append(doc)
        return results[: max(k, 1)]

    def _retrieve_all_sources(self, query: str, k: int) -> List[Document]:
        """Retrieve both course material and bibliography, then interleave them."""
        docs = self._qa_retrieval(query, source="general", k=k)
        for source in ("citazione", "manuale"):
            docs.extend(self._qa_retrieval(query, source=source, k=max(4, k // 2)))
        unique = []
        seen = set()
        for doc in docs:
            m = doc.metadata
            key = (m.get("title", ""), m.get("chapter", ""), m.get("page_range", ""), doc.page_content[:300])
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        return unique[: max(k * 2, 1)]

    def _summary_retrieval(self, query: str) -> Tuple[List[Document], str, float]:
        if not self._doc_catalog:
            docs = self.vector_db.similarity_search(_normalize_for_embedding(query), k=50)
            return docs, "unknown", 1.0 if docs else 0.0

        best = {"title": "", "source": "citazione"}
        best_score = 0.0
        for item in self._doc_catalog:
            score = _title_match_score(query, item["title"])
            if score > best_score:
                best_score = score
                best = item

        if best_score < SUMMARY_MATCH_THRESHOLD:
            return [], best["title"], best_score

        docs = self.vector_db.similarity_search(
            _normalize_for_embedding(best["title"]),
            k=500,
            filter={"title": best["title"], "source": best.get("source", "citazione")},
        )

        seen: set = set()
        unique: List[Document] = []
        for doc in docs:
            key = (doc.metadata.get("title", ""), doc.metadata.get("chapter", ""), doc.page_content[:200])
            if key not in seen:
                seen.add(key)
                unique.append(doc)

        return unique, best["title"], best_score

    # ------------------------------------------------------------------
    # Source post-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _sources_from_docs(docs: List[Document], limit: int = 8) -> List[str]:
        sources = []
        for doc in docs:
            m = doc.metadata
            title = m.get("title", "").strip()
            chapter = m.get("chapter", "").strip()
            page_range = m.get("page_range", "").strip()
            source_path = m.get("source_path", "").strip()
            if title or chapter or page_range:
                sources.append(" | ".join(part for part in (title, chapter, page_range) if part))
            elif source_path:
                sources.append(source_path)
        return _dedupe(sources, limit=limit)

    @staticmethod
    def _extract_sources_from_answer(answer: str) -> Tuple[str, List[str]]:
        lines = answer.splitlines()
        body_lines: List[str] = []
        source_lines: List[str] = []
        in_sources = False
        blank_after_sources = False

        for line in lines:
            if re.match(r"^\s*fonti\s*:", line, flags=re.IGNORECASE):
                in_sources = True
                blank_after_sources = False
                tail = line.split(":", 1)[1].strip()
                if tail:
                    source_lines.append(tail)
                continue
            if in_sources:
                stripped = line.strip()
                if not stripped:
                    blank_after_sources = True
                    continue
                likely_source = (
                    re.match(r"^\s*[-*\[]", line) is not None
                    or (stripped.count("|") >= 2 and "[" not in stripped)
                )
                if blank_after_sources and not likely_source:
                    in_sources = False
                    body_lines.append("")
                    body_lines.append(line)
                else:
                    blank_after_sources = False
                    source_lines.append(stripped)
            else:
                body_lines.append(line)

        body = "\n".join(body_lines).strip()
        sources: List[str] = []

        source_text = "\n".join(source_lines)
        sources.extend(re.findall(r"\[([^\[\]\n]*\|[^\[\]\n]*\|[^\[\]\n]*)\]", source_text))
        if not sources and source_text.strip():
            for line in source_lines:
                cleaned = re.sub(r"^\s*[-*]\s*", "", line).strip(" []")
                sources.extend(part.strip(" []") for part in cleaned.split(",") if part.strip())

        inline_sources = re.findall(r"\[([^\[\]\n]*\|[^\[\]\n]*\|[^\[\]\n]*)\]", body)
        sources.extend(inline_sources)
        body = re.sub(r"\s*\[[^\[\]\n]*\|[^\[\]\n]*\|[^\[\]\n]*\]\s*", " ", body)
        body = re.sub(r"[ \t]+([.,;:])", r"\1", body)
        body = re.sub(r"\n{3,}", "\n\n", body).strip()
        return body, _dedupe(sources)

    def _postprocess_sources(self, answer: str, docs: List[Document]) -> str:
        body, sources = self._extract_sources_from_answer(answer)
        # A refusal is an answer about the absence of evidence, not an answer
        # supported by every document returned by a broad vector search.
        # Remove both model-supplied and application-generated citations.
        if self._is_no_context_answer(body):
            return body or answer.strip()
        if not sources:
            sources = self._sources_from_docs(docs)
        if not sources:
            return body or answer.strip()
        # Citation numbering is deterministic and owned by the application.
        # The compact marker at the end of the prose keeps legacy model output
        # valid while the References block provides the full source details.
        markers = " ".join(f"[{i}]" for i in range(1, len(sources) + 1))
        return format_academic_response(f"{body or answer.strip()} {markers}", sources)

    @staticmethod
    def _is_no_context_answer(text: str) -> bool:
        """Return true for a refusal that must never receive citations."""
        normalized = re.sub(r"\s+", " ", text or "").strip().lower()
        if not normalized:
            return False
        refusal_patterns = (
            r"non ci sono informazioni.*(contesto|document|fornit)",
            r"non (?:ho|abbiamo) informazioni.*(contesto|document|fornit)",
            r"informazioni insufficienti.*(contesto|document)",
            r"non è possibile rispondere.*(contesto|document)",
            r"non posso rispondere.*(contesto|document)",
            r"not enough information.*(context|document)",
            r"not (?:available|contained) in the (?:provided )?context",
            r"cannot answer.*(context|document)",
        )
        return any(re.search(pattern, normalized) for pattern in refusal_patterns)

    # ------------------------------------------------------------------
    # Answer paths
    # ------------------------------------------------------------------

    def _answer_summary(self, query: str, k: int) -> str:
        docs, matched_title, score = self._summary_retrieval(query)
        if score < SUMMARY_MATCH_THRESHOLD or not docs:
            print(
                f"SUMMARY fallback to QA: matched_title={matched_title!r}, score={score:.3f}, docs={len(docs)}"
            )
            return self._answer_qa(query, source="general", k=k)

        context = _format_context(docs)
        system = SUMMARY_SYSTEM.format(context=context)
        answer = self._llm(system, f"Riassumi il documento: {matched_title}", include_history=False)
        return self._postprocess_sources(answer, docs)

    def _answer_information(self, query: str, k: int) -> str:
        docs = self._retrieve_all_sources(query, k)
        if not docs:
            return NO_SOURCE_ANSWER
        context = _format_context(docs)
        system = QA_CITATION_SYSTEM.format(context=context)
        answer = self._llm(system, query, include_history=False)
        return self._postprocess_sources(answer, docs)

    def _answer_manual(self, query: str) -> str:
        docs = self._qa_retrieval(query, source="manuale", k=100)
        if not docs:
            return NO_SOURCE_ANSWER
        context = _format_context(docs)
        system = MANUAL_SYSTEM.format(context=context)
        answer = self._llm(system, query, include_history=False)
        return self._postprocess_sources(answer, docs)

    def _answer_qa(self, query: str, source: str, k: int) -> str:
        answer_parts: List[str] = []
        used_docs: List[Document] = []

        primary_docs = self._retrieve_all_sources(query, k) if source == "general" else self._qa_retrieval(query, source=source, k=k)
        if primary_docs:
            context = _format_context(primary_docs)
            system = QA_SYSTEM.format(context=context)
            answer_parts.append(self._llm(system, query, include_history=False))
            used_docs.extend(primary_docs)

        if source not in ("citazione", "general"):
            citation_docs = self._qa_retrieval(query, source="citazione", k=k)
            if citation_docs:
                context = _format_context(citation_docs)
                system = QA_CITATION_SYSTEM.format(context=context)
                answer_parts.append(self._llm(system, query, include_history=False))
                used_docs.extend(citation_docs)

        if not answer_parts:
            return NO_SOURCE_ANSWER

        answer = "\n\n".join(answer_parts)
        return self._postprocess_sources(answer, used_docs)

    def _answer_direct_llm(self, query: str) -> str:
        try:
            raw_answer = self._llm(DIRECT_LLM_SYSTEM, query, include_history=True)
        except Exception as exc:
            raw_answer = f"Errore durante la generazione della risposta: {exc}"
        return (
            f"{raw_answer}\n\n"
            "Nota: la risposta e stata generata direttamente dal LLM senza consultare la base documentale."
        )

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def _update_memory(self, query: str, answer: str) -> None:
        self._history.append({"role": "user", "content": query})
        self._history.append({"role": "assistant", "content": answer})
        max_msgs = self.memory_k * 2
        if len(self._history) > max_msgs:
            self._history = self._history[-max_msgs:]

    def reset(self) -> None:
        self._history = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(self, user_query: str, k: int = 16, manual_confirmed: Optional[bool] = None) -> str:
        if not user_query or not user_query.strip():
            return ""

        query, source = self._parse_source_prefix(user_query)
        search_query = self._rewrite_query(query)
        intent = self._detect_intent(search_query)

        if intent == "MANUAL":
            answer = self._answer_manual(search_query)
        elif intent == "SUMMARY":
            answer = self._answer_summary(search_query, k=k)
        elif intent == "INFORMATION":
            answer = self._answer_information(search_query, k=k)
        elif intent == "LLM":
            answer = self._answer_direct_llm(search_query)
        else:
            answer = self._answer_qa(search_query, source=source, k=k)

        self._update_memory(query, answer)
        return answer
