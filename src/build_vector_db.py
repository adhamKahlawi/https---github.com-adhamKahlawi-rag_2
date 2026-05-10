"""
build_vector_db.py  – improved PDF handling for tables and images
"""

import os
import json
import time
import random
from typing import List, Optional

import vertexai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    Docx2txtLoader,
    TextLoader,
    UnstructuredPDFLoader,
)
import pymupdf4llm  # pip install pymupdf4llm
import pdfplumber  # pip install pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorDBBuilder:
    def __init__(self, project_id: str):
        vertexai.init(project=project_id)
        self.embeddings =  GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            project=project_id,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ext(file_path: str) -> str:
        return os.path.splitext(file_path)[1].lstrip(".").lower()

    # ------------------------------------------------------------------
    # PDF loading — three-tier fallback
    # ------------------------------------------------------------------

    def _load_pdf_pages(self, file_path: str) -> List[Document]:
        """
        Tier 1 – pymupdf4llm  : best for text + tables (renders Markdown).
        Tier 2 – pdfplumber   : good structured-text fallback with table repair.
        Tier 3 – unstructured : OCR fallback for scanned / image-heavy PDFs.
        """

        # ── Tier 1: pymupdf4llm ──────────────────────────────────────────
        try:
            md_pages: list[str] = pymupdf4llm.to_markdown(
                file_path, 
                pages=None,          # all pages
                page_chunks=True,    # one entry per page
            )
            if md_pages:
                docs = []
                for i, page in enumerate(md_pages):
                    # page_chunks=True returns dicts with 'text' key
                    text = page["text"] if isinstance(page, dict) else page
                    if text.strip():
                        docs.append(
                            Document(
                                page_content=text,
                                metadata={"page": i + 1, "content_type": "markdown"},
                            )
                        )
                if docs:
                    return docs
        except Exception as e:
            print(f"[pymupdf4llm] failed for {file_path}: {e}")

        # ── Tier 2: pdfplumber – extract text + repair tables ────────────
        try:

            docs = []
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    parts: list[str] = []

                    # Extract tables first and format as Markdown
                    tables = page.extract_tables()
                    table_bboxes = [t.bbox for t in page.find_tables()] if tables else []

                    for table in tables:
                        if not table:
                            continue
                        header, *rows = table
                        header = [c or "" for c in header]
                        md_table = (
                            "| " + " | ".join(header) + " |\n"
                            + "| " + " | ".join(["---"] * len(header)) + " |\n"
                        )
                        for row in rows:
                            row = [str(c or "") for c in row]
                            md_table += "| " + " | ".join(row) + " |\n"
                        parts.append(md_table)

                    # Extract remaining text (outside table bounding boxes)
                    words = page.extract_words(x_tolerance=3, y_tolerance=3)
                    if table_bboxes and words:
                        def outside_tables(w):
                            x0, y0, x1, y1 = w["x0"], w["top"], w["x1"], w["bottom"]
                            for bx0, by0, bx1, by1 in table_bboxes:
                                if x0 >= bx0 and y0 >= by0 and x1 <= bx1 and y1 <= by1:
                                    return False
                            return True
                        words = [w for w in words if outside_tables(w)]
                    body = " ".join(w["text"] for w in words) if words else (page.extract_text() or "")
                    if body.strip():
                        parts.append(body)

                    combined = "\n\n".join(parts).strip()
                    if combined:
                        docs.append(
                            Document(
                                page_content=combined,
                                metadata={"page": i + 1, "content_type": "text+table"},
                            )
                        )

            if docs:
                return docs
        except Exception as e:
            print(f"[pdfplumber] failed for {file_path}: {e}")

        # ── Tier 3: Unstructured OCR – for scanned / image-only PDFs ─────
        try:
            docs = UnstructuredPDFLoader(
                file_path,
                strategy="hi_res",          # triggers OCR via tesseract
                infer_table_structure=True,  # recovers tables via OCR
                mode="elements",
            ).load()

            # Merge elements back into per-page Documents
            page_map: dict[int, list[str]] = {}
            for doc in docs:
                pg = doc.metadata.get("page_number", 1)
                page_map.setdefault(pg, []).append(doc.page_content)

            merged = [
                Document(
                    page_content="\n\n".join(texts),
                    metadata={"page": pg, "content_type": "ocr"},
                )
                for pg, texts in sorted(page_map.items())
            ]
            if merged:
                return merged
        except Exception as e:
            print(f"[unstructured-ocr] failed for {file_path}: {e}")

        print(f"⚠️  All PDF loaders failed for: {file_path}")
        return []

    # ------------------------------------------------------------------
    # File loading dispatcher
    # ------------------------------------------------------------------

    def _load_pages(self, file_path: str) -> List[Document]:
        ext = self._ext(file_path)
        print(file_path)
        if ext == "pdf":
            return self._load_pdf_pages(file_path)
        if ext == "docx":
            try:
                return Docx2txtLoader(file_path).load()
            except Exception:
                return []
        try:
            return TextLoader(file_path).load()
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Chunking  (unchanged logic, keeps your metadata structure)
    # ------------------------------------------------------------------

    def _create_chunks(self, metadata_json: str) -> List[Document]:
        with open(metadata_json, "r", encoding="utf-8") as fh:
            metadata = json.load(fh)

        # Larger overlap helps when tables are split across chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            # Keep Markdown table rows together where possible
            separators=["\n\n", "\n| ", "\n", " ", ""],
        )
        chunks: List[Document] = []

        for file_path, structure in metadata.items():
            print("kahlawi")
            pages = self._load_pages(file_path)
            if not pages:
                continue

            for chap in structure.get("chapters", []):
                try:
                    start = max(0, chap["start_page"] - 1)
                    end = min(len(pages), chap["end_page"])
                    chapter_pages = pages[start:end]
                except Exception:
                    continue

                # Preserve content_type from whichever loader succeeded
                content_type = chapter_pages[0].metadata.get("content_type", "unknown") if chapter_pages else "unknown"
                text = "\n\n".join(p.page_content for p in chapter_pages)

                for chunk in splitter.split_text(text):
                    chunks.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "title": structure.get("title", ""),
                                "source": structure.get("source", "general"),
                                "chapter": chap.get("chapter_name", ""),
                                "page_range": f"{chap.get('start_page')}-{chap.get('end_page')}",
                                "source_path": file_path,
                                "content_type": content_type,   # ← new: text+table / markdown / ocr
                            },
                        )
                    )

        print(f"Total chunks created: {len(chunks)}")
        return chunks

    # ------------------------------------------------------------------
    # FAISS build (unchanged)
    # ------------------------------------------------------------------

    def _build_faiss(self, docs: List[Document], batch_size: int = 30) -> FAISS:
        if not docs:
            raise ValueError("No documents to index.")

        vector_db = None
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            for attempt in range(5):
                try:
                    if vector_db is None:
                        vector_db = FAISS.from_documents(batch, self.embeddings)
                    else:
                        vector_db.add_documents(batch)
                    time.sleep(0.5)
                    break
                except Exception as exc:
                    if attempt == 4:
                        raise RuntimeError(f"FAISS failed at batch {i}: {exc}")
                    time.sleep(2**attempt + random.random())

        return vector_db

    # ------------------------------------------------------------------
    # Public API (unchanged)
    # ------------------------------------------------------------------

    def create_database(self, metadata_json: str, persist_dir: str = "faiss_index") -> FAISS:
        chunks = self._create_chunks(metadata_json)
        vector_db = self._build_faiss(chunks)
        os.makedirs(persist_dir, exist_ok=True)
        vector_db.save_local(persist_dir)
        print(f"✅ FAISS index saved to {persist_dir} ({vector_db.index.ntotal} vectors)")
        return vector_db