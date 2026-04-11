"""
build_vector_db.py
Load documents described in the metadata JSON, chunk them, and build a FAISS
vector database using LiteLLM-compatible Vertex AI embeddings.
"""

import os
import json
import time
import random
from typing import List

import vertexai
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    PDFPlumberLoader,
    UnstructuredPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorDBBuilder:
    def __init__(self, project_id: str):
        vertexai.init(project=project_id)
        self.embeddings = VertexAIEmbeddings(
            model_name="text-embedding-004",
            project=project_id,
        )

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    @staticmethod
    def _ext(file_path: str) -> str:
        return os.path.splitext(file_path)[1].lstrip(".").lower()

    def _load_pages(self, file_path: str) -> List[Document]:
        ext = self._ext(file_path)

        if ext == "pdf":
            for Loader in [PyPDFLoader, PDFPlumberLoader]:
                try:
                    return Loader(file_path).load()
                except Exception:
                    pass
            try:
                return UnstructuredPDFLoader(file_path, strategy="ocr_only").load()
            except Exception:
                return []

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
    # Chunking
    # ------------------------------------------------------------------

    def _create_chunks(self, metadata_json: str) -> List[Document]:
        with open(metadata_json, "r", encoding="utf-8") as fh:
            metadata = json.load(fh)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks: List[Document] = []

        for file_path, structure in metadata.items():
            pages = self._load_pages(file_path)
            if not pages:
                continue

            for chap in structure.get("chapters", []):
                try:
                    start = max(0, chap["start_page"] - 1)
                    end = min(len(pages), chap["end_page"])
                    text = "\n".join(p.page_content for p in pages[start:end])
                except Exception:
                    continue

                for chunk in splitter.split_text(text):
                    chunks.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "title": structure.get("title", ""),
                                "source": structure.get("source", "general"),
                                "chapter": chap.get("chapter_name", ""),
                                "page_range": (
                                    f"{chap.get('start_page')}-{chap.get('end_page')}"
                                ),
                                "source_path": file_path,
                                
                            },
                        )
                    )

        print(f"Total chunks created: {len(chunks)}")
        return chunks

    # ------------------------------------------------------------------
    # FAISS build (with retry + back-off)
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
    # Public API
    # ------------------------------------------------------------------

    def create_database(self, metadata_json: str, persist_dir: str = "faiss_index") -> FAISS:
        chunks = self._create_chunks(metadata_json)
        vector_db = self._build_faiss(chunks)

        os.makedirs(persist_dir, exist_ok=True)
        vector_db.save_local(persist_dir)
        print(f"✅ FAISS index saved to {persist_dir} ({vector_db.index.ntotal} vectors)")
        return vector_db
