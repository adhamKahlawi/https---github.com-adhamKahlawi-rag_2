from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class DocumentMetadata:
    document_id: str
    title: str = ""
    source: str = "general"
    source_path: str = ""
    document_type: str = ""
    section_path: List[str] = field(default_factory=list)
    chapter: str = ""
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    content_type: str = "text"
    language: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        data = self.__dict__.copy()
        data["page_range"] = f"{self.page_start}-{self.page_end}" if self.page_start else ""
        data.update(self.extra)
        return data

@dataclass
class Session:
    session_id: str
    messages: List[Dict[str, str]] = field(default_factory=list)
    summary: str = ""

@dataclass
class IngestionRecord:
    document_id: str
    title: str
    source_path: str
    category: str
    content_hash: str
    status: str = "pending"
    parser_version: str = "1"
    chunking_version: str = "1"
    embedding_model: str = ""
    indexed_at: str = ""
    error: str = ""
