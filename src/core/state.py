import hashlib
import os
from datetime import datetime, timezone
from dataclasses import asdict
from typing import Dict
import yaml
from .schemas import IngestionRecord

class YAMLIngestionState:
    def __init__(self, path="vector_database/ingestion_state.yaml"):
        self.path = path
        self.records: Dict[str, dict] = {}
        self.load()

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                self.records = (yaml.safe_load(f) or {}).get("documents", {})

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.safe_dump({"documents": self.records}, f, sort_keys=True, allow_unicode=True)
        os.replace(tmp, self.path)

    @staticmethod
    def fingerprint(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()

    def needs_processing(self, path: str, parser_version="1", chunking_version="1") -> bool:
        key = os.path.abspath(path)
        old = self.records.get(key)
        return not old or old.get("content_hash") != self.fingerprint(path) or old.get("parser_version") != parser_version or old.get("chunking_version") != chunking_version or old.get("status") == "failed"

    def mark(self, record: IngestionRecord):
        record.indexed_at = datetime.now(timezone.utc).isoformat()
        self.records[os.path.abspath(record.source_path)] = asdict(record)
        self.save()
