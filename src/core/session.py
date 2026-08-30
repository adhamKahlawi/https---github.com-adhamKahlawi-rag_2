import json, os
from .schemas import Session

class SessionStore:
    def __init__(self, directory="vector_database/sessions"):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
    def load(self, session_id: str) -> Session:
        path = os.path.join(self.directory, f"{session_id}.json")
        if not os.path.exists(path): return Session(session_id)
        with open(path, encoding="utf-8") as f: return Session(**json.load(f))
    def save(self, session: Session):
        path = os.path.join(self.directory, f"{session.session_id}.json")
        with open(path, "w", encoding="utf-8") as f: json.dump(session.__dict__, f, ensure_ascii=False, indent=2)
