"""Migration facade: session-aware, skill-oriented RAG orchestration.

The legacy GeminiRAGSystem remains available for compatibility. New callers can
use this facade while ADK integration is introduced behind the same skill ports.
"""
from typing import Optional
from src.core.session import SessionStore
from src.skills.coreference import CoreferenceSkill
from src.skills.routing import QueryIntentSkill
from src.chat_llm import GeminiRAGSystem

class RAGCoordinator:
    def __init__(self, *, session_id: str = "default", session_store: Optional[SessionStore] = None, **legacy_kwargs):
        self.engine = GeminiRAGSystem(**legacy_kwargs)
        self.sessions = session_store or SessionStore()
        self.session = self.sessions.load(session_id)
        self.coreference = CoreferenceSkill()
        self.intent = QueryIntentSkill()

    def ask(self, query: str, k: int = 16) -> str:
        # Only inject history for likely follow-ups. The legacy engine's history
        # is deliberately populated from the persisted session, not shared globally.
        self.engine._history = list(self.session.messages) if self.coreference.is_follow_up(query) else []
        answer = self.engine.ask(query, k=k)
        self.session.messages.extend([
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer},
        ])
        self.session.messages = self.session.messages[-20:]
        self.sessions.save(self.session)
        return answer

    def reset(self):
        self.session.messages = []
        self.engine.reset()
        self.sessions.save(self.session)
