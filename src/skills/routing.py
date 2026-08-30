class QueryIntentSkill:
    def __init__(self, llm=None): self.llm = llm
    def classify(self, query: str) -> str:
        q = query.lower()
        if any(x in q for x in ("riassumi", "riassunto", "sintesi", "overview")): return "SUMMARY"
        if any(x in q for x in ("come uso", "come si usa", "configur", "procedura", "passaggi", "manuale", "mfe")): return "MANUAL"
        return "QA"

class RetrievalRoutingSkill:
    def source_for(self, intent: str, explicit_source: str = "") -> str:
        return explicit_source or ("manuale" if intent == "MANUAL" else "citazione" if intent in {"QA", "SUMMARY"} else "general")
