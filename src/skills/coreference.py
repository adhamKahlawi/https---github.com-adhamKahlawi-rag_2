class CoreferenceSkill:
    FOLLOW_UP = {"spiegalo", "spiegami", "come si esegue", "approfondisci", "e poi", "quindi"}
    def is_follow_up(self, query: str) -> bool:
        q = query.lower().strip()
        return any(q == x or q.startswith(x + " ") for x in self.FOLLOW_UP) or len(q.split()) <= 5 and any(x in q for x in ("questo", "quello", "sopra", "precedente"))
