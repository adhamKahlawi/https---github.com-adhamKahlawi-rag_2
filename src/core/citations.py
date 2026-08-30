from typing import List, Tuple

def format_academic_response(body: str, sources: List[str]) -> str:
    unique = list(dict.fromkeys(s.strip() for s in sources if s.strip()))
    if not unique: return body.strip()
    # The formatter owns numbering; callers can use [1], [2] in generated body.
    references = "\n".join(f"[{i}] {source}" for i, source in enumerate(unique, 1))
    return f"{body.strip()}\n\nReferences\n\n{references}"
