import json
from pathlib import Path
from src.core.state import YAMLIngestionState
from src.core.citations import format_academic_response
from src.skills.coreference import CoreferenceSkill
from src.skills.routing import QueryIntentSkill

def test_state_tracks_hash_and_skips_unchanged(tmp_path):
    document = tmp_path / "a.txt"
    document.write_text("hello", encoding="utf-8")
    state = YAMLIngestionState(str(tmp_path / "state.yaml"))
    assert state.needs_processing(str(document))
    from src.core.schemas import IngestionRecord
    state.mark(IngestionRecord("a", "A", str(document), "general", state.fingerprint(str(document)), status="metadata_ready"))
    state = YAMLIngestionState(str(tmp_path / "state.yaml"))
    assert not state.needs_processing(str(document))
    document.write_text("changed", encoding="utf-8")
    assert state.needs_processing(str(document))

def test_skills_and_citations():
    assert QueryIntentSkill().classify("Come uso MFE?") == "MANUAL"
    assert CoreferenceSkill().is_follow_up("spiegalo meglio")
    output = format_academic_response("Risposta [1]", ["Manuale, p. 4"])
    assert "References" in output and "[1] Manuale, p. 4" in output
