"""
build_doc_metadata.py
Analyse documents and extract structured metadata (title, pages, chapters).
Uses LiteLLM so the backend model can be swapped without code changes.
"""

import os
import re
import json
from typing import Dict, List, Tuple, Optional

import litellm
import pdfplumber
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader


CATEGORIES = ["general", "citazione", "manuale"]

# Patterns that strongly indicate a Table of Contents page
_TOC_PATTERNS = re.compile(
    r"(indice|sommario|table\s+of\s+contents|contents|indice\s+generale"
    r"|indice\s+dei\s+contenuti|tabella\s+dei\s+contenuti)",
    re.IGNORECASE,
)
# TOC pages typically have many lines ending with page numbers (dots/spaces then digits)
_TOC_LINE_PATTERN = re.compile(r"\.{3,}\s*\d+\s*$|…\s*\d+\s*$|\s{3,}\d+\s*$", re.MULTILINE)


def _is_toc_page(text: str) -> bool:
    """Return True if the page looks like a Table of Contents."""
    if not text:
        return False
    # Heading says "indice" / "contents" etc.
    if _TOC_PATTERNS.search(text[:200]):
        return True
    # Many lines ending with a page-number (dot leaders)
    matches = _TOC_LINE_PATTERN.findall(text)
    total_lines = max(text.count("\n"), 1)
    return len(matches) / total_lines > 0.25   # >25% of lines look like TOC entries

response_schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "pages": {"type": "integer"},
        "chapters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chapter_name": {"type": "string"},
                    "start_page": {"type": "integer"},
                    "end_page": {"type": "integer"}
                },
                "required": ["chapter_name", "start_page", "end_page"],
                "additionalProperties": False # Recommended for strictness
            }
        }
    },
    "required": ["title", "pages", "chapters"],
    "additionalProperties": False
}
EXTRACTION_PROMPT = """
Act as a Document Analysis Expert. Analyse the provided document and extract its
structural metadata as valid JSON.

The document text below uses explicit page markers:
  === PAGE N ===
and occasionally:
  [TOC PAGE N: table of contents - skip for chapter detection]

### Rules:
1. **Title** - main title from the first 1-2 pages (largest/centred heading).
2. **Total Pages** - use the IMPORTANT note at the bottom, not the last marker you see.
3. **Chapter detection** - find headings by:
   - Numbering: "1.", "1 ", "Capitolo 1", "Chapter 1", "I.", "1.0"
   - ALL-CAPS short lines standing alone on a page
   - Short bold-looking lines immediately after a === PAGE N === marker
4. **CRITICAL - Do NOT use TOC pages**: Pages marked [TOC PAGE N] list chapter names
   with page numbers, but those are NOT where the chapters actually start. Ignore them
   entirely. Only use === PAGE N === markers for start_page and end_page values.
5. **start_page** - the === PAGE N === where the chapter heading physically appears.
6. **end_page** - the page before the next chapter; last chapter uses total pages.

### Output (valid JSON only, no markdown fences):
{{
  "title": "...",
  "pages": <int>,
  "chapters": [
    {{"chapter_name": "...", "start_page": <int>, "end_page": <int>}}
  ]
}}

=== DOCUMENT CONTENT ===
{content}

{page_count_hint}
"""


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

def _true_page_count(file_path: str) -> int:
    """Read page count directly from PDF structure (works on scanned PDFs)."""
    try:
        with pdfplumber.open(file_path) as pdf:
            return len(pdf.pages)
    except Exception:
        pass
    try:
        from pypdf import PdfReader
        return len(PdfReader(file_path).pages)
    except Exception:
        pass
    return 0


def _extract_pages(file_path: str) -> Dict[int, str]:
    """Return {1-based page number: extracted text}."""
    pages: Dict[int, str] = {}
    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                pages[i] = (page.extract_text(x_tolerance=2, y_tolerance=2) or "").strip()
        if pages:
            return pages
    except Exception as e:
        print(f"  [pdfplumber] failed: {e} - trying PyPDFLoader")
    try:
        for i, doc in enumerate(PyPDFLoader(file_path).load(), start=1):
            pages[i] = doc.page_content.strip()
        if pages:
            return pages
    except Exception as e:
        print(f"  [PyPDFLoader] failed: {e}")
    return {}


def _build_pdf_content(file_path: str, max_chars: int = 120_000) -> Tuple[str, int]:
    """
    Build page-labelled text for the LLM prompt.

    - TOC pages are labelled [TOC PAGE N] so the LLM knows to skip them
      for chapter detection.
    - When content exceeds max_chars: always keeps first 20 pages (covers
      title + front matter + start of real content), then samples the rest
      evenly so chapter headings throughout the document are all visible.
    - Skipped ranges marked [PAGES A-B: omitted] so LLM knows doc continues.

    Returns (structured_text, true_page_count).
    """
    total_pages = _true_page_count(file_path)
    page_texts  = _extract_pages(file_path)

    if not page_texts:
        return "", total_pages

    toc_pages = {n for n, t in page_texts.items() if _is_toc_page(t)}
    if toc_pages:
        print(f"  [toc] detected TOC on pages: {sorted(toc_pages)}")

    def block(n: int) -> str:
        text = page_texts.get(n) or "[no extractable text]"
        if n in toc_pages:
            return f"[TOC PAGE {n}: table of contents - skip for chapter detection]\n{text}"
        return f"=== PAGE {n} ===\n{text}"

    all_pages    = sorted(page_texts)
    full_content = "\n\n".join(block(n) for n in all_pages)

    if len(full_content) <= max_chars:
        return full_content, total_pages

    # Smart sampling: keep first 20 pages + evenly distributed remainder
    ALWAYS_FIRST = 20
    first_pages  = [n for n in all_pages if n <= ALWAYS_FIRST]
    remain_pages = [n for n in all_pages if n > ALWAYS_FIRST]
    first_text   = "\n\n".join(block(n) for n in first_pages)
    budget       = max_chars - len(first_text)
    avg_len      = len(full_content) / max(len(all_pages), 1)
    extra_slots  = max(0, int(budget / max(avg_len, 1)))

    if extra_slots >= len(remain_pages):
        sampled = remain_pages
    elif extra_slots == 0:
        sampled = []
    else:
        step    = len(remain_pages) / extra_slots
        sampled = [remain_pages[int(i * step)] for i in range(extra_slots)]

    included = sorted(set(first_pages + sampled))
    blocks: List[str] = []
    prev = 0
    for n in included:
        if n - prev > 1:
            blocks.append(f"[PAGES {prev+1}-{n-1}: omitted to fit token limit]")
        blocks.append(block(n))
        prev = n
    if total_pages > prev:
        blocks.append(f"[PAGES {prev+1}-{total_pages}: not shown]")

    result = "\n\n".join(blocks)
    print(f"  [sampling] {len(included)}/{len(all_pages)} pages shown, "
          f"{len(toc_pages)} TOC pages labelled, true total={total_pages}")
    return result, total_pages


# ---------------------------------------------------------------------------
# MetadataGenerator
# ---------------------------------------------------------------------------

class MetadataGenerator:
    def __init__(
        self,
        project_id: str,
        model: str    = "vertex_ai/gemini-2.0-flash-001",
        location: str = "us-central1",
    ):
        self.project_id = project_id
        self.model      = model
        os.environ["VERTEXAI_PROJECT"]  = project_id
        os.environ["VERTEXAI_LOCATION"] = location

    @staticmethod
    def _ext(path: str) -> str:
        return os.path.splitext(path)[1].lstrip(".").lower()

    def _get_content(self, file_path: str) -> Tuple[str, Optional[int]]:
        """Return (text_for_prompt, true_page_count_or_None)."""
        ext = self._ext(file_path)
        if ext == "pdf":
            text, pages = _build_pdf_content(file_path)
            if not text:
                raise ValueError("PDF text extraction returned empty string")
            return text, pages
        if ext == "docx":
            docs = Docx2txtLoader(file_path).load()
            return "\n".join(d.page_content for d in docs), None
        if ext in {"txt", "md", "csv", "json", "html"}:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
                return fh.read(), None
        raise ValueError(f"Unsupported file type: .{ext}")

    def _call_llm(self, content: str, true_pages: Optional[int]) -> Dict:
        hint = (
            f"IMPORTANT: This PDF has exactly {true_pages} pages in total. "
            f"You MUST set the \"pages\" field to {true_pages} in your JSON output."
            if true_pages else ""
        )
        prompt = EXTRACTION_PROMPT.format(content=content, page_count_hint=hint)

        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "book_summary", # A unique name is required by some providers
                    "strict": True,          # Forces the model to follow the schema exactly
                    "schema": response_schema
                }
            },
        )
        raw = response.choices[0].message.content
        print(f"  [llm] {raw[:300]}")

        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        return json.loads(clean.strip())

    def _analyse(self, file_path: str) -> Dict:
        name = os.path.basename(file_path)

        try:
            content, true_pages = self._get_content(file_path)
        except Exception as exc:
            print(f"  [FAIL] text extraction: {exc}")
            return {"title": name, "pages": 0, "chapters": []}

        if not content.strip():
            print(f"  [FAIL] empty text")
            return {"title": name, "pages": 0, "chapters": []}

        print(f"  [ok] {len(content):,} chars | true_pages={true_pages} | "
              f"markers={content.count('=== PAGE')}")

        try:
            result = self._call_llm(content, true_pages)
        except Exception as exc:
            print(f"  [FAIL] LLM: {type(exc).__name__}: {exc}")
            return {"title": name, "pages": 0, "chapters": []}

        if not result.get("title") or not result.get("chapters"):
            print(f"  [FAIL] incomplete result: {result}")
            return {"title": name, "pages": 0, "chapters": []}

        # Always trust the file's true page count over the LLM's guess
        if true_pages:
            result["pages"] = true_pages

        print(f"  [done] title='{result['title']}' | pages={result['pages']} | "
              f"chapters={len(result['chapters'])}")
        return result

    def generate_metadata(self, root_path: str, output_json: str) -> None:
        """
        Walk every category sub-folder in root_path, analyse new files,
        and save results to output_json. Already-cached files are skipped.
        Progress is saved after every file so crashes don't lose work.
        """
        metadata: Dict = {}
        if os.path.exists(output_json):
            try:
                with open(output_json, "r", encoding="utf-8") as fh:
                    metadata = json.load(fh)
                print(f"Loaded existing metadata ({len(metadata)} entries).")
            except json.JSONDecodeError:
                print("Warning: metadata file corrupt - starting fresh.")

        def _save():
            out_dir = os.path.dirname(output_json)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(output_json, "w", encoding="utf-8") as fh:
                json.dump(metadata, fh, indent=2, ensure_ascii=False)

        for category in CATEGORIES:
            folder = os.path.join(root_path, category)
            if not os.path.isdir(folder):
                continue
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if file_path in metadata:
                    print(f"Skip (cached): {filename}")
                    continue
                print(f"\nAnalysing [{category}]: {filename}")
                result = self._analyse(file_path)
                metadata[file_path] = {**result, "source": category}
                _save()

        print(f"\nMetadata saved to {output_json} ({len(metadata)} documents)")