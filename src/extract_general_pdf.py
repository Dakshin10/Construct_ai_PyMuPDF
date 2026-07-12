"""
General-Purpose Drawing Extractor
MinoriLabs — Salit Steel Agent Practice

Extracts text and dimension data from ANY drawing PDF that is not handled by
the structural rebar parser (architectural, mechanical, civil-site, ...).
Also provides structure-fact and element parsers that are reusable on text
extracted from structural PDFs.

Usage:
    python extract_general_pdf.py <path_to_pdf>
"""

import subprocess, sys

def _ensure_deps():
    try:
        __import__("fitz")
    except ImportError:
        print("Installing pymupdf...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pymupdf",
                               "--break-system-packages", "-q"])

_ensure_deps()

import fitz  # PyMuPDF
import re
import json
import math
from collections import defaultdict


# ── Garbled-text heuristic (Issue 4) ────────────────────────────────────────────
# Custom CAD font subsets decode to control-char / ligature garbage in PyMuPDF
# (e.g. "\x15\x10\x036*'5'..."). Legit vector text never carries ASCII control
# chars after .strip(), so their presence is a reliable discriminator. We flag &
# skip such spans (routing them to Unmapped Text) rather than trying to fix the
# encoding — general-notes boilerplate is not a target extraction field.

def is_garbled(text):
    """True if a text span is corrupted CAD-font output, not real prose."""
    if not text:
        return False
    # Any ASCII control char (except tab/newline) => corrupted glyph mapping.
    if any(ord(c) < 32 and c not in "\t\n\r" for c in text):
        return True
    if len(text) < 12:
        return False
    # Fallback: very high ratio of non-standard chars in a long span.
    allowed = " .,-/()'\"@#&:;%°Øø±²³½¼–—’‘“”·"
    bad = sum(1 for c in text if not (c.isalnum() or c in allowed))
    return bad / len(text) >= 0.45


def split_garbled(lines):
    """Partition lines into (clean, unmapped_garbled)."""
    clean, unmapped = [], []
    for ln in lines:
        (unmapped if is_garbled(ln) else clean).append(ln)
    return clean, unmapped


# ── Document classification (Issue 1) ────────────────────────────────────────────
# Decide whether a PDF is rebar/structural-parseable before choosing a parser.

_STRUCT_INDICATORS = [
    re.compile(r"\b\d{2}M\b.{0,25}@\s*\d", re.I),          # "20M ... @ 250"
    re.compile(r"@\s*\d+.{0,25}\b\d{2}M\b", re.I),          # "@ 250 ... 20M"
    re.compile(r"\b\d+\s*[-]\s*\d{2}M\b", re.I),            # "16-20M"
    re.compile(r"\b\d{2}M\s+(?:BARS?|VERT|TIES|DOWELS)", re.I),
    re.compile(r"\bTIES?\s*@\s*\d", re.I),                  # "TIES @ 300"
    re.compile(r"REBAR\s+SCHEDULE|BAR\s+SCHEDULE|REINFORC(?:ING|EMENT)", re.I),
    re.compile(r"(?:PIER|FOOTING|COLUMN|BASE\s*PLATE|LINTEL|MASONRY\s+WALL)\s+SCHEDULE", re.I),
]


def classify_pdf(full_text, threshold=2):
    """
    Return ("structural", score) if the text shows rebar-schedule indicators,
    else ("general", score). doc_type stays an open label — this only decides
    the *parser route*, not a fixed enum.
    """
    score = sum(1 for p in _STRUCT_INDICATORS if p.search(full_text))
    return ("structural" if score >= threshold else "general"), score


# ── Pattern tables ────────────────────────────────────────────────────────────

MATERIAL_PATTERNS = [
    (re.compile(r"REINFORCED\s+CONCRETE|CAST[- ]IN[- ]PLACE|\bCONCRETE\b|\bCONC\.", re.I), "concrete"),
    (re.compile(r"STRUCTURAL\s+STEEL|STEEL\s+FRAME", re.I), "steel"),
    (re.compile(r"\bTIMBER\b|\bWOOD\s+FRAME\b", re.I), "timber"),
    (re.compile(r"\bMASONRY\b|\bCMU\b|CONCRETE\s+BLOCK", re.I), "masonry"),
]

STOREY_PATTERNS = [
    re.compile(r"(\d+)\s*[- ]?\s*STOR(?:E?Y|IES)", re.I),
    re.compile(r"(\d+)\s*[- ]?\s*FLOORS?\b", re.I),
    re.compile(r"NUMBER\s+OF\s+STOREYS?\s*[:=]?\s*(\d+)", re.I),
]

LEVEL_PATTERN = re.compile(r"\bLEVEL\s+(\d+)\b|\b(\d+)(?:ST|ND|RD|TH)\s+FLOOR\b", re.I)

# e.g. "24.5 m x 30.2 m", "80'-0\" x 120'-0\"", "AREA: 2400 SQ.M", "GFA 2,400 m2"
AREA_PATTERNS = [
    re.compile(r"(\d[\d,.]*)\s*(?:m|M)\s*[xX×]\s*(\d[\d,.]*)\s*(?:m|M)\b"),
    re.compile(r"(?:TOTAL\s+)?(?:GROSS\s+)?(?:FLOOR\s+)?AREA\s*[:=]?\s*([\d,.]+)\s*(?:SQ\.?\s?M|m2|m²|SQ\.?\s?FT)", re.I),
    re.compile(r"(\d+)'\s*-?\s*\d*\"?\s*[xX×]\s*(\d+)'\s*-?\s*\d*\"?"),
]

TYPE_PATTERNS = {
    "foundation_type": re.compile(
        r"(STRIP|SPREAD|PAD|RAFT|MAT|PILE[D]?|CAISSON|SLAB[- ]ON[- ]GRADE)"
        r"\s+(?:FOOTINGS?|FOUNDATIONS?)", re.I),
    "beam_type": re.compile(
        r"(CONCRETE|STEEL|PRECAST|COMPOSITE|GRADE|TIE|DROP)\s+BEAMS?", re.I),
    "column_type": re.compile(
        r"(CONCRETE|STEEL|PRECAST|CIRCULAR|SQUARE|RECTANGULAR)\s+COLUMNS?", re.I),
    "roof_type": re.compile(
        r"(FLAT|SLOPED|PITCHED|METAL\s+DECK|CONCRETE\s+SLAB|TRUSS(?:ED)?)\s+ROOF"
        r"|ROOF\s+TYPE\s*[:=]?\s*(\w+)", re.I),
}

# structural elements + attributes
ELEMENT_KEYWORDS = {
    "footing": re.compile(r"\bFOOTINGS?\b|\bFTG\.?\b", re.I),
    "beam":    re.compile(r"\bBEAMS?\b(?!\s+POCKET)", re.I),
    "column":  re.compile(r"\bCOLUMNS?\b|\bCOL\.?\b", re.I),
    "pier":    re.compile(r"\bPIERS?\b", re.I),
    "slab":    re.compile(r"\bSLABS?\b|\bS\.O\.G\.?\b", re.I),
}

ATTR_PATTERNS = {
    "spacing":      re.compile(r"@\s*(\d+)\s*(?:mm|MM|O\.?C\.?)?|SPACING\s*[:=]?\s*(\d+)", re.I),
    "cover":        re.compile(r"COVER\s*[:=]?\s*(\d+)\s*(?:mm)?", re.I),
    "stock_length": re.compile(r"STOCK\s+LENGTH\s*[:=]?\s*([\d.]+\s*(?:mm|m)?)", re.I),
    "lap_length":   re.compile(r"LAP(?:\s+LENGTH|\s+SPLICE)?\s*[:=]?\s*([\d.]+\s*(?:mm|m)?)", re.I),
    "hook_length":  re.compile(r"HOOK(?:\s+LENGTH)?\s*[:=]?\s*([\d.]+\s*(?:mm|m)?)", re.I),
    "stirrups":     re.compile(r"STIRRUPS?\s*[:=]?\s*(\d+M(?:\s*@\s*\d+)?)", re.I),
    "ties_type":    re.compile(r"TIES?\s*[:=]?\s*(\d+M)", re.I),
    "ties_length":  re.compile(r"TIES?\s+LENGTH\s*[:=]?\s*([\d.]+\s*(?:mm|m)?)", re.I),
    "num_rebars":   re.compile(r"(\d+)\s*[- ]\s*(\d+M)\b"),   # e.g. "6-20M"
    "size":         re.compile(r"(\d+)\s*(?:mm)?\s*[xX×]\s*(\d+)\s*(?:mm)?"),
    "count":        re.compile(r"(?:QTY|COUNT|NO\.?|TYP\.?\s+OF)\s*[:=]?\s*(\d+)", re.I),
    "shape_type":   re.compile(r"\b(STRAIGHT|BENT|L[- ]SHAPE[D]?|U[- ]SHAPE[D]?|HOOKED|CRANKED)\b", re.I),
}


def _first_group(match):
    return next((g for g in match.groups() if g), None)


# ── Schedule-table extraction (Issue 2) ──────────────────────────────────────────
# Structural schedules (PIER / FOOTING PAD / COLUMN / BASE PLATE ...) are drawn as
# CAD detail blocks: a short title/header span with size/reinforcing/ties callouts
# clustered around it. get_text() reading order scrambles these, so we work from
# span COORDINATES and assign each reinforcing callout to its nearest schedule
# anchor. This keeps schedule-sourced data (authoritative) separate from bare
# plan-view mark labels (which only point back to the schedule).

SCHEDULE_TITLE_PATTERNS = {
    "pier":         re.compile(r"PIER\s+SCHEDULE|^PIER(\s+(SIZE|PROFILE))?$", re.I),
    "footing":      re.compile(r"FOOTING(\s+PAD)?\s+SCHEDULE|NOMINAL\s+FOOTING\s+SIZE"
                               r"|^FOOTING(\s+(SIZE|PROFILE))?$", re.I),
    "column":       re.compile(r"COLUMN\s+SCHEDULE|^(HSS\s+)?COLUMN(\s+SIZE)?$", re.I),
    "base_plate":   re.compile(r"BASE\s*PLATE\s+SCHEDULE|BASE\s*PLATE\s+DIMENSIONS"
                               r"|^BASE\s*PLATE(\s+PROFILE)?$", re.I),
    "masonry_wall": re.compile(r"MASONRY\s+WALL\s+SCHEDULE", re.I),
    "lintel":       re.compile(r"LINTEL\s+SCHEDULE", re.I),
}

_RE_NUMBAR  = re.compile(r"(\d+)\s*[-]\s*(\d{2})M\b", re.I)       # "16-20M", "8 - 20M"
_RE_BARSIZE = re.compile(r"\b(\d{2})M\b", re.I)                    # "20M"
_RE_TIES    = re.compile(r"(\d{2})M\s*TIES?\s*@?\s*(\d+)?", re.I)  # "10M TIES @ 300"
_RE_SPACING = re.compile(r"@\s*(\d+)\s*(?:mm)?\s*c/c", re.I)       # "@ 250 c/c"
_RE_DIM     = re.compile(r"(\d{3,4})\s*[xX×]\s*(\d{3,4})(?:\s*[xX×]\s*(\d+))?")
_RE_MARK    = re.compile(r"^(P|F|C|BP|L)\d{1,2}[a-z]?$")           # P1, F5, C1, BP2


def _new_sched_elem(element_type):
    return {
        "element_type": element_type,
        "label": None,
        "marks": [],                 # schedule marks (P1, F1, ...) found in-region
        "size": None, "count": None, "num_rebars": None, "rebar_size": None,
        "shape_type": None, "stirrups": None, "spacing": None, "cover": None,
        "stock_length": None, "lap_length": None, "hook_length": None,
        "ties_type": None, "ties_length": None, "ties_spacing": None,
        "source": "schedule",        # provenance: from a schedule table, not plan view
        "confidence": "high",
        "raw_lines": [],
    }


def extract_schedules(spans, max_dist=280):
    """
    spans: list of {"page", "x", "y", "text"} with coordinates.
    Returns (elements, detected_titles):
      elements        — one aggregated record per (page, element_type) with reinf,
                        ties, size and schedule marks populated from real callouts.
      detected_titles — set of element_types whose schedule title appears.
    """
    by_page = defaultdict(list)
    for s in spans:
        by_page[s["page"]].append(s)

    elements = {}
    detected = set()

    for page, ps in by_page.items():
        anchors = defaultdict(list)
        for s in ps:
            t = s["text"].strip()
            if len(t) <= 30:
                for et, pat in SCHEDULE_TITLE_PATTERNS.items():
                    if pat.search(t):
                        anchors[et].append(s)
                        detected.add(et)
        if not anchors:
            continue

        for s in ps:
            t = s["text"].strip()
            up = t.upper()
            is_mark = bool(_RE_MARK.match(t))
            has_reinf = bool(
                _RE_NUMBAR.search(t) or _RE_TIES.search(t)
                or ("BARS" in up and _RE_BARSIZE.search(t))
                or ("VERT" in up and _RE_BARSIZE.search(t))
                or ("DOWEL" in up and _RE_BARSIZE.search(t))
            )
            if not (is_mark or has_reinf):
                continue

            best, bd = None, 1e9
            for et, alist in anchors.items():
                for a in alist:
                    d = math.hypot(s["x"] - a["x"], s["y"] - a["y"])
                    if d < bd:
                        bd, best = d, et
            if best is None or bd > max_dist:
                continue

            rec = elements.setdefault((page, best), _new_sched_elem(best))
            rec["raw_lines"].append(t)

            if is_mark:
                # A footing schedule shouldn't absorb a 'C1' column mark, etc.
                prefix = {"pier": "P", "footing": "F", "column": "C",
                          "base_plate": "BP", "lintel": "L"}.get(best)
                if prefix and t.upper().startswith(prefix) and t not in rec["marks"]:
                    rec["marks"].append(t)
                continue

            # A "N-NNM TIES" callout means N ties of size NNM — NOT N vertical
            # rebars. Only treat "N-NNM" as vertical reinforcing on non-tie lines,
            # so tie-notes don't fabricate a bogus rebar count/size.
            m = _RE_NUMBAR.search(t)
            if m and "TIES" not in up:
                rec["num_rebars"] = max(rec["num_rebars"] or 0, int(m.group(1)))
                rec["rebar_size"] = f"{int(m.group(2))}M"
            elif "TIES" not in up:
                mb = _RE_BARSIZE.search(t)
                if mb and rec["rebar_size"] is None:
                    rec["rebar_size"] = f"{int(mb.group(1))}M"

            mt = _RE_TIES.search(t)
            if mt:
                rec["ties_type"] = f"{int(mt.group(1))}M"
                if mt.group(2):
                    rec["ties_spacing"] = int(mt.group(2))

            ms = _RE_SPACING.search(t)
            if ms and rec["spacing"] is None:
                rec["spacing"] = int(ms.group(1))

            md = _RE_DIM.search(t)
            if md and rec["size"] is None:
                rec["size"] = "x".join(g for g in md.groups() if g)

    # Label each record with its marks, then drop non-usable records.
    #  - Vertical-steel elements (pier/column/footing) are DEFINED by a mark or
    #    vertical reinforcing. A record with neither — only a stray ties/spacing
    #    note (e.g. a "2-10M TIES @ TOP OF PIER" callout) — is a fragment, not a
    #    distinct element, and is filtered here rather than surfacing as a bogus
    #    row (or being silently dropped later by the LLM).
    #  - Other types (e.g. base_plate) legitimately carry only ties/anchor/
    #    spacing data, so ties/spacing counts as usable data for them.
    _VERTICAL_TYPES = {"pier", "column", "footing"}
    out = []
    for rec in elements.values():
        if rec["marks"]:
            rec["label"] = ",".join(sorted(set(rec["marks"])))
        has_identity = bool(rec["marks"])
        has_vertical = bool(rec["num_rebars"] or rec["rebar_size"] or rec["size"])
        if rec["element_type"] in _VERTICAL_TYPES:
            keep = has_identity or has_vertical
        else:
            keep = (has_identity or has_vertical
                    or rec["ties_type"] or rec["spacing"])
        if keep:
            out.append(rec)

    return out, sorted(detected)


# ── Free-text fact keywords (Issue 3) ────────────────────────────────────────────
_CONCRETE_STRENGTH = re.compile(r"(\d{2,3})\s*MPa", re.I)
# A MPa value is a CONCRETE strength only in concrete context, not steel yield /
# anchor-bolt / grout-strength / rebar-yield callouts.
_CONC_CTX  = re.compile(r"concrete|conc\.|footing|slab|sidewalk|\bpad\b|masonry|block|pier", re.I)
_CONC_EXCL = re.compile(r"anchor|bolt|yield|weld|A307|\brod\b|joist|\bsteel\b|grout|reinforc", re.I)

_FOUNDATION_KEYWORDS = [
    (re.compile(r"trench(?:ed)?\s+(?:&|and)?\s*poured|trench\s+footing", re.I),
     "trenched & poured strip footing"),
    (re.compile(r"pier\s*(?:&|and)\s*footing\s*pad", re.I), "pier and footing pad"),
    (re.compile(r"strip\s+footing", re.I), "strip footing"),
    (re.compile(r"slab[- ]on[- ]grade", re.I), "slab-on-grade"),
    (re.compile(r"spread\s+footing", re.I), "spread footing"),
]
_WALL_KEYWORDS = [
    (re.compile(r"\bCMU\b|concrete\s+block|masonry\s+(?:block|wall)|\bmasonry\b", re.I),
     "CMU/masonry block"),
    (re.compile(r"curtain\s+wall", re.I), "curtain wall"),
    (re.compile(r"metal\s+stud", re.I), "metal stud"),
]
_ROOF_KEYWORDS = [
    (re.compile(r"open\s+web\s+steel\s+joist|O\.?\s?W\.?\s?S\.?\s?J\.?", re.I),
     "open web steel joist"),
    (re.compile(r"metal\s+deck|steel\s+deck", re.I), "metal deck"),
]


def _context(full_text, match, width=45):
    """Return a trimmed sentence-ish context window around a regex match."""
    s = max(0, match.start() - width)
    e = min(len(full_text), match.end() + width)
    return re.sub(r"\s+", " ", full_text[s:e]).strip()


def _collect_keyword_facts(full_text, keyword_table):
    """Return (ordered unique labels, {label: source_context})."""
    labels, sources = [], {}
    for pat, label in keyword_table:
        m = pat.search(full_text)
        if m:
            if label not in labels:
                labels.append(label)
            sources[label] = _context(full_text, m)
    return labels, sources


def extract_structure_facts(full_text):
    """
    Best-effort structure-level facts from the full text of a drawing set.
    Adds (Issue 3) concrete strength, richer foundation/wall/roof detection, and a
    `fact_sources` map preserving the source context for each inferred fact, since
    these come from free text rather than a clean schedule.
    """
    facts = {
        "material": None, "storeys": None, "total_area": None,
        "foundation_type": None, "beam_type": None,
        "column_type": None, "roof_type": None,
        "concrete_strength": None, "wall_type": None,
        "structure_contents": [],
        "fact_sources": {},
    }

    for pat, label in MATERIAL_PATTERNS:
        if pat.search(full_text):
            facts["material"] = label
            break

    for pat in STOREY_PATTERNS:
        m = pat.search(full_text)
        if m:
            facts["storeys"] = int(m.group(1))
            break
    if facts["storeys"] is None:
        levels = [int(g) for m in LEVEL_PATTERN.finditer(full_text)
                  for g in m.groups() if g]
        if levels:
            facts["storeys"] = max(levels)

    for pat in AREA_PATTERNS:
        m = pat.search(full_text)
        if m:
            groups = [g for g in m.groups() if g]
            facts["total_area"] = " x ".join(groups) if len(groups) > 1 else groups[0]
            break

    for key, pat in TYPE_PATTERNS.items():
        m = pat.search(full_text)
        if m:
            facts[key] = _first_group(m)
            facts["fact_sources"][key] = _context(full_text, m)

    # Concrete strength(s) by use — keep each distinct MPa value + its context,
    # but only where the context is actually about concrete (not steel yield,
    # anchor bolts, or grout).
    strengths, strength_src = [], {}
    for m in _CONCRETE_STRENGTH.finditer(full_text):
        ctx = _context(full_text, m)
        if not _CONC_CTX.search(ctx) or _CONC_EXCL.search(ctx):
            continue
        val = f"{int(m.group(1))} MPa"
        if val not in strengths:
            strengths.append(val)
            strength_src[val] = ctx
    if strengths:
        facts["concrete_strength"] = "; ".join(strengths)
        facts["fact_sources"]["concrete_strength"] = " | ".join(
            f"{v}: {strength_src[v]}" for v in strengths)

    # Foundation type — richer keyword set; may override the TYPE_PATTERNS guess.
    found_labels, found_src = _collect_keyword_facts(full_text, _FOUNDATION_KEYWORDS)
    if found_labels:
        facts["foundation_type"] = "; ".join(found_labels)
        facts["fact_sources"]["foundation_type"] = " | ".join(
            f"{lbl}: {found_src[lbl]}" for lbl in found_labels)

    # Wall type
    wall_labels, wall_src = _collect_keyword_facts(full_text, _WALL_KEYWORDS)
    if wall_labels:
        facts["wall_type"] = "; ".join(wall_labels)
        facts["fact_sources"]["wall_type"] = " | ".join(
            f"{lbl}: {wall_src[lbl]}" for lbl in wall_labels)

    # Roof type — prefer explicit keywords, fall back to earlier TYPE_PATTERNS hit.
    roof_labels, roof_src = _collect_keyword_facts(full_text, _ROOF_KEYWORDS)
    if roof_labels:
        facts["roof_type"] = "; ".join(roof_labels)
        facts["fact_sources"]["roof_type"] = " | ".join(
            f"{lbl}: {roof_src[lbl]}" for lbl in roof_labels)

    for token, label in [("ELEVATED DECK", "elevated deck"), ("SITE PLAN", "site drawing"),
                         ("ELEVATION", "elevation views"), ("SECTION", "section views"),
                         ("FLOOR PLAN", "floor plans")]:
        if token in full_text.upper():
            facts["structure_contents"].append(label)

    return facts


def extract_elements(text_lines):
    """
    Best-effort element records from drawing text lines.
    Groups attribute annotations onto the most recent element keyword seen.
    Returns list of dicts: element_type, size, count, num_rebars, shape_type,
    stirrups, spacing, cover, stock_length, lap_length, hook_length,
    ties_type, ties_length, label.
    """
    elements = {}
    current = None

    for line in text_lines:
        for etype, pat in ELEMENT_KEYWORDS.items():
            if pat.search(line):
                # element label e.g. "FOOTING F1", "BEAM B2"
                lab = re.search(r"\b([A-Z]{1,3}\d{1,3})\b", line)
                key = f"{etype}:{lab.group(1) if lab else 'general'}"
                if key not in elements:
                    elements[key] = {
                        "element_type": etype,
                        "label": lab.group(1) if lab else None,
                        "size": None, "count": None, "num_rebars": None,
                        "rebar_size": None, "shape_type": None, "stirrups": None,
                        "spacing": None, "cover": None, "stock_length": None,
                        "lap_length": None, "hook_length": None,
                        "ties_type": None, "ties_length": None,
                        "raw_lines": [],
                    }
                current = key
                break

        if current is None:
            continue

        el = elements[current]
        el["raw_lines"].append(line)

        m = ATTR_PATTERNS["num_rebars"].search(line)
        if m and el["num_rebars"] is None:
            el["num_rebars"], el["rebar_size"] = int(m.group(1)), m.group(2)
        m = ATTR_PATTERNS["size"].search(line)
        if m and el["size"] is None and not ATTR_PATTERNS["num_rebars"].search(line):
            el["size"] = f"{m.group(1)}x{m.group(2)}"
        for attr in ("spacing", "cover", "stock_length", "lap_length", "hook_length",
                     "stirrups", "ties_type", "ties_length", "count", "shape_type"):
            if el[attr] is None:
                m = ATTR_PATTERNS[attr].search(line)
                if m:
                    val = _first_group(m)
                    el[attr] = val.strip() if isinstance(val, str) else val

    # Drop unlabeled elements with no attribute data at all — these come from
    # incidental keyword mentions (titles, general notes), not real elements.
    def _has_data(el):
        attrs = ("size", "count", "num_rebars", "rebar_size", "shape_type",
                 "stirrups", "spacing", "cover", "stock_length", "lap_length",
                 "hook_length", "ties_type", "ties_length")
        return el["label"] is not None or any(el[a] not in (None, "") for a in attrs)

    return [el for el in elements.values() if _has_data(el)]


def page_spans(page, page_num):
    """Span-level extraction with coordinates for a single page."""
    out = []
    for b in page.get_text("dict")["blocks"]:
        if b["type"] != 0:
            continue
        for l in b["lines"]:
            for s in l["spans"]:
                t = s["text"].strip()
                if t:
                    out.append({"page": page_num, "x": s["origin"][0],
                                "y": s["origin"][1], "text": t})
    return out


def process_pdf(pdf_path):
    """Extract text, dimensions, structure facts, and elements from any drawing PDF."""
    doc = fitz.open(pdf_path)
    per_page, all_lines, all_spans = [], [], []

    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text("text")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        all_lines.extend(lines)
        all_spans.extend(page_spans(page, page_num + 1))
        per_page.append({"page": page_num + 1, "line_count": len(lines), "lines": lines})

    # Issue 4: keep garbled CAD-font boilerplate out of structured extraction.
    clean_lines, unmapped_text = split_garbled(all_lines)
    full_text = "\n".join(clean_lines)
    dims = sorted(set(re.findall(r"\b\d{3,5}\s*mm\b|\b\d+'\s*-\s*\d+\"", full_text)))

    schedule_elements, schedules_detected = extract_schedules(all_spans)

    result = {
        "file": pdf_path,
        "pages": doc.page_count,
        "per_page": per_page,
        "dimensions": dims,
        "structure_facts": extract_structure_facts(full_text),
        "elements": extract_elements(clean_lines),
        "schedule_elements": schedule_elements,
        "schedules_detected": schedules_detected,
        "unmapped_text": unmapped_text,
    }
    doc.close()
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_general_pdf.py <pdf_path>")
        sys.exit(1)
    out = process_pdf(sys.argv[1])
    print(json.dumps({k: v for k, v in out.items() if k != "per_page"}, indent=2))
