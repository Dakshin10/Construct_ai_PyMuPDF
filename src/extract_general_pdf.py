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

# Storey count. The bare "(\d+) FLOORS?" form was removed (Issue 8): with `\s`
# matching newlines, it matched a drawing-index sheet code across a line break
# — e.g. "...M2\nFLOOR PLAN BUILDING 'A'..." captured "2" as a storey count on a
# single-storey building. Only match the digit when it directly qualifies
# "STOREY/STOREYS" (or an explicit "NUMBER OF STOREYS:"), require the digit not
# to be glued to a preceding letter (so sheet codes like "M2"/"A3" can't feed
# it), and disallow a newline between the digit and the STOREY word.
_WORD_NUM = {"single": 1, "one": 1, "two": 2, "three": 3, "four": 4,
             "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}

STOREY_PATTERNS = [
    re.compile(r"NUMBER\s+OF\s+STOR(?:E?YS?|IES)\s*[:=]?\s*(\d+)", re.I),
    re.compile(r"(?<![A-Za-z0-9])(\d+)[ \t]*[- ]?[ \t]*STOR(?:E?YS?|IES)", re.I),
]
# Spelled-out storey counts ("single storey", "two storey building").
_STOREY_WORD_RE = re.compile(
    r"\b(single|one|two|three|four|five|six|seven|eight|nine|ten)[ \t]*[- ]?[ \t]*"
    r"STOR(?:E?YS?|IES)", re.I)

# Level fallback: an explicit "LEVEL N" or an ordinal floor ("2ND FLOOR"). A
# bare "FLOOR PLAN" index line is NOT a level — the ordinal is required, so the
# constant "FLOOR PLAN"/"GROUND FLOOR" index text can't inflate the count.
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

# extract_elements' "current element" context bounds (Issue 6 — measured
# against Building C: genuine per-element data sits 0-30 lines after its
# keyword; unrelated general-notes/boilerplate runs span 40-400+ lines).
_NUMBERED_NOTE_RE = re.compile(r"^\d+\.\d*(?:\s|$)")   # "4.", "5.0  ...", "23.  ..."
_MAX_IDLE_LINES = 20

ATTR_PATTERNS = {
    "spacing":      re.compile(r"@\s*(\d+)\s*(?:mm|MM|O\.?C\.?)?|SPACING\s*[:=]?\s*(\d+)", re.I),
    "stock_length": re.compile(r"STOCK\s+LENGTH\s*[:=]?\s*([\d.]+\s*(?:mm|m)?)", re.I),
    "stirrups":     re.compile(r"STIRRUPS?\s*[:=]?\s*(\d+M(?:\s*@\s*\d+)?)", re.I),
    "ties_type":    re.compile(r"TIES?\s*[:=]?\s*(\d+M)", re.I),
    "ties_length":  re.compile(r"TIES?\s+LENGTH\s*[:=]?\s*([\d.]+\s*(?:mm|m)?)", re.I),
    "num_rebars":   re.compile(r"(\d+)\s*[- ]\s*(\d+M)\b"),   # e.g. "6-20M"
    "size":         re.compile(r"(\d+)\s*(?:mm)?\s*[xX×]\s*(\d+)\s*(?:mm)?"),
    "count":        re.compile(r"(?:QTY|COUNT|NO\.?|TYP\.?\s+OF)\s*[:=]?\s*(\d+)", re.I),
    "shape_type":   re.compile(r"\b(STRAIGHT|BENT|L[- ]SHAPE[D]?|U[- ]SHAPE[D]?|HOOKED|CRANKED)\b", re.I),
}
# cover / lap_length / hook_length are NOT in ATTR_PATTERNS above — Salit
# drawings write "VALUE KEYWORD" (e.g. "76mm CLEAR COVER", "457mm MIN. LAP",
# "+ 76mm HOOK"), often as several stacked one-word PyMuPDF lines rather than
# "KEYWORD: value" on one line, so they need the windowed bidirectional
# matcher below instead of a single ordered regex. See _extract_cover /
# _extract_lap_length / _extract_hook_length.


def _first_group(match):
    return next((g for g in match.groups() if g), None)


# ── Bidirectional windowed matching for cover / lap / hook (Issue 5) ───────────
# These three attributes are written value-before-keyword as often as
# keyword-before-value, and the value can sit 1-2 PyMuPDF "lines" away from the
# keyword (a stacked leader callout, e.g. "76mm" / "CLEAR" / "COVER" as three
# separate lines at nearly the same x, consecutive y). Confirmed against the
# real Building C (20-055) drawing text — see diagnostic pass.

_COVER_KW_RE = re.compile(r"\bCOVER\b|\bCLR\.?\b(?!\s*\.?\s*OPENING)|\bCVR\.?\b", re.I)
_LAP_KW_RE   = re.compile(r"\bLAP\b", re.I)
_HOOK_KW_RE  = re.compile(r"\bHOOKS?\b", re.I)   # \b excludes "HOOKED" (adjective, not a length)

_VALUE_MM_RE   = re.compile(r"\d+(?:\.\d+)?\s*mm\b", re.I)
_VALUE_BARE_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


_PAREN_SPAN_RE = re.compile(r"\([^()]*\)")


def _mask_parens(window):
    """Blank out parenthetical spans (same length, so offsets stay valid) —
    e.g. the "1 1/2" inside "(1 1/2\")" — so an imperial-equivalent
    annotation isn't mistaken for the primary metric value
    (p7: "40 (1 1/2\") COVER" must resolve to "40", not "2")."""
    return _PAREN_SPAN_RE.sub(lambda m: " " * len(m.group(0)), window)


def _nearest_value(window, kw_match, value_res, max_dist=45, skip_parenthetical=False):
    """Nearest number match to `kw_match` in `window`, searching both
    directions. `value_res` is tried in order; the first pattern with any
    candidate within range wins (so an explicit-unit pattern is preferred
    over a bare-number fallback)."""
    search_text = _mask_parens(window) if skip_parenthetical else window
    ks, ke = kw_match.span()
    for value_re in value_res:
        best, best_dist = None, None
        for vm in value_re.finditer(search_text):
            vs, ve = vm.span()
            if vs < ke and ve > ks:
                continue  # overlaps the keyword itself
            dist = (vs - ke) if vs >= ke else (ks - ve)
            if dist < 0 or dist > max_dist:
                continue
            if best_dist is None or dist < best_dist:
                best, best_dist = vm, dist
        if best is not None:
            return best
    return None


def _window_text(lines, i, radius=2):
    """Join lines[i-radius : i+radius+1], but never cross a numbered
    general-note boundary (e.g. "4. Minimum concrete cover for
    reinforcing:") — that boundary is exactly what resets `current` in
    extract_elements, so letting a naive radius-slice ignore it would pull a
    nearby list-item number (e.g. "4.") back in as if it were real data."""
    lo = i
    for k in range(i - 1, max(-1, i - radius - 1), -1):
        if _NUMBERED_NOTE_RE.match(lines[k]):
            break
        lo = k
    hi = i
    for k in range(i + 1, min(len(lines), i + radius + 1)):
        if _NUMBERED_NOTE_RE.match(lines[k]):
            break
        hi = k
    return " ".join(lines[lo:hi + 1])


def _extract_cover(window):
    kw = _COVER_KW_RE.search(window)
    if not kw:
        return None
    vm = _nearest_value(window, kw, (_VALUE_MM_RE, _VALUE_BARE_RE),
                         max_dist=70, skip_parenthetical=True)
    return vm.group(0).strip() if vm else None


def _extract_lap_length(line):
    kw = _LAP_KW_RE.search(line)
    if not kw:
        return None
    # mm-unit required: excludes bare per-size lap-table entries ("500 LAP
    # (10M)" — handled separately as a table, see extract_lap_table) and
    # formula notes ("Minimum lap shall be 36 bar DIA.." has no "mm").
    vm = _nearest_value(line, kw, (_VALUE_MM_RE,), max_dist=25)
    return vm.group(0).strip() if vm else None


def _extract_hook_length(line):
    kw = _HOOK_KW_RE.search(line)
    if not kw:
        return None
    vm = _nearest_value(line, kw, (_VALUE_MM_RE,), max_dist=25)
    return vm.group(0).strip() if vm else None


_LAP_TABLE_ROW = re.compile(r"(\d+)\s*LAP\s*\(\s*(\d{1,2})M\s*\)", re.I)

# Slab (Issue 8/Q13,Q30): "152x152-MW18.7 … W.W.M." is a WELDED-WIRE-MESH
# designation, not a slab plan dimension — guard the size pattern against it.
# A slab is described by its THICKNESS ("150 Thk. CONC. SLAB", "SLAB … THICKNESS
# 152mm"), which is captured instead.
_MESH_TAIL_RE = re.compile(r"^\s*[-–]\s*M[WD]", re.I)   # "-MW18.7", "-MD…"
_MESH_CTX_RE = re.compile(r"W\.?\s*W\.?\s*M\.?|\bWWM\b|\bWWF\b|\bMESH\b|-M[WD]\d", re.I)
_SLAB_THK_INLINE_RE = re.compile(r"(\d{2,3})\s*(?:mm)?\s*Thk\.?", re.I)


def _extract_slab_thickness(window):
    """Slab thickness from a joined window: an inline 'N Thk.' or a 'THICKNESS'
    label with the nearest N/Nmm value. Returns e.g. '152mm thick' or None."""
    m = _SLAB_THK_INLINE_RE.search(window)
    if m:
        return f"{m.group(1)}mm thick"
    kw = re.search(r"THICKNESS", window, re.I)
    if kw:
        vm = _nearest_value(window, kw, (_VALUE_MM_RE, _VALUE_BARE_RE), max_dist=40)
        if vm:
            n = re.search(r"\d+", vm.group(0)).group(0)
            return f"{n}mm thick"
    return None


def extract_lap_table(text_lines):
    """Per-bar-size lap-length lookup table (e.g. p8: '500 LAP (10M)' /
    '630 LAP (15M)' / '840 LAP (20M)'). Returns {} unless >=2 rows are found,
    since a single row is more likely a one-off element-specific lap_length
    than a genuine reusable table."""
    table = {}
    for ln in text_lines:
        for m in _LAP_TABLE_ROW.finditer(ln):
            table[f"{m.group(2)}M"] = f"{m.group(1)}mm"
    return table if len(table) >= 2 else {}


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

# Steel HSS column sections — e.g. "HSS 203x203x7.9",
# "203mm x 203mm x 7.9mm Wall Thk." Columns here are steel, not concrete.
_RE_HSS_SIZE = re.compile(
    r"HSS\s*\d+\s*[xX]\s*\d+\s*[xX]\s*[\d.]+"
    r"|\d+\s*mm\s*[xX]\s*\d+\s*mm\s*[xX]\s*[\d.]+\s*mm", re.I)
_RE_HSS = re.compile(r"\bHSS\b|WALL\s*THK", re.I)


# ── Systemic mis-anchoring guard (steel / hardware / cross-reference) ──────────
# Steel-member, bolt/hardware-grade, welded-wire-mesh and drawing-cross-reference
# callouts sit next to concrete reinforcement schedules on the same sheets and
# get mis-anchored to concrete element types (columns absorbed pier rebar; beams
# absorbed angle-section / bolt-grade text). This ONE reusable check classifies a
# candidate element by its mark + nearby text so it can be applied everywhere,
# rather than adding element-type-specific patches.
_BOLT_GRADE    = re.compile(r"\bA3(?:25|07)\b")
_DWG_REF       = re.compile(r"\bDWG\.?\s*(?:No\.?|#|Number)|SECTION\s+ON\s+DWG"
                            r"|SEE\s+DWG|\bON\s+DWG\.", re.I)
_STEEL_CTX     = re.compile(
    r"\bHSS\b|\bL\s?\d+\s*x\s*\d+|\bW\d{2,3}\s*x|\bMC?\d+\s*x|WALL\s*THK"
    r"|O\.?\s?W\.?\s?S\.?\s?J|X-ENP|X-HSN|\bDECK\b|GAUGE|\bGA\.", re.I)
# Note: welded-wire mesh (WWM / "152x152-MW18.7") is CONCRETE slab reinforcement,
# not a steel member — a mesh MARK (MW##) is rejected via _STEEL_MARK, but mesh
# CONTEXT must not reclassify a concrete slab as steel, so it's excluded here.
_STEEL_MARK    = re.compile(r"^(L|W|MC|MW|WWF|HSS)\d", re.I)   # angle/W/channel/mesh
_HARDWARE_MARK = re.compile(r"^A3(?:25|07)$", re.I)            # ASTM bolt grades
_REBAR_SIGNAL  = re.compile(r"\d+\s*-\s*\d{2}M|\b\d{2}M\b")


def classify_member(label, context):
    """
    Classify a candidate structural element by its mark + nearby text:
      'reject'   — not a structural element at all (bolt/hardware grade, drawing
                   cross-reference, angle/mesh/fastener callout)
      'steel'    — steel member (HSS / W-shape / channel): carries a section, not
                   rebar; rebar/tie fields are inapplicable
      'concrete' — genuine concrete element; keep its rebar/tie data
    """
    lab = (label or "").strip()
    ctx = context or ""
    if _HARDWARE_MARK.match(lab):
        return "reject"
    if _BOLT_GRADE.search(ctx) and not _REBAR_SIGNAL.search(ctx):
        return "reject"
    if _DWG_REF.search(ctx) and re.match(r"^[A-Z]\d", lab):
        return "reject"
    if _STEEL_MARK.match(lab):
        # angle (L), welded-wire-mesh (MW/WWF) marks are not primary members
        return "reject" if re.match(r"^(L|MW|WWF)\d", lab, re.I) else "steel"
    if _STEEL_CTX.search(ctx):
        return "steel"
    return "concrete"


_STEEL_CLEARED_FIELDS = ("num_rebars", "rebar_size", "ties_type", "ties_length",
                         "ties_spacing", "stirrups", "cover", "spacing",
                         "stock_length", "lap_length", "hook_length", "shape_type")


# ── Element de-duplication (single source of truth) ───────────────────────────
# Collapses duplicate element records (e.g. the same C1,C2 HSS column captured on
# two sheets, plus its unlabeled detail row) into one. Lives here in the
# extraction module — not the Chroma ingester — so both the raw extractor
# (run.py) and the ingestion path use the SAME logic and can't diverge.
_REBAR_DATA_FIELDS = ("num_rebars", "rebar_size", "ties_type", "ties_spacing",
                      "ties_length", "spacing", "size", "count", "stirrups")


def _rebar_richness(el):
    """How many rebar-ish fields are populated (for keep-the-richer dedup)."""
    return sum(1 for f in _REBAR_DATA_FIELDS if el.get(f) not in (None, "", []))


def _has_rebar_data(el):
    return _rebar_richness(el) > 0


def _merge_element(keeper, other):
    """Fill keeper's empty fields from other; union marks. keeper wins ties."""
    for k, v in other.items():
        if k == "marks":
            merged = list(dict.fromkeys((keeper.get("marks") or []) + (v or [])))
            keeper["marks"] = merged
        elif keeper.get(k) in (None, "", []) and v not in (None, "", []):
            keeper[k] = v
    if keeper.get("marks"):
        keeper["label"] = ",".join(sorted(set(keeper["marks"])))
    return keeper


def dedup_elements(project_id, elements, verbose=True):
    """
    Collapse duplicate element rows.

    1. Rows sharing (element_type, label) are merged into the richest one
       (field-union); sparser duplicates are dropped.
    2. Conservative association: when a type has exactly ONE labeled group that
       carries NO rebar data and exactly ONE unlabeled record that DOES, fold
       the data into the labeled row (so e.g. columns C1,C2 get their data).
       Guarded so types whose labeled group already has data (footings, piers)
       keep their distinct unlabeled rows untouched.
    3. Columns are steel HSS identified by mark; an unlabeled column detail
       record is the same columns' schedule detail — consolidated into the sole
       labeled column group so it isn't double-counted.

    verbose=False silences the [dedup]/[assoc] log lines (used for display).
    """
    def _log(msg):
        if verbose:
            print(msg)

    # --- Pass 1: merge exact (element_type, label) duplicates ---
    groups = {}
    passthrough = []   # records without a real label — kept as-is for now
    for el in elements:
        label = el.get("label")
        if not label:
            passthrough.append(el)
            continue
        key = (el.get("element_type"), label)
        if key not in groups:
            groups[key] = el
        else:
            keeper, drop = groups[key], el
            if _rebar_richness(drop) > _rebar_richness(keeper):
                keeper, drop = drop, keeper
            _merge_element(keeper, drop)
            groups[key] = keeper
            _log(f"  [dedup] dropped duplicate {drop.get('element_type')} "
                 f"'{label}' from {drop.get('source_doc_id','?')} "
                 f"(kept richer row for same mark)")

    labeled = list(groups.values())

    # --- Pass 2: conservative association of unlabeled data into empty labels ---
    def _is_sched(el):
        return el.get("source") == "schedule"

    by_type = defaultdict(list)
    for el in labeled:
        by_type[el.get("element_type")].append(el)
    remaining_unlabeled = []
    for el in passthrough:
        et = el.get("element_type")
        if not (_is_sched(el) and _has_rebar_data(el)):
            remaining_unlabeled.append(el)
            continue
        empty_labeled = [g for g in by_type.get(et, [])
                         if _is_sched(g) and not _has_rebar_data(g)]
        siblings = [u for u in passthrough
                    if u.get("element_type") == et and _is_sched(u) and _has_rebar_data(u)]
        if len(empty_labeled) == 1 and len(siblings) == 1:
            _merge_element(empty_labeled[0], el)
            _log(f"  [assoc] folded unlabeled {et} data "
                 f"(rebar={el.get('rebar_size')}, n={el.get('num_rebars')}) "
                 f"into '{empty_labeled[0].get('label')}'")
            continue
        remaining_unlabeled.append(el)

    labeled_cols = [g for g in labeled if g.get("element_type") == "column"]
    if len(labeled_cols) == 1:
        keep, rest = labeled_cols[0], []
        for el in remaining_unlabeled:
            if el.get("element_type") == "column":
                _merge_element(keep, el)
                _log(f"  [assoc] consolidated unlabeled column detail "
                     f"into '{keep.get('label')}'")
            else:
                rest.append(el)
        remaining_unlabeled = rest

    return labeled + remaining_unlabeled


# ── Concrete grade-beam callout (A1) ─────────────────────────────────────────────
# A concrete grade beam appears only as a free-text foundation-plan callout
# ("350mm WIDE x 300mm Dp. CONT. CONC. (FORMED) GRADE BEAM REINF. w/ 2-15M CONT.
# BARS"), not as a schedule row, so the generic keyword parser mis-groups its
# rebar onto a neighbouring footing/pier. This targeted scan captures it as its
# own concrete beam element (deduped — the typical detail repeats many times).
_GRADE_BEAM = re.compile(r"GRADE\s*BEAM", re.I)
_GB_SIZE    = re.compile(r"(\d{2,4})\s*(?:mm)?\s*WIDE\s*x\s*(\d{2,4})\s*(?:mm)?\s*Dp", re.I)
# The grade beam's own reinforcing is the "... CONT. BARS AT TOP OF WALL" phrase —
# distinct from a neighbouring footing's "... CONT. BARS TOP & BOTTOM", so tie
# only that phrase to the grade beam and don't absorb footing rebar.
_GB_REBAR   = re.compile(r"(\d+)\s*-\s*(\d{2})M\s*CONT\.?\s+BARS?\s+AT\s+TOP\s+OF\s+WALL", re.I)


def _blank_element(etype, label):
    return {
        "element_type": etype, "label": label,
        "size": None, "count": None, "num_rebars": None, "rebar_size": None,
        "shape_type": None, "stirrups": None, "spacing": None, "cover": None,
        "stock_length": None, "lap_length": None, "hook_length": None,
        "ties_type": None, "ties_length": None, "material": None, "raw_lines": [],
    }


def extract_grade_beams(text_lines):
    """Capture concrete grade-beam callouts as beam elements (deduped)."""
    if not any(_GRADE_BEAM.search(ln) for ln in text_lines):
        return []
    beams = {}
    for i, ln in enumerate(text_lines):
        if not _GRADE_BEAM.search(ln):
            continue
        window = " ".join(text_lines[max(0, i - 4):i + 5])
        if classify_member(None, window) != "concrete":
            continue  # must be confirmed concrete, not a steel/hardware neighbour
        sz = _GB_SIZE.search(window)
        rb = _GB_REBAR.search(window)
        # Require BOTH the grade beam's own size and its own rebar phrase, so a
        # window that only overlaps a neighbouring footing's rebar is skipped.
        if not (sz and rb):
            continue
        size = f"{sz.group(1)}x{sz.group(2)}"
        num = int(rb.group(1))
        rebar = f"{rb.group(2)}M"
        key = (size, num, rebar)
        if key in beams:
            continue
        # size-specific label so genuinely different sections (e.g. 350x300 vs
        # 350x400 grade beams) aren't collapsed by the (type, label) dedup.
        rec = _blank_element("beam", f"grade beam {size}")
        rec.update(size=size, num_rebars=num, rebar_size=rebar,
                   shape_type="continuous", material="concrete")
        rec["raw_lines"].append(re.sub(r"\s+", " ", window).strip()[:160])
        beams[key] = rec
    return list(beams.values())


# Steel roof/framing beams (Issue 8/Q9,Q17,Q27). The framing plans call out
# steel beams by section: W-shapes ("W460x60 BM.") are beams (columns in this
# project are HSS, never W); rectangular HSS ("HSS 254x152x7.9 … OUTRIGGER
# BEAMS", differing cross dims) are girts/outriggers/beams. Square HSS
# ("203x203") is a column and X-BRACING is not a beam — both excluded. These are
# steel SECTIONS (types), one per distinct designation, with an occurrence count.
_W_BEAM_RE = re.compile(r"\bW\d{3}\s*x\s*\d+\b", re.I)
_HSS_ANY_RE = re.compile(r"\bHSS\s*(\d+)\s*x\s*(\d+)\s*x\s*[\d.]+", re.I)
_BRACE_RE = re.compile(r"BRAC(?:E|ING)", re.I)


def extract_steel_beams(text_lines):
    """Capture steel framing-beam sections (W-shapes + rectangular HSS) as steel
    beam elements, one per distinct section with an occurrence count."""
    counts = {}          # normalized section -> [count, kind]
    for ln in text_lines:
        for m in _W_BEAM_RE.finditer(ln):
            sec = re.sub(r"\s+", "", m.group(0)).upper()
            counts.setdefault(sec, [0, "W-shape beam"])[0] += 1
        for m in _HSS_ANY_RE.finditer(ln):
            a, b = int(m.group(1)), int(m.group(2))
            if a == b:
                continue                      # square HSS = column, not a beam
            if _BRACE_RE.search(ln):
                continue                      # X-bracing, not a beam
            sec = re.sub(r"\s+", "", m.group(0)).upper()   # normalize all spacing
            counts.setdefault(sec, [0, "rectangular HSS beam/girt"])[0] += 1
    beams = []
    for sec, (n, kind) in sorted(counts.items()):
        rec = _blank_element("beam", f"steel beam {sec}")
        rec.update(size=sec, material="steel", shape_type=kind, count=n)
        rec["note"] = f"steel section; {n} occurrence(s) on framing plans"
        beams.append(rec)
    return beams


def _new_sched_elem(element_type):
    return {
        "element_type": element_type,
        "label": None,
        "marks": [],                 # schedule marks (P1, F1, ...) found in-region
        "size": None, "count": None, "num_rebars": None, "rebar_size": None,
        "shape_type": None, "stirrups": None, "spacing": None, "cover": None,
        "stock_length": None, "lap_length": None, "hook_length": None,
        "ties_type": None, "ties_length": None, "ties_spacing": None,
        "material": None,            # "steel" for HSS columns, else None/concrete
        "note": None,                # data-quality warnings (e.g. mismatch flag)
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
            has_hss = bool(_RE_HSS_SIZE.search(t) or _RE_HSS.search(t))
            if not (is_mark or has_reinf or has_hss):
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
            is_col = (best == "column")

            if is_mark:
                # A footing schedule shouldn't absorb a 'C1' column mark, etc.
                prefix = {"pier": "P", "footing": "F", "column": "C",
                          "base_plate": "BP", "lintel": "L"}.get(best)
                if prefix and t.upper().startswith(prefix) and t not in rec["marks"]:
                    rec["marks"].append(t)
                continue

            # ── Columns are STEEL (HSS) in this project — capture the section, and
            #    NEVER attach concrete rebar/tie data to them. A rebar callout that
            #    lands near a column heading is pier/wall reinforcement that reused a
            #    C-mark or sits nearby; flag it as a probable mismatch rather than
            #    silently ingesting it as column rebar. ──────────────────────────
            if is_col:
                hs = _RE_HSS_SIZE.search(t)
                if hs:
                    spec = re.sub(r"\s+", " ", hs.group(0)).strip()
                    if rec["size"] is None:
                        rec["size"] = spec
                    elif spec not in rec["size"]:
                        rec["size"] += "; " + spec   # keep both C1=7.9 / C2=9.5
                if _RE_HSS.search(t):
                    rec["material"] = "steel"
                if has_reinf:
                    rec["note"] = ("rebar/tie callout near an HSS column heading "
                                   "ignored as probable pier/wall data")
                continue

            # A "N-NNM TIES" callout means N ties of size NNM — NOT N vertical
            # rebars. Only treat "N-NNM" as vertical reinforcing on non-tie lines,
            # so tie-notes don't fabricate a bogus rebar count/size.
            m = _RE_NUMBAR.search(t)
            if m and "TIES" not in up:
                n = int(m.group(1))
                # Keep rebar_size tied to the LARGEST bar group, so a minor
                # "1-15M vertical adjacent to openings" note doesn't overwrite the
                # dominant "16-20M" vertical steel.
                if n >= (rec["num_rebars"] or 0):
                    rec["rebar_size"] = f"{int(m.group(2))}M"
                rec["num_rebars"] = max(rec["num_rebars"] or 0, n)
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
        # Mis-anchor guard (Issue 8): TIES are a pier/column feature. A spread
        # footing (pad schedule: "BARS @ spacing TOP & BOTTOM") and a base plate
        # (anchor bolts, no rebar) do NOT carry rebar ties. An UNLABELED
        # footing/base_plate record whose reinforcing is tie data is a pier tie
        # callout ("10M TIES @ 300") that got anchored to the wrong heading —
        # the genuine pier already owns this data. Drop it rather than surface a
        # phantom footing/base plate. (Labelled records are trusted as-is.)
        if (rec["element_type"] in ("footing", "base_plate")
                and not rec["marks"] and rec.get("ties_type")):
            continue
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
        "reinforcement_notes": None,
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
        wm = _STOREY_WORD_RE.search(full_text)
        if wm:
            facts["storeys"] = _WORD_NUM[wm.group(1).lower()]
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

    # Reinforcement general notes (Issue 8/Q7): project-wide cover and lap-splice
    # standards live in the general notes, not in per-element schedule rows. Q7's
    # availability audit wrongly reported cover/lap "unavailable" because these
    # were never surfaced. Capture them here so the structure_overview carries
    # them and an availability check reflects the drawings.
    reinf = []
    fm = re.search(r"COVER\s+FOR\s+FOOTING[\s\S]{0,80}?TOP:?\s*(\d+)\s*mm"
                   r"[\s\S]{0,40}?BOTTOM:?\s*(\d+)\s*mm", full_text, re.I)
    if fm:
        reinf.append(f"concrete cover — footings: top {fm.group(1)}mm, bottom {fm.group(2)}mm")
    # Other stated cover values (e.g. "76mm CLEAR COVER", "38mm CONC. COVER").
    covers = []
    for m in re.finditer(r"(\d{2,3})\s*mm[\s\S]{0,16}?(?:CLEAR|CONC\.?)[\s\S]{0,10}?COVER", full_text, re.I):
        # Real concrete cover is small (≈20–100mm); larger values are spacings/
        # depths that happen to sit near the word COVER.
        if int(m.group(1)) > 100:
            continue
        v = f"{m.group(1)}mm"
        if v not in covers:
            covers.append(v)
    if covers:
        reinf.append("other clear cover values noted: " + ", ".join(covers))
    # Lap-splice table by bar size ("500 LAP (10M)" …) + minimum-lap rule.
    lap_tbl = extract_lap_table(full_text.splitlines())
    if lap_tbl:
        reinf.append("lap splice by bar size: "
                     + ", ".join(f"{k} {v}" for k, v in sorted(lap_tbl.items())))
    lm = re.search(r"[Mm]inimum\s+lap[\s\S]{0,40}?(\d+)\s*bar\s*DIA", full_text, re.I)
    if lm:
        reinf.append(f"minimum lap: {lm.group(1)} bar diameters")
    if reinf:
        facts["reinforcement_notes"] = "; ".join(reinf)

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


def extract_elements(text_lines, rejections=None):
    """
    Best-effort element records from drawing text lines.
    Groups attribute annotations onto the most recent element keyword seen.
    Returns list of dicts: element_type, size, count, num_rebars, shape_type,
    stirrups, spacing, cover, stock_length, lap_length, hook_length,
    ties_type, ties_length, label.

    If `rejections` is a list, each candidate dropped by the mis-anchoring guard
    is appended as {"element_type", "label", "reason"} (used by the --dry-run
    audit). Pass nothing for normal extraction.
    """
    elements = {}
    current = None
    idle = 0   # consecutive lines since `current` last saw a keyword or real data

    for i, line in enumerate(text_lines):
        # Numbered general-note lines ("4.", "5.0  THE MINIMUM DEPTH OF SOIL
        # COVER REQUIRED FOR FOOTINGS...", "23.  DOWELS FROM THE FOOTINGS...")
        # are spec-notes, not element schedule data — checked BEFORE keyword
        # detection, since these notes routinely also mention an element
        # keyword in the same sentence (they're often *about* footings/
        # columns) and would otherwise re-anchor `current` to prose instead
        # of resetting it. Hard boundary: never starts nor extends `current`.
        if _NUMBERED_NOTE_RE.match(line):
            current = None
            continue

        matched_kw = False
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
                idle = 0
                matched_kw = True
                break

        if current is None:
            continue

        if not matched_kw:
            # `current` persists across lines until the next element keyword,
            # which lets it silently absorb general-notes boilerplate that has
            # nothing to do with the element (measured: real per-element data
            # sits 0-30 lines after the keyword in this document, while
            # boilerplate runs span 40-400+ lines — see diagnostic). An idle
            # budget bounds this: `current` expires if MAX_IDLE_LINES pass
            # with no new keyword AND no new attribute data, so an unnumbered
            # boilerplate run (e.g. repeated "ALL HSS COLUMNS" callouts)
            # can't drift indefinitely either.
            idle += 1
            if idle > _MAX_IDLE_LINES:
                current = None
                continue

        el = elements[current]
        el["raw_lines"].append(line)
        found_data = False

        m = ATTR_PATTERNS["num_rebars"].search(line)
        if m and el["num_rebars"] is None:
            el["num_rebars"], el["rebar_size"] = int(m.group(1)), m.group(2)
            found_data = True
        m = ATTR_PATTERNS["size"].search(line)
        if m and el["size"] is None and not ATTR_PATTERNS["num_rebars"].search(line):
            # Mesh guard: skip "152x152" when it's a welded-wire-mesh spec
            # ("152x152-MW18.7 … W.W.M."), not an element plan dimension.
            if not (_MESH_TAIL_RE.match(line[m.end():]) or _MESH_CTX_RE.search(line)):
                el["size"] = f"{m.group(1)}x{m.group(2)}"
                found_data = True
        # Slab thickness IS the slab's defining dimension (its "size").
        if el["element_type"] == "slab" and el["size"] is None:
            thk = _extract_slab_thickness(_window_text(text_lines, i, radius=2))
            if thk:
                el["size"] = thk
                found_data = True
        for attr in ("spacing", "stock_length",
                     "stirrups", "ties_type", "ties_length", "count", "shape_type"):
            if el[attr] is None:
                m = ATTR_PATTERNS[attr].search(line)
                if m:
                    val = _first_group(m)
                    el[attr] = val.strip() if isinstance(val, str) else val
                    found_data = True

        # cover / lap_length / hook_length: value-before-or-after-keyword,
        # sometimes spanning stacked lines — use the windowed matcher instead
        # of a single ordered regex (see Issue 5 diagnostic).
        if el["cover"] is None:
            # radius=3, not 2: the stacked "76mm" / "CLEAR" / "COVER" callout
            # can sit 3 lines before the element keyword it re-anchors
            # `current` on (e.g. "...COVER FOOTINGS ARE 1.7m to 2.5m MAX...":
            # the keyword line is 3 lines after "76mm"), and at radius=2 the
            # window would miss "76mm" entirely and fall back to a stray
            # digit from an unrelated "1.7m" depth-range annotation instead.
            val = _extract_cover(_window_text(text_lines, i, radius=3))
            if val:
                el["cover"] = val
                found_data = True
        if el["lap_length"] is None:
            val = _extract_lap_length(line)
            if val:
                el["lap_length"] = val
                found_data = True
        if el["hook_length"] is None:
            val = _extract_hook_length(line)
            if val:
                el["hook_length"] = val
                found_data = True

        if found_data:
            idle = 0

    # Drop unlabeled elements with no attribute data at all — these come from
    # incidental keyword mentions (titles, general notes), not real elements.
    def _has_data(el):
        attrs = ("size", "count", "num_rebars", "rebar_size", "shape_type",
                 "stirrups", "spacing", "cover", "stock_length", "lap_length",
                 "hook_length", "ties_type", "ties_length")
        return el["label"] is not None or any(el[a] not in (None, "") for a in attrs)

    out = []
    for el in elements.values():
        if not _has_data(el):
            continue
        # Use only the element's immediate callout (label + first few lines) for
        # material classification. extract_elements greedily accumulates raw_lines
        # up to the next element keyword, so the full block can bleed into general
        # notes containing steel words ("HSS", "DECK", "steel design…") — which
        # would wrongly flag a concrete slab-on-grade as a steel member.
        ctx = (el.get("label") or "") + " " + " ".join(el.get("raw_lines", [])[:4])
        verdict = classify_member(el.get("label"), ctx)
        if verdict == "reject":
            if rejections is not None:
                rejections.append({"element_type": el.get("element_type"),
                                   "label": el.get("label"),
                                   "reason": "steel section / bolt grade / drawing "
                                             "reference (classify_member)"})
            continue  # bolt grade / drawing reference / angle / mesh — not an element
        # Implausible dimensions (garbage numbers grabbed from notes prose, e.g.
        # a "4x37" beam) — no real structural member has a <50mm plan dimension.
        sz = el.get("size")
        if sz:
            nums = [int(n) for n in re.findall(r"\d+", sz)]
            if nums and min(nums) < 50:
                if rejections is not None:
                    rejections.append({"element_type": el.get("element_type"),
                                       "label": el.get("label"),
                                       "reason": f"implausible dimension {sz}"})
                continue
        if verdict == "steel":
            el["material"] = "steel"
            for f in _STEEL_CLEARED_FIELDS:
                if f in el:
                    el[f] = None
        out.append(el)

    # A1: add the concrete grade beam(s) the greedy keyword parser can't isolate.
    out.extend(extract_grade_beams(text_lines))
    # Issue 8: steel framing beams (W-shapes + rectangular HSS) from framing plans.
    out.extend(extract_steel_beams(text_lines))
    return out


# ── Plan-view instance counting (Issue 8) ────────────────────────────────────
# Schedules define element TYPES (F1, F2, P1, C1 …); the foundation/framing plan
# stamps a combined callout at every physical location — "F1,P1" where an F1
# footing carries a P1 pier, "C1,BP1" where a C1 column sits on a BP1 base plate.
# Counting these callouts (deduped by location) yields the true INSTANCE count,
# which is what "how many footings/columns are there?" actually asks — distinct
# from the handful of schedule types. Only the combined format is used: it's an
# unambiguous plan-instance stamp, unlike a bare "F1" that may be a schedule or
# section reference.
_PLAN_COMBO_RE = re.compile(r"^(F\d{1,2})\s*,\s*(P\d{1,2})$"
                            r"|^(C\d{1,2})\s*,\s*(BP\d{1,2})$", re.I)


def extract_plan_instances(spans, cluster_px=3):
    """
    spans: list of {"page","x","y","text"} (from page_spans).
    Returns {element_type: {mark: instance_count}} counting distinct plan
    locations of each combined callout, plus derived per-mark tallies for the
    paired element. E.g. "F1,P1" ×33 contributes footing F1=33 and pier P1=33.
    Locations within `cluster_px` on a page are treated as one (guards against
    a callout drawn twice for bold/shadow effect).
    """
    # Bucket callout locations per (page, mark-pair) then dedupe by cluster.
    seen = defaultdict(set)   # (footing_mark, pier_mark) or (col_mark, bp_mark) -> {(page, cx, cy)}
    kind = {}                 # pair -> ("footing","pier") | ("column","base_plate")
    for s in spans:
        m = _PLAN_COMBO_RE.match(s["text"].strip())
        if not m:
            continue
        if m.group(1):        # F#,P#
            a, b = m.group(1).upper(), m.group(2).upper()
            kind[(a, b)] = ("footing", "pier")
        else:                 # C#,BP#
            a, b = m.group(3).upper(), m.group(4).upper()
            kind[(a, b)] = ("column", "base_plate")
        cx = round(s["x"] / cluster_px) * cluster_px
        cy = round(s["y"] / cluster_px) * cluster_px
        seen[(a, b)].add((s["page"], cx, cy))

    out = {"footing": defaultdict(int), "pier": defaultdict(int),
           "column": defaultdict(int), "base_plate": defaultdict(int)}
    for (a, b), locs in seen.items():
        t_a, t_b = kind[(a, b)]
        out[t_a][a] += len(locs)
        out[t_b][b] += len(locs)
    # plain dicts, drop empty types
    return {et: dict(marks) for et, marks in out.items() if marks}


# ── Footing pad schedule (column-keyed matrix) — Issue 8/Q10 ──────────────────
# The Spread Footing Pad Schedule is a CAD matrix: F# type marks form a header
# ROW (same y, distinct x); LENGTH / WIDTH / DEPTH and reinforcing values sit in
# rows below, each value in the column of its footing type (aligned by x). The
# generic nearest-anchor schedule parser scrambles this; reconstructing it by
# column recovers per-type sizes (Q10 wrongly reported "no sizes exist").
_FOOT_DIM_RE = re.compile(r"^(\d{3,5})\s*mm$", re.I)
_FOOT_LABEL_RE = re.compile(r"^(LENGTH|WIDTH|DEPTH)$", re.I)
_FOOT_REINF_RE = re.compile(r"(\d{2})M\s*BARS?\s*@\s*(\d+)", re.I)


def extract_footing_pad_schedule(spans, x_tol=22):
    """
    spans: list of {"page","x","y","text"}.
    Returns {mark: {"size": "LxWxD", "reinforcing": "..."}} for footing types
    read out of the Spread Footing Pad Schedule matrix, or {} if not present.
    """
    by_page = defaultdict(list)
    for s in spans:
        by_page[s["page"]].append(s)

    result = {}
    for page, ps in by_page.items():
        # Only proceed on a page that actually has the schedule title.
        if not any(re.search(r"FOOTING\s+PAD\s+SCHEDULE|NOMINAL\s+FOOTING\s+SIZE"
                             r"|SPREAD\s+F\s*O\s*O\s*T\s*I\s*N\s*G|F O O T I N G     P A D",
                             s["text"], re.I) for s in ps):
            continue
        # Header row: >=3 F# marks sharing a y (within 8px), distinct x.
        fmarks = [(s["x"], s["y"], s["text"].upper()) for s in ps
                  if re.fullmatch(r"F[1-9]", s["text"].strip(), re.I)]
        if len(fmarks) < 3:
            continue
        # cluster by y to find the densest horizontal band of F-marks
        fmarks.sort(key=lambda t: t[1])
        best_band, i = [], 0
        for x, y, t in fmarks:
            band = [m for m in fmarks if abs(m[1] - y) <= 8]
            if len(band) > len(best_band):
                best_band = band
        # column centers: mark -> x (dedupe marks, keep first x)
        cols = {}
        for x, y, t in sorted(best_band, key=lambda m: m[0]):
            cols.setdefault(t, x)
        if len(cols) < 3:
            continue
        header_y = sum(m[1] for m in best_band) / len(best_band)

        def col_for(x):
            best, bd = None, 1e9
            for mk, cx in cols.items():
                if abs(x - cx) < bd:
                    bd, best = abs(x - cx), mk
            return best if bd <= x_tol else None

        # Find LENGTH/WIDTH/DEPTH label rows (by their y), gather dim values in band.
        dims = {mk: {} for mk in cols}          # mark -> {"LENGTH":v,...}
        reinf = {mk: None for mk in cols}
        label_ys = {}
        for s in ps:
            lm = _FOOT_LABEL_RE.fullmatch(s["text"].strip())
            if lm and s["y"] > header_y:
                label_ys[lm.group(1).upper()] = s["y"]
        for dim_name, ly in label_ys.items():
            for s in ps:
                if abs(s["y"] - ly) <= 12:
                    dm = _FOOT_DIM_RE.fullmatch(s["text"].strip())
                    if dm:
                        mk = col_for(s["x"])
                        if mk and dim_name not in dims[mk]:
                            dims[mk][dim_name] = int(dm.group(1))
        # Reinforcing: bar callouts below the header, column-assigned (first per col).
        for s in ps:
            if s["y"] <= header_y:
                continue
            rm = _FOOT_REINF_RE.search(s["text"])
            if rm:
                mk = col_for(s["x"])
                if mk and reinf[mk] is None:
                    reinf[mk] = f"{rm.group(1)}M @ {rm.group(2)} c/c"

        # NOTE: per-mark size/reinforcing here come from COLUMN alignment in the
        # schedule matrix (each F-mark header sits at an x that its LENGTH/WIDTH/
        # DEPTH and reinforcing values align to). Do NOT try to corroborate a
        # size against linearized get_text() output — a CAD matrix linearizes
        # cell-scrambled, so genuine values (e.g. F4 2500x2500x850) never appear
        # as an adjacent "LxWxD" string even though they are real column-aligned
        # entries. Verified against Building C: F1 1300x1300x400/15M@250 …
        # F4 2500x2500x850/20M@150, F5 2000x2000x500/20M@250 all column-correct.
        for mk in cols:
            d = dims[mk]
            size = None
            if "LENGTH" in d and "WIDTH" in d:
                size = f"{d['LENGTH']}x{d['WIDTH']}" + (f"x{d['DEPTH']}" if "DEPTH" in d else "")
            if size or reinf[mk]:
                result[mk] = {"size": size, "reinforcing": reinf[mk]}
    return result


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
        "plan_instances": extract_plan_instances(all_spans),
        "footing_schedule": extract_footing_pad_schedule(all_spans),
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
