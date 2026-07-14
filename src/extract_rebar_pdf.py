"""
Rebar PDF Vector Extractor
MinoriLabs — Salit Steel Agent Practice

Extracts vector elements from rebar construction PDFs:
- Line segments (rod geometry)
- Text annotations (bar marks, diameters, lengths)
- Tables (bar bending schedules)
- Dimensions

Usage:
    python extract_rebar_pdf.py <pdf1> <pdf2> ... [--verbose] [--output json]

Examples:
    python extract_rebar_pdf.py arch.pdf struct.pdf
    python extract_rebar_pdf.py arch.pdf struct.pdf --output json
"""

import subprocess, sys

def _ensure_deps():
    deps = {"fitz": "pymupdf"}
    for mod, pkg in deps.items():
        try:
            __import__(mod)
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg,
                                   "--break-system-packages", "-q"])

_ensure_deps()


import fitz  # PyMuPDF
import json
import sys
import os
import re
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_general_pdf import (
    is_garbled, extract_schedules, extract_structure_facts, page_spans,
    extract_plan_instances, extract_footing_pad_schedule,
)


def extract_text_blocks(page):
    """Extract all text with position info."""
    blocks = []
    for block in page.get_text("dict")["blocks"]:
        if block["type"] == 0:  # text block
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if text:
                        blocks.append({
                            "text": text,
                            "x": round(span["origin"][0], 2),
                            "y": round(span["origin"][1], 2),
                            "size": round(span["size"], 2),
                            "font": span["font"]
                        })
    return blocks


def extract_lines(page):
    """Extract vector line segments (rod representations)."""
    lines = []
    paths = page.get_drawings()
    for path in paths:
        for item in path["items"]:
            if item[0] == "l":  # line
                p1, p2 = item[1], item[2]
                length = ((p2.x - p1.x)**2 + (p2.y - p1.y)**2) ** 0.5
                lines.append({
                    "x1": round(p1.x, 2),
                    "y1": round(p1.y, 2),
                    "x2": round(p2.x, 2),
                    "y2": round(p2.y, 2),
                    "length_pts": round(length, 2),
                    "color": path.get("color"),
                    "width": path.get("width")
                })
    return lines


def extract_bar_schedule(text_blocks):
    """
    Attempt to parse bar bending schedule from text blocks.
    Handles Canadian (15M, 20M), standard mm/dia, counts (e.g., 3-15M), and spacing.
    """
    bar_marks = {}

    # Patterns
    bar_pattern = re.compile(r'^([A-Z]\d+)$')
    # Canadian 10M, 15M, 20M, 25M, 30M, 35M, 45M, 55M or standard 10mm-50mm
    dia_pattern = re.compile(r'(\d{1,2})[MmøØ]|(\d+)\s*(?:mm|dia)', re.IGNORECASE)
    # Count + Bar: 3-15M
    count_dia_pattern = re.compile(r'(\d+)[x\-]((\d{1,2})[MmøØ]|\d+\s*mm)', re.IGNORECASE)
    # Spacing pattern: @ 200 o/c, @ 300
    spacing_pattern = re.compile(r'@\s*(\d{2,3})', re.IGNORECASE)

    # Canadian standard bar size lookup (nominal diameter in mm)
    CANADIAN_BAR_DIA = {
        10: 11, 15: 16, 20: 20, 25: 25, 30: 30, 35: 35, 45: 45, 55: 55
    }

    current_mark = None
    for block in text_blocks:
        text = block["text"]

        # Issue 4: never let corrupted CAD-font glyphs seed or feed a bar mark.
        if is_garbled(text):
            continue

        if bar_pattern.match(text):
            current_mark = text
            if current_mark not in bar_marks:
                bar_marks[current_mark] = {
                    "bar_mark":    current_mark,
                    "bar_size":    None,    # e.g. "15M" or "16mm"
                    "diameter_mm": None,    # numeric mm
                    "count":       None,    # number of bars (from 3-15M style)
                    "spacing_mm":  None,    # c/c spacing if found
                    "confidence":  "low",   # upgraded once real rebar data attaches
                    "raw_annotations": []
                }

        if current_mark:
            entry = bar_marks[current_mark]

            # 1. Count + Canadian bar: "3-15M", "2-20M BARS"
            count_can = re.search(r'(\d+)\s*[-x]\s*(\d{2})M\b', text, re.IGNORECASE)
            if count_can and entry["diameter_mm"] is None:
                cnt, size = int(count_can.group(1)), int(count_can.group(2))
                entry["count"]       = cnt
                entry["bar_size"]    = f"{size}M"
                entry["diameter_mm"] = CANADIAN_BAR_DIA.get(size, size)

            # 2. Standalone Canadian bar size: "15M", "20M BARS"
            can_only = re.search(r'\b(\d{2})M\b', text, re.IGNORECASE)
            if can_only and entry["diameter_mm"] is None:
                size = int(can_only.group(1))
                entry["bar_size"]    = f"{size}M"
                entry["diameter_mm"] = CANADIAN_BAR_DIA.get(size, size)

            # 3. Explicit mm / ø notation: "16mm", "ø16", "T16", "H16"
            exp_dia = re.search(r'(?:ø|Ø|T(?=\d)|H(?=\d)|R(?=\d))?(\d+)\s*(?:mm|dia)\b',
                                text, re.IGNORECASE)
            if exp_dia and entry["diameter_mm"] is None:
                val = int(exp_dia.group(1))
                if 6 <= val <= 50:           # sanity check: realistic bar dia range
                    entry["diameter_mm"] = val
                    entry["bar_size"]    = f"{val}mm"

            # 4. Spacing: "@ 300mm c/c", "@ 150 c/c"
            sp = re.search(r'@\s*(\d{2,4})\s*(?:mm)?\s*c/c', text, re.IGNORECASE)
            if sp and entry["spacing_mm"] is None:
                entry["spacing_mm"] = int(sp.group(1))

            # 5. Count from "x3", "2 No.", "qty: 4" patterns
            if entry["count"] is None:
                qty = re.search(r'\b(\d+)\s*(?:No\.|nos?\.?|pcs?\.?|qty)\b', text, re.IGNORECASE)
                if qty:
                    entry["count"] = int(qty.group(1))

            entry["raw_annotations"].append(text)

    # Issue 1: a mark is a real rebar mark only if a genuine rebar-size signal
    # resolved. On these CAD sheets any stray dimension (e.g. "50mm", "38mm")
    # sits near a section/detail callout, so a bare mm number is NOT proof of a
    # bar. We require a Canadian bar size (NNM) or an explicit bar count; a lone
    # mm "size" or spacing value is treated as a dimension, not a bar mark.
    for entry in bar_marks.values():
        if _has_rebar_signal(entry):
            entry["confidence"] = "high"

    return list(bar_marks.values())


_CANADIAN_SIZE = re.compile(r"^\d{2}M$")


def _has_rebar_signal(bar):
    size = bar.get("bar_size") or ""
    count = bar.get("count")
    return bool(_CANADIAN_SIZE.match(size) or (isinstance(count, int) and count > 0))


def is_confident_mark(bar):
    """A bar-mark row is emitted only when it carries a genuine rebar signal."""
    return _has_rebar_signal(bar)


def deduplicate_bars(bars):
    """
    Remove duplicate bar entries (same bar mark appearing in multiple views).
    Keeps the entry with the most complete data.
    """
    seen = {}
    for bar in bars:
        mark = bar["bar_mark"]
        if mark not in seen:
            seen[mark] = bar
        else:
            # Keep the one with more data filled in
            existing_score = sum(1 for v in seen[mark].values() if v)
            new_score = sum(1 for v in bar.values() if v)
            if new_score > existing_score:
                seen[mark] = bar
    return list(seen.values())


def extract_tables(page):
    """Extract tables using PyMuPDF's table finder."""
    tables = []
    try:
        tab = page.find_tables()
        for t in tab.tables:
            df = t.to_pandas()
            tables.append(df.to_dict(orient="records"))
    except Exception as e:
        tables = [{"error": str(e)}]
    return tables


def process_pdf(pdf_path, verbose=False):
    doc = fitz.open(pdf_path)
    result = {
        "file": pdf_path,
        "doc_type": "structural",
        "pages": doc.page_count,
        "per_page": [],
        "bar_schedule": [],
        "dropped_marks": [],
        "superseded_by_schedule": [],
        "schedule_elements": [],
        "schedules_detected": [],
        "plan_instances": {},
        "footing_schedule": {},
        "structure_facts": {},
        "unmapped_text": [],
        "summary": {}
    }

    all_bars = []
    all_spans = []
    clean_lines = []
    unmapped_text = []

    for page_num in range(doc.page_count):
        page = doc[page_num]
        print(f"  Processing page {page_num + 1}/{doc.page_count}...")

        text_blocks = extract_text_blocks(page)
        lines = extract_lines(page)
        bars = extract_bar_schedule(text_blocks)
        tables = extract_tables(page)

        all_bars.extend(bars)
        all_spans.extend(page_spans(page, page_num + 1))
        for b in text_blocks:
            # Issue 4: garbled CAD-font spans go to Unmapped Text, not parsing.
            (unmapped_text if is_garbled(b["text"]) else clean_lines).append(b["text"])

        page_data = {
            "page": page_num + 1,
            "text_block_count": len(text_blocks),
            "line_count": len(lines),
            "bar_marks_found": [b["bar_mark"] for b in bars],
            "tables": tables
        }

        if verbose:
            page_data["text_blocks"] = text_blocks
            page_data["lines"] = lines

        result["per_page"].append(page_data)

    # Deduplicate bars across all pages (handles multi-view double counting)
    deduped = deduplicate_bars(all_bars)

    # Issue 2: parse PIER / FOOTING / COLUMN / BASE PLATE schedule tables from
    # span coordinates. These are the authoritative source of element rebar data.
    result["schedule_elements"], result["schedules_detected"] = extract_schedules(all_spans)

    # Issue 8: plan-view instance counts (distinct F#,P# / C#,BP# callout
    # locations) — the true "how many footings/columns" answer, vs. the handful
    # of schedule types.
    result["plan_instances"] = extract_plan_instances(all_spans)
    # Issue 8/Q10: per-type footing sizes from the column-keyed pad schedule.
    result["footing_schedule"] = extract_footing_pad_schedule(all_spans)

    # Marks owned by a schedule (P1, F1, C1, ...) — plan-view mentions of these
    # must NOT re-report or overwrite the schedule's data (Issue 2 dedup).
    schedule_marks = {m for el in result["schedule_elements"] for m in el.get("marks", [])}

    confident = [b for b in deduped if is_confident_mark(b)]
    result["bar_schedule"] = [b for b in confident if b["bar_mark"] not in schedule_marks]
    result["superseded_by_schedule"] = sorted(
        b["bar_mark"] for b in confident if b["bar_mark"] in schedule_marks)
    # Issue 1: bare marks with no genuine rebar signal are dropped as non-marks.
    result["dropped_marks"] = [b["bar_mark"] for b in deduped if not is_confident_mark(b)]

    # Issue 3: structure-level facts from the (garble-filtered) cover-sheet text.
    result["structure_facts"] = extract_structure_facts("\n".join(clean_lines))

    # Issue 4: garbled boilerplate captured separately (deduped, capped for size).
    result["unmapped_text"] = sorted(set(unmapped_text))

    result["summary"] = {
        "total_pages": doc.page_count,
        "unique_bar_marks": len(result["bar_schedule"]),
        "bar_marks": [b["bar_mark"] for b in result["bar_schedule"]],
        "dropped_mark_count": len(result["dropped_marks"]),
        "superseded_by_schedule": result["superseded_by_schedule"],
        "schedule_element_count": len(result["schedule_elements"]),
        "schedules_detected": result["schedules_detected"],
        "unmapped_text_count": len(result["unmapped_text"]),
    }

    doc.close()
    return result


def main():
    # Separate PDF paths from flags
    flags = {"--verbose", "--output", "json"}
    pdf_paths = [arg for arg in sys.argv[1:] if not arg.startswith("--") and arg not in flags]
    verbose = "--verbose" in sys.argv
    output_json = "--output" in sys.argv and "json" in sys.argv

    if not pdf_paths:
        print("Usage: python extract_rebar_pdf.py <pdf1> [pdf2 ...] [--verbose] [--output json]")
        print("Example: python extract_rebar_pdf.py arch.pdf struct.pdf --output json")
        sys.exit(1)

    all_results = []
    total_bars_combined = set()

    for pdf_path in pdf_paths:
        print(f"\n{'=' * 60}")
        print(f"Extracting: {pdf_path}")
        print("=" * 60)

        try:
            result = process_pdf(pdf_path, verbose=verbose)
        except Exception as e:
            print(f"  [ERROR] Failed to process '{pdf_path}': {e}")
            continue

        all_results.append(result)

        # Per-file summary
        print(f"\n  Summary for: {pdf_path}")
        print(f"    Pages processed : {result['summary']['total_pages']}")
        print(f"    Unique bar marks: {result['summary']['unique_bar_marks']}")
        marks_str = ', '.join(result['summary']['bar_marks']) or 'None detected'
        print(f"    Bar marks found : {marks_str}")

        if result["bar_schedule"]:
            print(f"\n  Bar Schedule:")
            print(f"    {'MARK':<6} {'BAR SIZE':<10} {'DIA(mm)':<9} {'COUNT':<7} {'SPACING':<10} ANNOTATIONS")
            print(f"    {'-'*6} {'-'*10} {'-'*9} {'-'*7} {'-'*10} -----------")
            for bar in result["bar_schedule"]:
                size    = bar.get("bar_size")    or "unknown"
                dia     = f"{bar['diameter_mm']}mm" if bar["diameter_mm"] else "unknown"
                count   = str(bar.get("count"))  if bar.get("count")   else "-"
                spacing = f"{bar['spacing_mm']}mm c/c" if bar.get("spacing_mm") else "-"
                ann     = len(bar["raw_annotations"])
                print(f"    {bar['bar_mark']:<6} {size:<10} {dia:<9} {count:<7} {spacing:<10} {ann}")

        # Track all unique bar marks across all files
        total_bars_combined.update(result['summary']['bar_marks'])

        # Save per-file JSON if requested
        if output_json:
            out_path = pdf_path.replace(".pdf", "_extracted.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n  Output saved to: {out_path}")

    # Combined summary across all PDFs
    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print(f"COMBINED SUMMARY ({len(all_results)} PDFs processed)")
        print("=" * 60)
        print(f"  Total PDFs processed     : {len(all_results)}")
        print(f"  Total unique bar marks   : {len(total_bars_combined)}")
        print(f"  All bar marks (combined) : {', '.join(sorted(total_bars_combined)) or 'None detected'}")
        print("=" * 60)


if __name__ == "__main__":
    main()
