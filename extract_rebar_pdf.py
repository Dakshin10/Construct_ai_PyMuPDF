"""
Rebar PDF Vector Extractor
MinoriLabs — Salit Steel Agent Practice

Extracts vector elements from rebar construction PDFs:
- Line segments (rod geometry)
- Text annotations (bar marks, diameters, lengths)
- Tables (bar bending schedules)
- Dimensions

Usage:
    python extract_rebar_pdf.py <path_to_pdf>
    python extract_rebar_pdf.py <path_to_pdf> --output json
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
import re
from collections import defaultdict


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
    Looks for patterns like: B1, 16mm, 2400, 12 (bar mark, dia, length, count)
    """
    bar_marks = {}

    # Common bar mark patterns: B1, T12, R8, H16, etc.
    bar_pattern = re.compile(
        r'^([A-Z]\d+)$'          # bar mark e.g. B1, T2
    )
    dia_pattern = re.compile(
        r'(\d+)\s*(?:mm|dia|ø|Ø)', re.IGNORECASE
    )
    dim_pattern = re.compile(
        r'(\d{3,5})'             # dimensions 100-99999 (mm)
    )

    current_mark = None
    for block in text_blocks:
        text = block["text"]

        if bar_pattern.match(text):
            current_mark = text
            bar_marks[current_mark] = {
                "bar_mark": current_mark,
                "diameter_mm": None,
                "lengths_mm": [],
                "count": None,
                "raw_annotations": []
            }

        if current_mark:
            dia_match = dia_pattern.search(text)
            if dia_match:
                bar_marks[current_mark]["diameter_mm"] = int(dia_match.group(1))

            bar_marks[current_mark]["raw_annotations"].append(text)

    return list(bar_marks.values())


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
        "pages": doc.page_count,
        "per_page": [],
        "bar_schedule": [],
        "summary": {}
    }

    all_bars = []

    for page_num in range(doc.page_count):
        page = doc[page_num]
        print(f"  Processing page {page_num + 1}/{doc.page_count}...")

        text_blocks = extract_text_blocks(page)
        lines = extract_lines(page)
        bars = extract_bar_schedule(text_blocks)
        tables = extract_tables(page)

        all_bars.extend(bars)

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
    result["bar_schedule"] = deduplicate_bars(all_bars)

    result["summary"] = {
        "total_pages": doc.page_count,
        "unique_bar_marks": len(result["bar_schedule"]),
        "bar_marks": [b["bar_mark"] for b in result["bar_schedule"]]
    }

    doc.close()
    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_rebar_pdf.py <pdf_path> [--verbose] [--output json]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    verbose = "--verbose" in sys.argv
    output_json = "--output" in sys.argv and "json" in sys.argv

    print(f"\nExtracting vector data from: {pdf_path}")
    print("-" * 50)

    result = process_pdf(pdf_path, verbose=verbose)

    print(f"\nSummary:")
    print(f"  Pages processed : {result['summary']['total_pages']}")
    print(f"  Unique bar marks: {result['summary']['unique_bar_marks']}")
    print(f"  Bar marks found : {', '.join(result['summary']['bar_marks']) or 'None detected'}")

    if result["bar_schedule"]:
        print(f"\nBar Schedule:")
        for bar in result["bar_schedule"]:
            dia = f"{bar['diameter_mm']}mm" if bar['diameter_mm'] else "dia unknown"
            print(f"  {bar['bar_mark']:5} | {dia:12} | annotations: {len(bar['raw_annotations'])}")

    if output_json:
        out_path = pdf_path.replace(".pdf", "_extracted.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nFull output saved to: {out_path}")


if __name__ == "__main__":
    main()
