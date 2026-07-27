"""
Rebar PDF Extractor — Main Runner
MinoriLabs — Salit Steel Agent Practice

Usage:
    python run.py                   # reads input/pdf_list.txt
    python run.py --verbose         # with full text block details
    python run.py --no-json         # skip JSON output
    python run.py --list input/my_list.txt  # use a custom list file

Output:
    Extracted JSON files are saved to the  output/  folder.
"""

import sys
import os
import json
import io

# Fix Windows console encoding for special characters
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── Locate project root (where this script lives) ────────────────────────────
ROOT       = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR  = os.path.join(ROOT, "input")
OUTPUT_DIR = os.path.join(ROOT, "output")
SRC_DIR    = os.path.join(ROOT, "src")
LIST_FILE  = os.path.join(INPUT_DIR, "pdf_list.txt")

# Add src/ to path so we can import the extractor
sys.path.insert(0, SRC_DIR)

# ── Ensure output folder exists ───────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)


def resolve_pdf_path(raw_name: str) -> str | None:
    """
    Try to locate a PDF given a raw entry from pdf_list.txt.
    Search order:
      1. Absolute path (as-is)
      2. Relative to input/ folder
      3. Relative to project root
    Returns the resolved absolute path, or None if not found.
    """
    candidates = [
        raw_name,                                      # absolute / already correct
        os.path.join(INPUT_DIR, raw_name),             # relative to input/
        os.path.join(ROOT, raw_name),                  # relative to project root
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return None


def read_pdf_list(list_path: str) -> list[str]:
    """Read pdf_list.txt, strip comments and blank lines."""
    if not os.path.isfile(list_path):
        print(f"[ERROR] List file not found: {list_path}")
        sys.exit(1)

    entries = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().strip('"').strip("'")
            if not line or line.startswith("#"):
                continue
            entries.append(line)
    return entries


def main():
    # ── Parse args ────────────────────────────────────────────────────────────
    args        = sys.argv[1:]
    verbose     = "--verbose"  in args
    save_json   = "--no-json"  not in args
    list_path   = LIST_FILE

    if "--list" in args:
        idx = args.index("--list")
        if idx + 1 < len(args):
            list_path = args[idx + 1]
        else:
            print("[ERROR] --list requires a path argument.")
            sys.exit(1)

    # ── Import extractors from src/ ───────────────────────────────────────────
    try:
        import fitz
        from extract_rebar_pdf import process_pdf as process_structural
        from extract_general_pdf import (
            process_pdf as process_general, classify_pdf, split_garbled,
            dedup_elements,
        )
    except ImportError as e:
        print(f"[ERROR] Could not import extractors from src/: {e}")
        sys.exit(1)

    # ── Load PDF list ─────────────────────────────────────────────────────────
    raw_entries = read_pdf_list(list_path)
    if not raw_entries:
        print(f"[WARN] No PDF entries found in: {list_path}")
        sys.exit(0)

    print(f"\n{'=' * 65}")
    print(f"  Rebar PDF Extractor")
    print(f"  List file : {list_path}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  PDFs found in list: {len(raw_entries)}")
    print(f"{'=' * 65}")

    all_results          = []
    total_bars_combined  = set()
    failed               = []
    comparison           = []   # (name, before_marks, after_marks, doc_type)

    def _prior_bar_count(base_name):
        """Read a previously-saved extraction as the 'before' snapshot."""
        prior = os.path.join(OUTPUT_DIR, f"{base_name}_extracted.json")
        if os.path.isfile(prior):
            try:
                with open(prior, encoding="utf-8") as f:
                    return len(json.load(f).get("bar_schedule", []))
            except Exception:
                return None
        return None

    # ── Process each PDF ──────────────────────────────────────────────────────
    for raw in raw_entries:
        pdf_path = resolve_pdf_path(raw)

        print(f"\n{'-' * 65}")
        if pdf_path is None:
            print(f"  [SKIP] Not found: {raw!r}")
            print(f"         Searched in input/ and project root.")
            failed.append(raw)
            continue

        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        before_marks = _prior_bar_count(base_name)

        print(f"  Processing: {os.path.basename(pdf_path)}")
        print(f"  Full path : {pdf_path}")

        # ── Classify BEFORE parsing: route rebar-parseable PDFs to the structural
        #    extractor, everything else to the general extractor (Issue 1). ──────
        try:
            doc = fitz.open(pdf_path)
            full_text = "\n".join(doc[i].get_text("text") for i in range(doc.page_count))
            doc.close()
        except Exception as e:
            print(f"  [ERROR] Could not read PDF: {e}")
            failed.append(raw)
            continue

        doc_type, score = classify_pdf(full_text)
        print(f"  Classified: {doc_type.upper()}  (rebar-indicator score={score})")
        print(f"{'-' * 65}")

        try:
            if doc_type == "structural":
                result = process_structural(pdf_path, verbose=verbose)
            else:
                result = process_general(pdf_path)
        except Exception as e:
            print(f"  [ERROR] {e}")
            failed.append(raw)
            continue

        all_results.append(result)

        # ── Per-file summary ──────────────────────────────────────────────────
        if doc_type == "structural":
            after_marks = len(result["bar_schedule"])
            print(f"\n  Summary:")
            print(f"    Pages processed     : {result['summary']['total_pages']}")
            print(f"    Confident bar marks : {result['summary']['unique_bar_marks']}")
            print(f"    Dropped (no data)   : {result['summary']['dropped_mark_count']}"
                  f"  {result['dropped_marks'] or ''}")
            print(f"    Deferred to schedule: {result['summary']['superseded_by_schedule'] or 'none'}")
            print(f"    Schedules detected  : {', '.join(result['schedules_detected']) or 'none'}")
            print(f"    Schedule elements   : {result['summary']['schedule_element_count']}")
            print(f"    Unmapped text spans : {result['summary']['unmapped_text_count']} (garbled/boilerplate)")

            if result["bar_schedule"]:
                print(f"\n  Bar Schedule (confident marks only):")
                print(f"    {'MARK':<6} {'BAR SIZE':<10} {'DIA(mm)':<9} {'COUNT':<7} {'SPACING':<12} CONF")
                print(f"    {'-'*6} {'-'*10} {'-'*9} {'-'*7} {'-'*12} ----")
                for bar in result["bar_schedule"]:
                    size    = bar.get("bar_size")    or "unknown"
                    dia     = f"{bar['diameter_mm']}mm" if bar.get("diameter_mm") else "unknown"
                    count   = str(bar.get("count"))  if bar.get("count")          else "-"
                    spacing = f"{bar['spacing_mm']}mm c/c" if bar.get("spacing_mm") else "-"
                    conf    = bar.get("confidence", "-")
                    print(f"    {bar['bar_mark']:<6} {size:<10} {dia:<9} {count:<7} {spacing:<12} {conf}")
            else:
                print(f"\n  Bar Schedule: no confident plan-view marks "
                      f"(element data comes from schedule tables below).")

            if result["schedule_elements"]:
                # Show the SAME consolidated view that gets ingested/queried:
                # collapse cross-sheet duplicate rows (e.g. two C1,C2 HSS column
                # records + an unlabeled detail row → one column). Uses the shared
                # dedup_elements so run.py can't diverge from the DB; verbose=False
                # keeps the [dedup] log out of this display.
                sched_display = dedup_elements(base_name, result["schedule_elements"],
                                               verbose=False)
                print(f"\n  Schedule Elements (from PIER/FOOTING/COLUMN/... tables):")
                print(f"    {'TYPE':<11} {'MARKS':<12} {'SIZE':<14} {'#REBAR':<7} {'REBAR':<7} {'TIES':<14}")
                print(f"    {'-'*11} {'-'*12} {'-'*14} {'-'*7} {'-'*7} {'-'*14}")
                for el in sched_display:
                    ties = el.get("ties_type") or "-"
                    if el.get("ties_spacing"):
                        ties += f" @{el['ties_spacing']}"
                    print(f"    {el['element_type']:<11} {(el.get('label') or '-'):<12} "
                          f"{(el.get('size') or '-'):<14} "
                          f"{str(el.get('num_rebars') or '-'):<7} "
                          f"{(el.get('rebar_size') or '-'):<7} {ties:<14}")

            total_bars_combined.update(result['summary']['bar_marks'])
        else:
            after_marks = 0
            facts = result.get("structure_facts", {})
            print(f"\n  Summary (non-structural — no rebar parsing):")
            print(f"    Pages processed   : {result['pages']}")
            print(f"    Rebar bar marks   : 0  (routed to general extraction)")
            print(f"    Material          : {facts.get('material') or 'n/a'}")
            print(f"    Concrete strength : {facts.get('concrete_strength') or 'n/a'}")
            print(f"    Foundation type   : {facts.get('foundation_type') or 'n/a'}")
            print(f"    Unmapped text spans (garbled/boilerplate): "
                  f"{len(result.get('unmapped_text', []))}")

        comparison.append((base_name, before_marks, after_marks, doc_type))

        # -- Save JSON to output/ ----------------------------------------------
        if save_json:
            out_path   = os.path.join(OUTPUT_DIR, f"{base_name}_extracted.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"  [OK] Saved: output/{base_name}_extracted.json")

    # -- Combined summary ------------------------------------------------------
    print(f"\n{'=' * 65}")
    print(f"  COMBINED SUMMARY")
    print(f"{'=' * 65}")
    print(f"  PDFs in list     : {len(raw_entries)}")
    print(f"  Successfully done: {len(all_results)}")
    print(f"  Failed / skipped : {len(failed)}")
    if failed:
        for f in failed:
            print(f"    [FAIL] {f}")
    print(f"  Total unique bars: {len(total_bars_combined)}")
    if total_bars_combined:
        print(f"  All bar marks    : {', '.join(sorted(total_bars_combined))}")
    if save_json and all_results:
        print(f"\n  JSON files saved to: {OUTPUT_DIR}")
    print(f"{'=' * 65}")

    # ── Before / After bar-schedule comparison ────────────────────────────────
    if comparison:
        print(f"\n  BAR SCHEDULE — BEFORE vs AFTER")
        print(f"  {'-' * 63}")
        print(f"  {'DOCUMENT':<40} {'ROUTE':<12} {'BEFORE':>7} {'AFTER':>6}")
        print(f"  {'-'*40} {'-'*12} {'-'*7} {'-'*6}")
        for name, before, after, dtype in comparison:
            b = "n/a" if before is None else str(before)
            short = (name[:37] + "...") if len(name) > 40 else name
            print(f"  {short:<40} {dtype:<12} {b:>7} {after:>6}")
        print(f"  {'-' * 63}")
        print("  (BEFORE = bar_schedule rows from the prior saved extraction;")
        print("   AFTER = rows after classification + confidence gating.)")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
