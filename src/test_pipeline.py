"""
End-to-end pipeline test
MinoriLabs — Salit Steel Agent Practice

1. Generates 3 synthetic drawing PDFs (structural + structural revision +
   architectural) so the N-document path (N=3, two of them structural) is
   genuinely exercised — not just a pair.
2. Ingests all 3 under one project_id via ingest_project.
3. Runs one query per diagram stage (6 total) through the full pipeline:
   retrieve -> route -> generate. Prints the routing decision, matched
   chunk_types / source_doc_ids, and the generated answer (or the raw
   chunks + a clear error if ANTHROPIC_API_KEY is not set).

Usage:
    python test_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import fitz

HERE = Path(__file__).parent
TEST_DIR = HERE / "test_docs"
PROJECT = "DUNVEGAN"


# ── 1. Synthetic test PDFs ────────────────────────────────────────────────────

STRUCTURAL_R01 = [
    "DUNVEGAN WAREHOUSE — STRUCTURAL DRAWING S1 R01",
    "REINFORCED CONCRETE STRUCTURE",
    "2 STOREY BUILDING",
    "SPREAD FOOTINGS TYP.",
    "CONCRETE BEAMS AND CONCRETE COLUMNS",
    "FOOTING F1",
    "1200 x 1200",
    "QTY: 8",
    "6-20M", "B1", "16mm dia", "STRAIGHT BARS",
    "COVER: 75 mm", "SPACING @ 300", "LAP LENGTH: 600 mm",
    "STOCK LENGTH: 12 m", "HOOK LENGTH: 150 mm",
    "BEAM BM1",
    "300 x 600",
    "QTY: 12",
    "4-25M", "B2", "25mm dia", "BENT BARS",
    "STIRRUPS: 10M @ 200", "COVER: 40 mm",
    "COLUMN CL1",
    "400 x 400",
    "QTY: 6",
    "8-20M", "C1", "20mm dia",
    "TIES: 10M", "TIES LENGTH: 1.5 m", "COVER: 40 mm",
    "SECTION A-A", "ELEVATION",
]

# Revision: same marks (must dedup) + one new mark + a CONFLICTING storey count
STRUCTURAL_R02 = [
    "DUNVEGAN WAREHOUSE — STRUCTURAL DRAWING S1 R02 (REVISION)",
    "REINFORCED CONCRETE STRUCTURE",
    "3 STOREY BUILDING",
    "SPREAD FOOTINGS TYP.",
    "FOOTING F1",
    "1200 x 1200",
    "6-20M", "B1", "16mm dia",
    "COVER: 75 mm", "SPACING @ 300",
    "BEAM BM1",
    "300 x 600",
    "4-25M", "B2", "25mm dia",
    "STIRRUPS: 10M @ 200",
    "SLAB SL1",
    "200 x 6000",
    "15M @ 250", "B3", "15mm dia",
    "COVER: 30 mm", "LAP LENGTH: 500 mm",
    "SECTION B-B",
]

ARCHITECTURAL_A1 = [
    "DUNVEGAN WAREHOUSE — ARCHITECTURAL DRAWING A1",
    "FLOOR PLAN — LEVEL 1 AND LEVEL 2",
    "2 STOREY BUILDING",
    "TOTAL AREA: 2400 SQ.M",
    "40 m x 60 m",
    "FLAT ROOF",
    "ELEVATION — NORTH",
    "ELEVATED DECK AT LEVEL 2",
]


def make_pdf(path, lines):
    doc = fitz.open()
    page = doc.new_page()
    y = 60
    for line in lines:
        page.insert_text((60, y), line, fontsize=10)
        y += 18
        if y > 780:
            page = doc.new_page()
            y = 60
    doc.save(path)
    doc.close()


def build_test_docs():
    TEST_DIR.mkdir(exist_ok=True)
    specs = [
        ("structural", TEST_DIR / "DUNVEGAN_S1_R01.pdf", STRUCTURAL_R01),
        ("structural", TEST_DIR / "DUNVEGAN_S1_R02.pdf", STRUCTURAL_R02),
        ("architectural", TEST_DIR / "DUNVEGAN_A1.pdf", ARCHITECTURAL_A1),
    ]
    for _, path, lines in specs:
        make_pdf(str(path), lines)
        print(f"  wrote {path.name}")
    return [(dt, str(p)) for dt, p, _ in specs]


# ── 2 & 3. Ingest + stage queries ─────────────────────────────────────────────

STAGE_QUERIES = [
    ("STAGE 1", "What material is the structure made of, how many storeys does "
                "it have, and what is the total area? What drawing types are present?"),
    ("STAGE 2", "Verify required information: is cover, spacing, hook length, "
                "stock length, lap length, ties type, ties length and stirrup "
                "size available for the concrete elements? Cite chunk ids."),
    ("STAGE 3", "What structural elements (beams, footings, piers, columns, "
                "slabs) exist in this structure, and how many storeys are there?"),
    ("STAGE 4", "Enumerate the structural elements: number of footings, columns, "
                "beams and slabs, and per element the number of rebars, shape "
                "type, stirrups, spacing and cover."),
    ("STAGE 5", "Build a structural element register: per element the size and "
                "number of rebars, size of stirrups, bar bending type, spacing, "
                "lap length and cover thickness."),
    ("STAGE 6", "Produce an element-wise estimate summary table with type, size, "
                "length, stirrup size and weight in kg for each beam, footing, "
                "column and slab."),
]


def main():
    print("=" * 70)
    print("1. Generating 3 synthetic test PDFs (2 structural + 1 architectural)")
    print("=" * 70)
    doc_specs = build_test_docs()

    print("\n" + "=" * 70)
    print(f"2. Ingesting {len(doc_specs)} documents for project {PROJECT}")
    print("=" * 70)
    from ingest_project import ingest_project
    chunks = ingest_project(PROJECT, doc_specs)

    overview = next(c for c in chunks if c["meta"]["chunk_type"] == "structure_overview")
    print("\nStructure overview chunk:")
    print(f"  {overview['text']}")
    if overview["meta"].get("conflicts"):
        print(f"  CONFLICTS RECORDED: {overview['meta']['conflicts']}")

    n_elements = sum(1 for c in chunks if c["meta"]["chunk_type"] == "element_summary")
    print(f"\nElement summary chunks: {n_elements}")

    print("\n" + "=" * 70)
    print("3. Stage queries: retrieve -> route -> generate")
    print("=" * 70)
    from build_ragdb import query

    for stage, q in STAGE_QUERIES:
        print(f"\n{'#' * 70}\n{stage}: {q}\n{'#' * 70}")
        result = query(q, n_results=6, db="salit", generate=True)
        if result:
            sources = sorted({c["meta"].get("source_doc_id", "?")
                              for c in result["chunks"]})
            types = sorted({c["meta"].get("chunk_type", "?")
                            for c in result["chunks"]})
            print(f"\n>> matched chunk_types in results: {types}")
            print(f">> matched source_doc_ids in results: {sources}")
            if result["answer"] is None:
                print(">> NOTE: no generated answer (see error above) — "
                      "raw chunks were printed instead.")


if __name__ == "__main__":
    main()
