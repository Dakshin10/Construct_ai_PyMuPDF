"""
Project-Level N-Document Ingestion
MinoriLabs — Salit Steel Agent Practice

Ingests an ARBITRARY number of PDFs per project into ChromaDB. Each PDF
carries a free-text doc_type label (structural, architectural, mechanical,
civil-site, ...) and is routed to the right parser via PARSER_REGISTRY.
All documents share one project_id so chunks from any number of docs can be
joined at query time.

Usage:
    python ingest_project.py --project "DUNVEGAN" \
        --doc structural:"path/to/structural.pdf" \
        --doc architectural:"path/to/architectural.pdf" \
        --doc mechanical:"path/to/mechanical.pdf"

Accepts 1 to N --doc flags — nothing is hardcoded to 2.

To add a new document category (e.g. "electrical"), register a parser:
    PARSER_REGISTRY["electrical"] = my_electrical_parser
No changes to the ingestion logic are needed.
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from extract_rebar_pdf import process_pdf as process_structural_pdf
from extract_general_pdf import (
    process_pdf as process_general_pdf,
    extract_structure_facts,
    extract_elements,
)
from ingest_to_chroma import (
    get_collection,
    build_structure_overview_chunk,
    build_element_summary_chunks,
    _std_meta,
    KG_PER_M,
)

BAR_SIZE_RE = re.compile(r"\b(10M|15M|20M|25M|30M|35M|45M|55M)\b")


# ── Parsers ───────────────────────────────────────────────────────────────────
# Each parser takes (pdf_path) and returns a normalized dict:
#   {"structure_facts": {...}, "elements": [...], "bar_schedule": [...],
#    "pages": int, "page_texts": [str, ...]}

def parse_structural(pdf_path):
    """Rebar/bar-mark parsing + schedule tables + structure facts + elements."""
    raw = process_structural_pdf(pdf_path, verbose=True)  # verbose keeps text_blocks
    lines = []
    for page in raw.get("per_page", []):
        lines.extend(b["text"] for b in page.get("text_blocks", []))
    # Schedule elements (authoritative — real pier/footing/column rebar data)
    # take priority; plan-view keyword elements only fill gaps they don't cover.
    schedule_elements = raw.get("schedule_elements", [])
    covered = {e.get("element_type") for e in schedule_elements}
    rejections = []
    plan_elements = [e for e in extract_elements(lines, rejections=rejections)
                     if e.get("element_type") not in covered]
    return {
        "structure_facts": raw.get("structure_facts", {}),
        "elements": schedule_elements + plan_elements,
        "bar_schedule": raw.get("bar_schedule", []),
        "schedules_detected": raw.get("schedules_detected", []),
        "plan_instances": raw.get("plan_instances", {}),
        "footing_schedule": raw.get("footing_schedule", {}),
        "pages": raw.get("pages", 0),
        "page_texts": lines,
        "unmapped_text": raw.get("unmapped_text", []),
        "rejections": rejections,
    }


def parse_general(pdf_path):
    """General text/dimension extractor for any non-structural drawing type."""
    raw = process_general_pdf(pdf_path)
    lines = [ln for page in raw.get("per_page", []) for ln in page.get("lines", [])]
    return {
        "structure_facts": raw.get("structure_facts", {}),
        "elements": raw.get("schedule_elements", []) + raw.get("elements", []),
        "bar_schedule": [],
        "schedules_detected": raw.get("schedules_detected", []),
        "plan_instances": raw.get("plan_instances", {}),
        "footing_schedule": raw.get("footing_schedule", {}),
        "pages": raw.get("pages", 0),
        "page_texts": lines,
        "unmapped_text": raw.get("unmapped_text", []),
        "rejections": [],
    }


# doc_type -> parser. Free-text doc_types not present here fall back to the
# general parser, so any new drawing category works out of the box; register
# a dedicated parser to specialise it.
PARSER_REGISTRY = {
    "structural": parse_structural,
}
DEFAULT_PARSER = parse_general


def get_parser(doc_type):
    return PARSER_REGISTRY.get(doc_type.lower().strip(), DEFAULT_PARSER)


# ── Chunk building ────────────────────────────────────────────────────────────

def _dedupe_bar_marks(docs):
    """
    Dedup bar marks across ALL structural sources (revisions, extra sheets —
    works for any N, not just 2). Same logic as extract_rebar_pdf.deduplicate_bars
    but applied across documents.
    """
    seen = {}
    for doc in docs:
        for bar in doc["parsed"].get("bar_schedule", []):
            mark = bar.get("bar_mark")
            if not mark:
                continue
            score = sum(1 for v in bar.values() if v)
            if mark not in seen or score > seen[mark][0]:
                seen[mark] = (score, bar, doc["source_doc_id"])
    return [{"bar": bar, "source_doc_id": src} for _, bar, src in seen.values()]


def build_project_chunks(project_id, docs):
    """
    docs: list of {"doc_type", "source_doc_id", "path", "parsed"} — any length.
    Returns all 5 chunk types with full standard metadata.
    """
    chunks = []

    # 1. structure_overview — aggregated across ALL docs
    doc_facts = [{"source_doc_id": d["source_doc_id"], "doc_type": d["doc_type"],
                  "facts": d["parsed"].get("structure_facts", {})} for d in docs]
    chunks.append(build_structure_overview_chunk(project_id, doc_facts))

    # 2. element_summary — one per element, tagged with its source doc
    elements = []
    for d in docs:
        for el in d["parsed"].get("elements", []):
            elements.append({**el, "source_doc_id": d["source_doc_id"],
                             "doc_type": d["doc_type"]})
    chunks.extend(build_element_summary_chunks(project_id, elements))

    # 2b. element_inventory — plan-view INSTANCE counts (Issue 8). The
    # element_summary chunks describe schedule TYPES (F1, P1, C1 …); this chunk
    # carries how many physical instances of each appear on the plan, so
    # "how many footings/columns are there?" is answered by real quantities
    # instead of a type count. Aggregated across all docs that carry callouts.
    merged_inst = {}   # element_type -> {mark: count}
    for d in docs:
        for et, marks in (d["parsed"].get("plan_instances") or {}).items():
            bucket = merged_inst.setdefault(et, {})
            for mk, c in marks.items():
                bucket[mk] = bucket.get(mk, 0) + c
    # Per-type footing sizes from the pad schedule, merged across docs.
    foot_sched = {}
    for d in docs:
        for mk, info in (d["parsed"].get("footing_schedule") or {}).items():
            foot_sched.setdefault(mk.upper(), info)
    if merged_inst:
        parts = [f"Plan-view element instance counts for project {project_id}, "
                 f"counted from distinct foundation/framing-plan location callouts "
                 f"(F#,P# and C#,BP#). These are PHYSICAL INSTANCE quantities — the "
                 f"actual number of each element in the building — as opposed to the "
                 f"handful of schedule TYPES."]
        inv_meta = {}
        for et in ("footing", "pier", "column", "base_plate"):
            marks = merged_inst.get(et)
            if not marks:
                continue
            total = sum(marks.values())
            # For footings, fold in each type's schedule size where known.
            if et == "footing" and foot_sched:
                breakdown = ", ".join(
                    f"{mk}×{c}" + (f" [size {foot_sched[mk]['size']}"
                                   + (f", {foot_sched[mk]['reinforcing']}]"
                                      if foot_sched[mk].get('reinforcing') else "]")
                                   if mk in foot_sched and foot_sched[mk].get('size') else "")
                    for mk, c in sorted(marks.items()))
            else:
                breakdown = ", ".join(f"{mk}×{c}" for mk, c in sorted(marks.items()))
            parts.append(f"{et.replace('_',' ').capitalize()}: {total} instances "
                         f"({breakdown}).")
            inv_meta[f"{et}_instances"] = str(total)
        # Footing types present in the schedule but with no plan callout (belt-and-
        # suspenders so "list all foundation types with size" never drops one).
        if foot_sched:
            listed = merged_inst.get("footing", {})
            extra = [mk for mk in sorted(foot_sched) if mk not in listed]
            if extra:
                parts.append("Additional footing types in the pad schedule (size "
                             "known, no distinct plan callout counted): "
                             + "; ".join(
                                 f"{mk} size {foot_sched[mk]['size']}"
                                 + (f", {foot_sched[mk]['reinforcing']}"
                                    if foot_sched[mk].get('reinforcing') else "")
                                 for mk in extra if foot_sched[mk].get('size')) + ".")
        chunks.append({
            "id": f"inventory__{project_id}",
            "text": " ".join(parts),
            "meta": {
                **_std_meta(project_id, "aggregate",
                            ", ".join(d["source_doc_id"] for d in docs),
                            "element_inventory", 0),
                "type": "element_inventory",
                "project": project_id,
                **inv_meta,
            },
        })

    # 3. project_summary — one per project
    structural_docs = [d for d in docs if d["parsed"].get("bar_schedule")]
    deduped = _dedupe_bar_marks(structural_docs)
    all_marks = sorted({e["bar"]["bar_mark"] for e in deduped})
    all_sizes = sorted({m.group(1) for d in docs
                        for ln in d["parsed"].get("page_texts", [])
                        for m in BAR_SIZE_RE.finditer(ln)})
    # Bar-bending-schedule marks (e.g. "F3", "F5") are schedule row labels, a
    # different concept from element/footing identifiers (e.g. "F1", "F2",
    # "footing-6") even when they share a letter prefix. Spelling this out
    # here (not just relying on a system-prompt instruction) heads off an
    # LLM conflating "5 unique bar marks (F3, F5, ...)" with a footing count
    # — see the Q20 footing-count-inflation investigation.
    text = (
        f"Project {project_id}. "
        f"Documents ingested: {len(docs)} "
        f"({', '.join(d['source_doc_id'] + ' [' + d['doc_type'] + ']' for d in docs)}). "
        f"Unique bar-bending-schedule mark labels after cross-document dedup "
        f"(NOT element/footing identifiers, despite any shared letter prefix): "
        f"{len(deduped)} ({', '.join(all_marks) or 'none detected'}). "
        f"Bar sizes referenced: {', '.join(all_sizes) or 'none detected'}."
    )
    chunks.append({
        "id": f"proj__{project_id}",
        "text": text,
        "meta": {
            **_std_meta(project_id, "aggregate",
                        ", ".join(d["source_doc_id"] for d in docs),
                        "project_summary", 0),
            "type": "project_summary",
            "project": project_id,
            "doc_count": str(len(docs)),
            "bar_marks": ", ".join(all_marks),
            "sizes": ", ".join(all_sizes),
        },
    })

    # 4. pdf_detail — one per PDF
    for seq, d in enumerate(docs):
        marks = [b.get("bar_mark") for b in d["parsed"].get("bar_schedule", [])
                 if b.get("bar_mark")]
        text = (
            f"Drawing document {d['source_doc_id']} ({d['doc_type']}) "
            f"in project {project_id}. Pages: {d['parsed'].get('pages', '?')}. "
            f"Bar marks on this document: {', '.join(marks) or 'none'}. "
            f"Text annotations: {len(d['parsed'].get('page_texts', []))}."
        )
        chunks.append({
            "id": f"pdf__{project_id}__{d['source_doc_id']}",
            "text": text,
            "meta": {
                **_std_meta(project_id, d["doc_type"], d["source_doc_id"],
                            "pdf_detail", seq),
                "type": "pdf_detail",
                "project": project_id,
                "file": d["source_doc_id"],
            },
        })

    # 5. size_row — one per bar size per structural PDF
    size_seq = 0
    for d in docs:
        sizes_in_doc = sorted({m.group(1)
                               for ln in d["parsed"].get("page_texts", [])
                               for m in BAR_SIZE_RE.finditer(ln)})
        for sz in sizes_in_doc:
            mentions = [ln for ln in d["parsed"]["page_texts"] if sz in ln]
            text = (
                f"{sz} rebar in project {project_id}, "
                f"document {d['source_doc_id']} ({d['doc_type']}). "
                f"Unit weight: {KG_PER_M.get(sz, '?')} kg/m. "
                f"Annotations mentioning {sz}: {' | '.join(mentions[:8])}"
            )
            chunks.append({
                "id": f"size__{project_id}__{d['source_doc_id']}__{sz}",
                "text": text,
                "meta": {
                    **_std_meta(project_id, d["doc_type"], d["source_doc_id"],
                                "size_row", size_seq),
                    "type": "size_row",
                    "project": project_id,
                    "file": d["source_doc_id"],
                    "size": sz,
                },
            })
            size_seq += 1

    return chunks


# ── Ingestion ─────────────────────────────────────────────────────────────────

def _print_validation_report(project_id, docs, chunks):
    """Dry-run audit: accepted/rejected elements per document + chunk counts."""
    print(f"\n{'='*66}")
    print(f"  DRY RUN / VALIDATE — project {project_id}  (nothing written to DB)")
    print(f"{'='*66}")
    from collections import Counter
    for d in docs:
        p = d["parsed"]
        els = p.get("elements", [])
        rej = p.get("rejections", [])
        by_type = Counter(e.get("element_type") for e in els)
        print(f"\n  [{d['doc_type']}] {d['source_doc_id']}  ({p.get('pages','?')} pages)")
        print(f"    schedules detected : {', '.join(p.get('schedules_detected', [])) or 'none'}")
        print(f"    ACCEPTED elements  : {sum(by_type.values())}  {dict(by_type)}")
        for e in els:
            mat = f" [{e.get('material')}]" if e.get("material") else ""
            print(f"        + {e.get('element_type'):9} {str(e.get('label') or '-'):16}"
                  f" rebar={e.get('rebar_size') or '-'} size={e.get('size') or '-'}{mat}")
        print(f"    REJECTED candidates: {len(rej)}")
        for r in rej:
            print(f"        - {r.get('element_type'):9} {str(r.get('label') or '-'):16} "
                  f"({r.get('reason')})")
        print(f"    unmapped/garbled text spans: {len(p.get('unmapped_text', []))}")

    by_chunk = Counter(c["meta"]["chunk_type"] for c in chunks)
    print(f"\n  CHUNKS THAT WOULD BE WRITTEN ({len(chunks)} total): {dict(by_chunk)}")
    print(f"  (re-run without --dry-run to persist to ChromaDB)")


def ingest_project(project_id, doc_specs, dry_run=False):
    """
    doc_specs: list of (doc_type, pdf_path) tuples — 1 to N entries.
    dry_run:   run extraction + classification + chunk building, print an
               accepted/rejected audit per document, and DO NOT write to ChromaDB.
    """
    docs = []
    for doc_type, path in doc_specs:
        path = str(path)
        parser = get_parser(doc_type)
        print(f"\nParsing [{doc_type}] {path} via {parser.__name__} ...")
        parsed = parser(path)
        docs.append({
            "doc_type": doc_type.lower().strip(),
            "source_doc_id": Path(path).name,
            "path": path,
            "parsed": parsed,
        })

    print(f"\nBuilding chunks for project {project_id} "
          f"({len(docs)} document(s))...")
    chunks = build_project_chunks(project_id, docs)

    by_type = {}
    for c in chunks:
        by_type[c["meta"]["chunk_type"]] = by_type.get(c["meta"]["chunk_type"], 0) + 1
    for t, n in sorted(by_type.items()):
        print(f"  {t:18}: {n}")

    if dry_run:
        _print_validation_report(project_id, docs, chunks)
        return chunks

    col = get_collection()
    # Re-ingesting a project replaces its chunks (element sets can shrink)
    try:
        col.delete(where={"project_id": str(project_id)})
    except Exception:
        pass
    batch = 100
    for i in range(0, len(chunks), batch):
        b = chunks[i:i + batch]
        col.upsert(
            ids=[c["id"] for c in b],
            documents=[c["text"] for c in b],
            metadatas=[c["meta"] for c in b],
        )
    print(f"\nUpserted {len(chunks)} chunks. Collection total: {col.count()}")
    return chunks


def _parse_doc_flag(value):
    if ":" not in value:
        raise argparse.ArgumentTypeError(
            f'--doc must be doc_type:path (got "{value}")')
    doc_type, path = value.split(":", 1)
    path = path.strip('"').strip()
    if not doc_type.strip():
        raise argparse.ArgumentTypeError("--doc doc_type may not be empty")
    if not Path(path).exists():
        raise argparse.ArgumentTypeError(f"PDF not found: {path}")
    return (doc_type.strip(), path)


def main():
    p = argparse.ArgumentParser(description="Ingest N PDFs for one project")
    p.add_argument("--project", required=True, help="Shared project_id")
    p.add_argument("--doc", action="append", required=True,
                   type=_parse_doc_flag, metavar="DOC_TYPE:PATH",
                   help='e.g. structural:"drawings/S1.pdf" — repeatable, 1..N')
    p.add_argument("--dry-run", "--validate", dest="dry_run", action="store_true",
                   help="Run extraction + classification and print an accepted/"
                        "rejected audit per document WITHOUT writing to ChromaDB")
    args = p.parse_args()
    ingest_project(args.project, args.doc, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
