"""
Canadian Rebar RAG Ingestion
MinoriLabs — Salit Steel POC

Reads qc_results.json (from batch extraction) and loads into ChromaDB
with five chunk types optimised for estimator queries:

  project_summary    — one chunk per project (all revisions, total weight)
  pdf_detail         — one chunk per PDF (rev, sizes, weight breakdown)
  size_row           — one chunk per size per PDF (granular bar queries)
  structure_overview — one chunk per project, aggregated across ALL ingested
                       docs (material, area, storeys, foundation/beam/column/
                       roof types); conflicts noted in metadata
  element_summary    — one chunk per structural element (beam/footing/pier/
                       column/slab) with rebar, stirrup, spacing, cover,
                       stock/lap/hook lengths, ties

Every chunk carries: project_id, doc_type, source_doc_id, chunk_type,
chunk_id ({project_id}_{doc_type}_{chunk_type}_{seq}) so results can be
filtered with Chroma `where` clauses and grouped by source document.
The set of doc_types and the number of documents per project are open —
nothing here assumes a fixed count.
"""

import subprocess, sys

def _ensure_deps():
    for mod, pkg in [("chromadb","chromadb"),("sentence_transformers","sentence-transformers")]:
        try: __import__(mod)
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable,"-m","pip","install",pkg,"--break-system-packages","-q"])

_ensure_deps()

import json, chromadb
from chromadb.utils import embedding_functions
from collections import defaultdict
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
_HERE          = Path(__file__).parent
DB_PATH        = str(_HERE / "salit_chroma_db")
COLLECTION     = "salit_steel_rebar"
EMBED_MODEL    = "all-MiniLM-L6-v2"
QC_JSON        = str(_HERE / "qc_results.json")

# Canadian rebar linear weight for reverse-calculating linear metres
KG_PER_M = {"10M":0.785,"15M":1.570,"20M":2.355,"25M":3.925,"30M":5.495,"35M":7.850,"45M":11.775,"55M":19.625}

def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

def _std_meta(project_id, doc_type, source_doc_id, chunk_type, seq):
    """Standard metadata carried by EVERY chunk (all 5 types)."""
    return {
        "project_id": str(project_id),
        "doc_type": str(doc_type),
        "source_doc_id": str(source_doc_id),
        "chunk_type": str(chunk_type),
        "chunk_id": f"{project_id}_{doc_type}_{chunk_type}_{seq}",
    }


# Fields aggregated into the structure_overview chunk
_OVERVIEW_FIELDS = ["material", "storeys", "total_area", "foundation_type",
                    "beam_type", "column_type", "roof_type",
                    "concrete_strength", "wall_type", "reinforcement_notes"]


# Fields where the STRUCTURAL document is the authority: when both a structural
# and a non-structural doc report a value, the structural value wins regardless
# of parse order (an architectural CMU-wall note must not override the structural
# doc's concrete material / strength / foundation). Other fields keep
# most-recent-wins so architectural-only facts (area, storeys, ...) flow through.
_STRUCTURAL_AUTHORITY = {"material", "concrete_strength", "foundation_type"}


def build_structure_overview_chunk(project_id, doc_facts):
    """
    One chunk per project, aggregated across ALL ingested docs (any count).

    doc_facts: list of {"source_doc_id", "doc_type", "facts": {...}} in
    ingestion order. Default resolution is most-recent-wins; for the
    structural-authority fields (material, concrete_strength, foundation_type)
    the structural doc wins over non-structural docs regardless of order.
    Genuine conflicts (both docs report a differing value) are recorded in
    metadata with the winner marked — a structural-silent / architectural-filled
    gap is NOT a conflict.
    """
    merged, provenance, winner_dtype, conflicts = {}, {}, {}, {}
    for entry in doc_facts:
        facts = entry.get("facts") or {}
        dtype = entry.get("doc_type", "")
        for field in _OVERVIEW_FIELDS:
            val = facts.get(field)
            if val in (None, "", []):
                continue

            if field not in merged:
                merged[field] = val
                provenance[field] = entry["source_doc_id"]
                winner_dtype[field] = dtype
                continue

            if str(merged[field]) == str(val):
                continue  # same value, no conflict

            # Genuine conflict: both docs report a differing value.
            if field in _STRUCTURAL_AUTHORITY:
                incoming_struct = (dtype == "structural")
                current_struct = (winner_dtype[field] == "structural")
                take = incoming_struct or not current_struct  # struct wins; else recent
            else:
                take = True  # most-recent wins

            if take:
                won = (entry["source_doc_id"], val)
                other = (provenance[field], merged[field])
                merged[field] = val
                provenance[field] = entry["source_doc_id"]
                winner_dtype[field] = dtype
            else:
                won = (provenance[field], merged[field])
                other = (entry["source_doc_id"], val)

            conflicts[field] = {"won": f"{won[0]}={won[1]}",
                                "other": f"{other[0]}={other[1]}"}

    contents = sorted({c for e in doc_facts
                       for c in (e.get("facts") or {}).get("structure_contents", [])})
    doc_types = sorted({e["doc_type"] for e in doc_facts})
    source_ids = [e["source_doc_id"] for e in doc_facts]

    # Gap 3: the per-use concrete strength breakdown (which MPa applies to
    # footings vs slab-on-grade vs exterior) lives in fact_sources context, not
    # in the flat concrete_strength value. Fold it into the retrievable text so
    # a "footings vs slab-on-grade strength" question can actually be answered.
    cs_context = None
    for e in doc_facts:
        fsrc = (e.get("facts") or {}).get("fact_sources") or {}
        if fsrc.get("concrete_strength"):
            cs_context = fsrc["concrete_strength"]
            break

    parts = [f"Structure overview for project {project_id}."]
    parts.append(f"Drawing types present: {', '.join(doc_types) or 'unknown'}.")
    for field in _OVERVIEW_FIELDS:
        label = field.replace("_", " ")
        if field == "concrete_strength" and merged.get(field):
            parts.append(f"Concrete strength: {merged[field]}.")
            if cs_context:
                parts.append(f"Concrete strength by use: {cs_context}.")
        else:
            parts.append(f"{label.capitalize()}: {merged.get(field, 'not specified')}.")
    if contents:
        parts.append(f"Structure contents: {', '.join(contents)}.")
    parts.append(f"Aggregated from {len(doc_facts)} document(s): {', '.join(source_ids)}.")
    if conflicts:
        detail = "; ".join(
            f"{f} (kept {c['won']}, also saw {c['other']})"
            for f, c in sorted(conflicts.items()))
        parts.append(f"Note: conflicting values were found between documents for: {detail}.")

    meta = _std_meta(project_id, "aggregate", ", ".join(source_ids),
                     "structure_overview", 0)
    meta["contributing_doc_types"] = ", ".join(doc_types)
    for field in _OVERVIEW_FIELDS:
        meta[field] = str(merged.get(field, ""))
    meta["conflicts"] = "; ".join(
        f"{f}: kept {c['won']} vs {c['other']}" for f, c in sorted(conflicts.items())) or ""

    # Issue 3: preserve the source context for each inferred free-text fact, so
    # an estimator can trace a fact back to the sentence it came from.
    sources = {}
    for e in doc_facts:
        for field, ctx in ((e.get("facts") or {}).get("fact_sources") or {}).items():
            sources.setdefault(field, f"[{e['source_doc_id']}] {ctx}")
    meta["fact_sources"] = " || ".join(f"{f}: {c}" for f, c in sorted(sources.items()))[:1800]

    return {
        "id": f"overview__{project_id}",
        "text": " ".join(parts),
        "meta": meta,
    }


_ELEMENT_FIELDS = ["size", "count", "num_rebars", "rebar_size", "shape_type",
                   "stirrups", "spacing", "cover", "stock_length", "lap_length",
                   "hook_length", "ties_type", "ties_length", "ties_spacing",
                   "marks", "source"]

# Rebar/tie-geometry fields that don't apply to a steel (HSS) member — surfaced
# as "Not applicable", not "Not specified", so the LLM doesn't read a steel
# column as having missing data. In this project ALL columns are structural
# steel HSS (confirmed project-wide), so element_type == "column" is treated
# as steel here even when a specific record's own classify_member() verdict
# came back "concrete" from weak/incidental keyword context (e.g. a stray
# "...CENTER OF FOOTING & COLUMN 16-20M VERT..." mention inside an unrelated
# pier detail) — see Issue 7.
#
# num_rebars/rebar_size are included too (Issue 9): extract_elements()
# collapses every UNLABELED mention of an element_type into one shared
# "<type>:general" record and calls classify_member() only ONCE for it, using
# just the first 4 accumulated raw_lines — whichever keyword hit happened
# first in the document. For "column:general" that's generic notes with no
# HSS signal ("a.) All Columns..."), so the verdict comes back "concrete" and
# _STEEL_CLEARED_FIELDS (which DOES already list num_rebars/rebar_size) never
# fires, even though a later unrelated merge (the same stray "16-20M VERT."
# pier fragment) populates them. Fixing classify_member's per-record context
# would mean reworking the shared-key aggregation itself — out of scope here;
# suppressing at chunk-build time via the already-established "columns are
# steel HSS project-wide" convention is the consistent, minimal fix.
_STEEL_INAPPLICABLE_FIELDS = ("cover", "hook_length", "lap_length", "spacing",
                              "num_rebars", "rebar_size")


# Element de-dup lives in extract_general_pdf (single source of truth), used
# by both the raw extractor (run.py) and ingestion here.
from extract_general_pdf import dedup_elements


def build_element_summary_chunks(project_id, elements):
    """
    One chunk per structural element (beam/footing/pier/column/slab).

    elements: list of element dicts (see extract_general_pdf.extract_elements)
    each additionally carrying "source_doc_id" and "doc_type".
    Duplicate/sparse rows are collapsed first (FIX 2).
    """
    elements = dedup_elements(project_id, elements)
    chunks = []
    for seq, el in enumerate(elements):
        etype = el.get("element_type", "element")
        label = el.get("label") or f"{etype}-{seq}"
        # Issue 3: deterministic count — number of marks if marked (e.g. C1,C2 → 2),
        # else keep an explicitly extracted count, else 1. No LLM inference.
        # Issue 3 / A2: deterministic count. Number of marks if marked (C1,C2 → 2),
        # else 1. We do NOT trust an extracted `count` for unmarked elements — on
        # these drawings the plan-view parser grabs stray "NO. N" numbers out of
        # general-notes prose (e.g. slab-on-grade wrongly showed count=10).
        marks = el.get("marks") or []
        el["count"] = len(marks) if marks else 1
        # See _STEEL_INAPPLICABLE_FIELDS comment: this project's columns are
        # steel HSS throughout, so treat element_type == "column" as steel
        # even if this specific record's own material tag wasn't set (a weak/
        # incidental keyword match won't get the classify_member() context
        # right, but the project-level fact still holds).
        is_steel = el.get("material") == "steel" or etype == "column"

        # Issue 8: this drawing set expresses transverse/hoop reinforcement as
        # "TIES" (e.g. "10M TIES @ 300mm c/c.") — a sized "STIRRUPS ##M"
        # callout never appears (only 2 boilerplate CSA-spec mentions of the
        # bare word "stirrups" in the whole corpus, neither with a size; see
        # KNOWN_LIMITATIONS.md). Rather than chase a term this corpus doesn't
        # use, surface the already-extracted ties data under "stirrups" too —
        # via a local `effective` view, so the source `el` dict (and its
        # ties_type/ties_spacing fields) are never mutated or double-counted.
        effective = dict(el)
        stirrups_from_ties = None
        if not el.get("stirrups") and el.get("ties_type"):
            stirrups_from_ties = el["ties_type"]
            if el.get("ties_spacing"):
                stirrups_from_ties += f" @ {el['ties_spacing']}mm c/c."
            effective["stirrups"] = stirrups_from_ties

        parts = [f"{etype.capitalize()} {label} in project {project_id}."]
        # Surface steel HSS members so rebar reads as inapplicable, not missing.
        if is_steel:
            parts.append("Material: structural steel (HSS section) — rebar/cover, "
                         "lap/hook length and spacing do not apply to this member.")
        for field in _ELEMENT_FIELDS:
            if is_steel and field in _STEEL_INAPPLICABLE_FIELDS:
                continue   # reported separately below as "Not applicable"
            val = effective.get(field)
            if val not in (None, "", []):
                if field == "stirrups" and stirrups_from_ties:
                    parts.append(
                        f"Stirrups: {val} (this drawing set records transverse "
                        f"reinforcement as \"ties\", not a sized \"stirrups\" "
                        f"callout — value derived from ties_type/ties_spacing).")
                else:
                    parts.append(f"{field.replace('_', ' ').capitalize()}: {val}.")
        suppressed = set(_STEEL_INAPPLICABLE_FIELDS) if is_steel else set()
        missing = [f for f in _ELEMENT_FIELDS
                   if f not in suppressed and effective.get(f) in (None, "", [])]
        if missing:
            parts.append(f"Not specified on drawing: {', '.join(missing)}.")
        if suppressed:
            parts.append(f"Not applicable (steel member): {', '.join(sorted(suppressed))}.")

        meta = _std_meta(project_id, el.get("doc_type", "structural"),
                         el.get("source_doc_id", ""), "element_summary", seq)
        meta["element_type"] = etype
        meta["element_label"] = str(label)
        for field in _ELEMENT_FIELDS:
            if field in suppressed:
                meta[field] = "Not applicable"
            else:
                meta[field] = str(effective.get(field) or "")
        if stirrups_from_ties:
            meta["stirrups_source"] = "ties"   # flags this as derived, not a native callout

        chunks.append({
            "id": f"element__{project_id}__{el.get('source_doc_id','')}__{etype}__{label}",
            "text": " ".join(parts),
            "meta": meta,
        })
    return chunks


def build_chunks(data):
    chunks = []

    # ── 1. project_summary — aggregate all revisions per project ──────────────
    by_project = defaultdict(list)
    for d in data:
        if d.get("total_bars", 0) > 0:
            by_project[d["project"]].append(d)

    for proj, entries in by_project.items():
        total_bars   = sum(e["total_bars"] for e in entries)
        total_weight = sum(e["weight_kg"] for e in entries)
        revisions    = [e.get("rev","") for e in entries]
        all_sizes    = sorted({s for e in entries for s in (e.get("sizes") or {})})
        proj_short   = proj.replace("25-CAN-","").replace("SSS-D","").strip("-").strip()

        # Size weight totals across all revisions
        size_totals = defaultdict(float)
        for e in entries:
            for sz, sv in (e.get("sizes") or {}).items():
                size_totals[sz] += sv["weight_kg"]

        size_summary = "; ".join(
            f"{sz}: {round(wt,1)} kg ({round(wt/KG_PER_M.get(sz,1),1)} lin.m)"
            for sz, wt in sorted(size_totals.items())
        )

        text = (
            f"Project {proj_short}. "
            f"Revisions: {', '.join(revisions)}. "
            f"Total bars: {total_bars}. "
            f"Total weight: {round(total_weight,1)} kg. "
            f"Sizes used: {', '.join(all_sizes)}. "
            f"Weight by size: {size_summary}."
        )
        chunks.append({
            "id": f"proj__{proj}",
            "text": text,
            "meta": {
                **_std_meta(proj, "structural",
                            ", ".join(e["file"] for e in entries),
                            "project_summary", 0),
                "type": "project_summary",
                "project": proj,
                "project_short": proj_short,
                "total_bars": str(total_bars),
                "total_weight_kg": str(round(total_weight,1)),
                "sizes": ", ".join(all_sizes),
                "revisions": ", ".join(revisions),
            }
        })

    # ── 2. pdf_detail — one chunk per PDF ─────────────────────────────────────
    for seq, d in enumerate(data):
        if d.get("total_bars", 0) == 0:
            continue
        proj_short = d["project"].replace("25-CAN-","").replace("SSS-D","").strip("-").strip()
        sizes_text = "; ".join(
            f"{sz} qty={sv['qty']} weight={sv['weight_kg']} kg"
            f" linm={round(sv['weight_kg']/KG_PER_M.get(sz,1),1)}"
            for sz, sv in sorted((d.get("sizes") or {}).items())
        )
        xls_text = ""
        if d.get("xls_ref_kg"):
            xls_text = f" XLS reference weight: {d['xls_ref_kg']} kg. Variance: {d.get('variance_pct',0):.3f}%."

        text = (
            f"Drawing {d.get('rev','')} from project {proj_short} file {d['file']}. "
            f"Total bars: {d['total_bars']}. "
            f"Total weight: {d['weight_kg']} kg. "
            f"Breakdown: {sizes_text}."
            f"{xls_text}"
        )
        chunks.append({
            "id": f"pdf__{d['project']}__{d['file']}",
            "text": text,
            "meta": {
                **_std_meta(d["project"], "structural", d["file"],
                            "pdf_detail", seq),
                "type": "pdf_detail",
                "project": d["project"],
                "project_short": proj_short,
                "file": d["file"],
                "rev": d.get("rev",""),
                "total_bars": str(d["total_bars"]),
                "weight_kg": str(d["weight_kg"]),
                "sizes": ", ".join((d.get("sizes") or {}).keys()),
                "xls_ref_kg": str(d.get("xls_ref_kg") or ""),
                "variance_pct": str(d.get("variance_pct") or ""),
            }
        })

    # ── 3. size_row — one chunk per size per PDF ──────────────────────────────
    size_seq = 0
    for d in data:
        for sz, sv in sorted((d.get("sizes") or {}).items()):
            proj_short = d["project"].replace("25-CAN-","").replace("SSS-D","").strip("-").strip()
            linear_m   = round(sv["weight_kg"] / KG_PER_M.get(sz, 1), 1)
            text = (
                f"{sz} rebar in {proj_short} drawing {d.get('rev','')}. "
                f"Quantity: {sv['qty']} bars. "
                f"Weight: {sv['weight_kg']} kg. "
                f"Linear metres: {linear_m} m. "
                f"Unit weight: {KG_PER_M.get(sz,'?')} kg/m."
            )
            chunks.append({
                "id": f"size__{d['project']}__{d['file']}__{sz}",
                "text": text,
                "meta": {
                    **_std_meta(d["project"], "structural", d["file"],
                                "size_row", size_seq),
                    "type": "size_row",
                    "project": d["project"],
                    "project_short": proj_short,
                    "file": d["file"],
                    "rev": d.get("rev",""),
                    "size": sz,
                    "qty": str(sv["qty"]),
                    "weight_kg": str(sv["weight_kg"]),
                    "linear_m": str(linear_m),
                }
            })
            size_seq += 1
    return chunks

def ingest():
    with open(QC_JSON) as f:
        data = json.load(f)

    print(f"Building chunks from {len(data)} PDF records...")
    chunks = build_chunks(data)
    print(f"Total chunks: {len(chunks)}")
    print(f"  project_summary : {sum(1 for c in chunks if c['meta']['type']=='project_summary')}")
    print(f"  pdf_detail      : {sum(1 for c in chunks if c['meta']['type']=='pdf_detail')}")
    print(f"  size_row        : {sum(1 for c in chunks if c['meta']['type']=='size_row')}")

    col = get_collection()
    batch = 100
    for i in range(0, len(chunks), batch):
        b = chunks[i:i+batch]
        col.upsert(
            ids=[c["id"] for c in b],
            documents=[c["text"] for c in b],
            metadatas=[c["meta"] for c in b],
        )
        print(f"  Upserted batch {i//batch+1} ({len(b)} chunks)")

    print(f"\nDB saved to: {DB_PATH}")
    print(f"Total documents in collection: {col.count()}")
    return col

def query(q, n=5):
    col = get_collection()
    if col.count() == 0:
        print("DB empty — run ingest() first.")
        return
    res = col.query(query_texts=[q], n_results=min(n, col.count()),
                    include=["documents","metadatas","distances"])
    print(f"\nQuery: {q}")
    print("-"*60)
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        rel = round((1-dist)*100, 1)
        print(f"\n[{rel}%] type={meta.get('type')} | project={meta.get('project_short')} | rev={meta.get('rev','')}")
        print(f"  {doc}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ingest", action="store_true")
    p.add_argument("--query", type=str)
    p.add_argument("--results", type=int, default=5)
    args = p.parse_args()

    if args.ingest:
        ingest()
    elif args.query:
        query(args.query, args.results)
    else:
        print("Usage:  python ingest_to_chroma.py --ingest")
        print("        python ingest_to_chroma.py --query 'total 15M bars'")
