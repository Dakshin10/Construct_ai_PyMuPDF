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
                    "concrete_strength", "wall_type"]


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


def dedup_elements(project_id, elements):
    """
    FIX 2: collapse duplicate element_summary rows.

    1. Rows sharing (project_id, element_type, label) are merged into the
       richest one (field-union); sparser duplicates are dropped and logged.
    2. Conservative association: when a type has exactly ONE labeled group that
       carries NO rebar data and exactly ONE unlabeled record that DOES, fold
       the data into the labeled row (so e.g. columns C1,C2 get their rebar/tie
       data). Guarded so types whose labeled group already has data (footings,
       piers) keep their distinct unlabeled rows untouched.
    """
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
            print(f"  [dedup] dropped duplicate {drop.get('element_type')} "
                  f"'{label}' from {drop.get('source_doc_id','?')} "
                  f"(kept richer row for same mark)")

    labeled = list(groups.values())

    # --- Pass 2: conservative association of unlabeled data into empty labels ---
    # Only within authoritative SCHEDULE data — never fold into noisy plan-view
    # keyword elements (e.g. a "beam A325" where A325 is actually a bolt grade).
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
        # sole empty schedule label for the type + sole unlabeled schedule data
        siblings = [u for u in passthrough
                    if u.get("element_type") == et and _is_sched(u) and _has_rebar_data(u)]
        if len(empty_labeled) == 1 and len(siblings) == 1:
            _merge_element(empty_labeled[0], el)
            print(f"  [assoc] folded unlabeled {et} data "
                  f"(rebar={el.get('rebar_size')}, n={el.get('num_rebars')}) "
                  f"into '{empty_labeled[0].get('label')}'")
            continue
        remaining_unlabeled.append(el)

    return labeled + remaining_unlabeled


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
        parts = [f"{etype.capitalize()} {label} in project {project_id}."]
        for field in _ELEMENT_FIELDS:
            val = el.get(field)
            if val not in (None, "", []):
                parts.append(f"{field.replace('_', ' ').capitalize()}: {val}.")
        missing = [f for f in _ELEMENT_FIELDS if el.get(f) in (None, "", [])]
        if missing:
            parts.append(f"Not specified on drawing: {', '.join(missing)}.")

        meta = _std_meta(project_id, el.get("doc_type", "structural"),
                         el.get("source_doc_id", ""), "element_summary", seq)
        meta["element_type"] = etype
        meta["element_label"] = str(label)
        for field in _ELEMENT_FIELDS:
            meta[field] = str(el.get(field) or "")

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
