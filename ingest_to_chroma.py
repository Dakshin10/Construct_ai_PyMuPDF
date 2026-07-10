"""
Canadian Rebar RAG Ingestion
MinoriLabs — Salit Steel POC

Reads qc_results.json (from batch extraction) and loads into ChromaDB
with three chunk types optimised for estimator queries:

  project_summary  — one chunk per project (all revisions, total weight)
  pdf_detail       — one chunk per PDF (rev, sizes, weight breakdown)
  size_row         — one chunk per size per PDF (granular bar queries)
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
    for d in data:
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
