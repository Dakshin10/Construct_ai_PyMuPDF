"""
Rebar RAG Database Builder
MinoriLabs — Salit Steel Agent Practice

Takes the JSON output from extract_rebar_pdf.py and builds a
ChromaDB vector database for RAG queries.

Install dependencies first:
    pip install chromadb sentence-transformers pymupdf

Usage:
    # Build DB from extracted JSON
    python build_rag_db.py --input drawing_extracted.json

    # Build DB directly from PDF (runs extraction first)
    python build_rag_db.py --pdf your_drawing.pdf

    # Query the DB
    python build_rag_db.py --query "all 16mm diameter bars"
    python build_rag_db.py --query "total weight of B1 bars"
"""

import subprocess, sys

def _ensure_deps():
    deps = {
        "chromadb": "chromadb",
        "sentence_transformers": "sentence-transformers"
    }
    for mod, pkg in deps.items():
        try:
            __import__(mod)
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg,
                                   "--break-system-packages", "-q"])

_ensure_deps()


import json
import argparse
import os
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer


# --- Config ---
DB_PATH = "./rebar_chroma_db"
COLLECTION_NAME = "rebar_drawings"
EMBED_MODEL = "all-MiniLM-L6-v2"  # Fast, free, runs locally (~80MB)


# ─────────────────────────────────────────────
# 1. CHUNKING — semantic units, not token windows
# ─────────────────────────────────────────────

def chunk_bar_schedule(bar_schedule, source_file):
    """One chunk per bar mark — the most query-relevant unit."""
    chunks = []
    for bar in bar_schedule:
        mark = bar.get("bar_mark", "unknown")
        dia = bar.get("diameter_mm")
        annotations = bar.get("raw_annotations", [])
        count = bar.get("count")
        lengths = bar.get("lengths_mm", [])

        # Human-readable text for embedding
        text = f"Bar mark {mark}"
        if dia:
            text += f", diameter {dia}mm"
        if count:
            text += f", quantity {count}"
        if lengths:
            text += f", lengths {lengths}mm"
        if annotations:
            text += f". Notes: {' | '.join(annotations[:5])}"

        chunks.append({
            "id": f"{source_file}__bar__{mark}",
            "text": text,
            "metadata": {
                "type": "bar_mark",
                "source": source_file,
                "bar_mark": mark,
                "diameter_mm": str(dia) if dia else "",
                "count": str(count) if count else "",
            }
        })
    return chunks


def chunk_tables(per_page, source_file):
    """One chunk per table row — preserves schedule structure."""
    chunks = []
    for page_data in per_page:
        page_num = page_data.get("page", 0)
        for t_idx, table in enumerate(page_data.get("tables", [])):
            if isinstance(table, dict) and "error" in table:
                continue
            for r_idx, row in enumerate(table):
                # Flatten row dict to readable text
                row_text = ", ".join(
                    f"{k}: {v}" for k, v in row.items() if v and str(v).strip()
                )
                if row_text:
                    chunks.append({
                        "id": f"{source_file}__p{page_num}__t{t_idx}__r{r_idx}",
                        "text": row_text,
                        "metadata": {
                            "type": "table_row",
                            "source": source_file,
                            "page": str(page_num),
                        }
                    })
    return chunks


def chunk_text_blocks(per_page, source_file):
    """
    Group nearby text blocks into contextual chunks.
    Useful for drawing notes, specifications, titles.
    """
    chunks = []
    for page_data in per_page:
        page_num = page_data.get("page", 0)
        blocks = page_data.get("text_blocks", [])

        # Simple grouping: collect all text on the page as one chunk
        # (refine later by y-coordinate proximity if needed)
        if blocks:
            page_text = " | ".join(b["text"] for b in blocks if len(b["text"]) > 2)
            if page_text:
                chunks.append({
                    "id": f"{source_file}__p{page_num}__text",
                    "text": f"Page {page_num} annotations: {page_text}",
                    "metadata": {
                        "type": "page_text",
                        "source": source_file,
                        "page": str(page_num),
                    }
                })
    return chunks


def build_chunks(extracted_json, source_file):
    """Combine all chunk types."""
    chunks = []
    chunks += chunk_bar_schedule(extracted_json.get("bar_schedule", []), source_file)
    chunks += chunk_tables(extracted_json.get("per_page", []), source_file)
    chunks += chunk_text_blocks(extracted_json.get("per_page", []), source_file)
    return chunks


# ─────────────────────────────────────────────
# 2. EMBED & STORE
# ─────────────────────────────────────────────

def get_collection():
    """Get or create the ChromaDB collection."""
    client = chromadb.PersistentClient(path=DB_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def ingest(extracted_json, source_file):
    """Chunk, embed, and store extracted PDF data."""
    print(f"\nBuilding chunks from: {source_file}")
    chunks = build_chunks(extracted_json, source_file)
    print(f"  Total chunks: {len(chunks)}")

    if not chunks:
        print("  No chunks to ingest.")
        return

    collection = get_collection()

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        collection.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )
        print(f"  Stored batch {i // batch_size + 1} ({len(batch)} chunks)")

    print(f"\nDB saved to: {DB_PATH}")
    print(f"Total documents in collection: {collection.count()}")


# ─────────────────────────────────────────────
# 3. QUERY
# ─────────────────────────────────────────────

def query(question, n_results=5):
    """Query the RAG database with a natural language question."""
    collection = get_collection()

    if collection.count() == 0:
        print("Database is empty. Run ingestion first.")
        return

    results = collection.query(
        query_texts=[question],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    print(f"\nQuery: {question}")
    print("-" * 50)

    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        relevance = round((1 - dist) * 100, 1)
        print(f"\n[{i+1}] Relevance: {relevance}% | Type: {meta.get('type')} | Source: {meta.get('source')}")
        print(f"    {doc}")

    return results


# ─────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Rebar RAG DB Builder")
    parser.add_argument("--input", help="Path to extracted JSON file")
    parser.add_argument("--pdf", help="Path to PDF (runs extraction first)")
    parser.add_argument("--query", help="Query the database")
    parser.add_argument("--results", type=int, default=5, help="Number of results")
    args = parser.parse_args()

    if args.query:
        query(args.query, n_results=args.results)

    elif args.input:
        with open(args.input) as f:
            data = json.load(f)
        source = Path(args.input).stem
        ingest(data, source)

    elif args.pdf:
        # Import and run extraction first
        sys.path.insert(0, str(Path(__file__).parent))
        from extract_rebar_pdf import process_pdf
        print(f"Extracting PDF: {args.pdf}")
        data = process_pdf(args.pdf, verbose=True)
        source = Path(args.pdf).stem
        ingest(data, source)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
