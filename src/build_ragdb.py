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

    # Query the DB (retrieve -> route -> generate natural-language answer)
    python build_rag_db.py --query "all 16mm diameter bars"
    python build_rag_db.py --query "how many storeys" --db salit

    # Raw chunks only (skip the LLM generation step)
    python build_rag_db.py --query "total weight of B1 bars" --no-generate

Generation runs on a LOCAL model via Ollama by default — no API key needed.
Start Ollama (`ollama serve`) and pull the model (`ollama pull qwen3:14b`).
Model defaults to qwen3:14b; override with the MODEL env var (a local tag).
Optional cloud fallback: --backend groq (requires GROQ_API_KEY).
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
import re
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# Load optional config (MODEL, OLLAMA_HOST, LLM_BACKEND, and GROQ_API_KEY for the
# optional groq backend) from the project-root .env, so settings can live in one
# place. Ollama is the default backend and needs no API key.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except Exception:
    pass


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
# 3. GENERATION LAYER
# ─────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """\
You are a structural estimation assistant for Salit Steel, helping
estimators interpret rebar and structural drawing data extracted from an
arbitrary number of project documents (architectural, structural, or
other drawing types). You act with the judgment of a level-1 civil
engineer reading construction drawings.

CONTEXT RULES:
- Answer only using the provided context chunks below. Do not use outside
  knowledge about typical construction practices unless the context is
  genuinely silent and you are making a clearly-labeled inference.
- Context chunks are tagged with chunk_type (structure_overview,
  element_summary, pdf_detail, or size_row), doc_type, and source_doc_id,
  each with a chunk_id. Match the chunk_type to the question's
  granularity. There may be any number of source documents contributing
  chunks — do not assume exactly two.
- If the context does not contain enough information to answer, say so
  explicitly and name what's missing. Never guess a number.
- If chunks from multiple documents are relevant, synthesize across all
  of them and note which document (doc_type / source_doc_id) each fact
  came from. If documents disagree, report the conflict rather than
  picking one silently.

ANSWER STYLE:
- Be precise and quantitative where the data supports it.
- For multi-part questions, answer each part in order.
- When asked to enumerate or tabulate, return a table, not prose.
- Flag ambiguity or conflicting values across chunks instead of silently
  picking one.
- COMPLETENESS: When enumerating or tabulating elements, include EVERY
  element present in the context — one row each. If a retrieved element is
  missing a field the table requires, include the row anyway with "not
  specified" in that column rather than omitting the row. A missing row in
  an estimate is a more dangerous error than a row with a blank field. Do
  not drop, merge, or skip an element just because its data is incomplete.

STAGE 1 — Understand Structural Details
Answer using structure_overview chunks: drawing type(s) present, material
(steel/concrete), total area (length/width/height), structure contents
(elevated deck, site drawing, etc.), foundation/beam/column/roof types.

STAGE 2 — Verify Required Information Is Available
List elements consisting of concrete. Then check status (present/missing/
unclear) for: 1. size of structure, 2. size of rebar, 3. type of structure,
4. cover/spacing, 5. hook length, 6. stock length, 7. lap length, 8. ties
type, 9. ties length, 10. stirrup size. For each item, cite the supporting
chunk_id, chunk_type, and source_doc_id; if unavailable, say which
document(s) were checked and found lacking.

STAGE 3 — Understand Layout and Map Spatial Assembly
Elements in the structure (beam/foundation/pier/column/concrete); type,
size, count of beams/foundations/piers; number of storeys.

STAGE 4 — Enumerate Structural Elements
Tabulate: no. of footings/columns/beams/slabs, length per floor/level.
Per element: number of rebars, shape type, stirrups used, spacing value,
cover value, stock length, lap length.

STAGE 5 — Build Structural Element Register
Per element: size and number of re-bar, size and number of stirrups,
type(s) of bar bending, spacing length, count of lap length, count of
ties, cover thickness.

STAGE 6 — Apply Details / Abstract to Element-Level Estimate
Summary table, element-wise: Beam/Footing/Column/Slab — [Type][Size]
[Length][Stirrup Size][Weight in Kg].

Given retrieved chunks and a user question, identify which stage the
question belongs to (it may span more than one), then answer using only
the relevant chunk types and format for that stage.
"""


def format_chunks_for_context(retrieved_chunks):
    """Label each chunk with chunk_id, chunk_type, doc_type, source_doc_id."""
    blocks = []
    for i, chunk in enumerate(retrieved_chunks):
        if isinstance(chunk, dict):
            meta = chunk.get("meta") or chunk.get("metadata") or {}
            text = chunk.get("text") or chunk.get("document") or ""
        else:
            meta, text = {}, str(chunk)
        blocks.append(
            f"[chunk {i+1}] "
            f"chunk_id={meta.get('chunk_id', 'n/a')} | "
            f"chunk_type={meta.get('chunk_type', meta.get('type', 'n/a'))} | "
            f"doc_type={meta.get('doc_type', 'n/a')} | "
            f"source_doc_id={meta.get('source_doc_id', meta.get('source', 'n/a'))}\n"
            f"{text}"
        )
    return "\n\n".join(blocks)


OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = "qwen3:14b"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

# qwen3 (and other reasoning models) can emit <think>...</think> preamble; strip
# it so the estimator sees only the final answer / table.
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _strip_think(text):
    return _THINK_RE.sub("", text or "").strip()


def _compose_user_message(query, retrieved_chunks):
    context = format_chunks_for_context(retrieved_chunks)
    return (
        "---\nCONTEXT CHUNKS:\n"
        f"{context}\n\n"
        "---\nUSER QUESTION:\n"
        f"{query}"
    )


def _check_ollama(model):
    """Confirm Ollama is up and the requested model is pulled."""
    import urllib.request
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=5) as r:
            tags = json.load(r)
    except Exception:
        raise RuntimeError(
            "Ollama not running — start it with `ollama serve`. "
            "Confirm the model is available with `ollama list`; "
            f"current MODEL={model} must appear in that list."
        )
    names = {m.get("name") for m in tags.get("models", [])}
    if model not in names and f"{model}:latest" not in names:
        raise RuntimeError(
            f"Ollama model '{model}' not found — pull it with `ollama pull {model}`. "
            f"Available models: {', '.join(sorted(names)) or '(none)'}."
        )


def _generate_ollama(query, retrieved_chunks):
    model = os.environ.get("MODEL", DEFAULT_OLLAMA_MODEL)
    _check_ollama(model)
    user_message = _compose_user_message(query, retrieved_chunks)
    try:
        import ollama
        client = ollama.Client(host=OLLAMA_HOST, timeout=600)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
            {"role": "user",   "content": user_message},
        ]
        opts = {"temperature": 0.2, "num_ctx": 8192}
        try:
            # think=False keeps reasoning models from spending tokens on a
            # <think> preamble; older ollama clients don't accept the kwarg.
            resp = client.chat(model=model, messages=messages, options=opts, think=False)
        except TypeError:
            resp = client.chat(model=model, messages=messages, options=opts)
        return _strip_think(resp["message"]["content"])
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Ollama call failed ({model}): {e}") from e


def _generate_groq(query, retrieved_chunks):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set — see README for Groq backend setup")
    model = os.environ.get("MODEL", DEFAULT_GROQ_MODEL)
    user_message = _compose_user_message(query, retrieved_chunks)
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
                {"role": "user",   "content": user_message},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Groq API call failed ({model}): {e}") from e


def generate_answer(query, retrieved_chunks, backend=None):
    """
    Send system prompt + labeled context + query to a local LLM (Ollama, the
    default) and return the model's natural-language answer.

    backend: "ollama" (default) or "groq". Model comes from the MODEL env var
    (default qwen3:14b for Ollama). No API key is needed for Ollama. Raises
    RuntimeError with a clear message if the backend is unreachable — the
    caller falls back to printing raw chunks.
    """
    backend = (backend or os.environ.get("LLM_BACKEND") or "ollama").lower()
    if backend == "groq":
        return _generate_groq(query, retrieved_chunks)
    return _generate_ollama(query, retrieved_chunks)


# ─────────────────────────────────────────────
# 4. QUERY
# ─────────────────────────────────────────────

def _resolve_collection(db="rebar"):
    """rebar = this script's single-PDF DB; salit = the project-level DB
    written by ingest_to_chroma.py / ingest_project.py."""
    if db == "salit":
        import ingest_to_chroma
        return ingest_to_chroma.get_collection()
    return get_collection()


def query(question, n_results=5, db="rebar", generate=True,
          doc_type=None, source_doc_id=None, backend=None):
    """
    Full pipeline: retrieve -> route -> generate.
    Prints the generated answer; falls back to the raw chunk dump when
    generation is disabled or fails.
    """
    collection = _resolve_collection(db)

    if collection.count() == 0:
        print("Database is empty. Run ingestion first.")
        return

    from query_router import retrieve
    chunks, routing = retrieve(question, collection, n_results=n_results,
                               doc_type=doc_type, source_doc_id=source_doc_id)

    print(f"\nQuery: {question}")
    print(f"Routing: chunk_types={routing['matched_chunk_types'] or 'ALL'}"
          f" | doc_type={routing['matched_doc_type'] or '-'}"
          f" | source_doc_id={routing['matched_source_doc_id'] or '-'}"
          f" | top_k={routing.get('n_results_used', n_results)}"
          f"{' | enumerate' if routing.get('enumerate_intent') else ''}"
          f"{' | (unfiltered fallback)' if routing['fallback_unfiltered'] else ''}")
    print("-" * 50)

    if generate:
        import time
        try:
            t0 = time.time()
            answer = generate_answer(question, chunks, backend=backend)
            elapsed = time.time() - t0
            print(f"\n{answer}\n")
            print(f"[generated in {elapsed:.1f}s via "
                  f"{(backend or os.environ.get('LLM_BACKEND') or 'ollama')}"
                  f" / MODEL={os.environ.get('MODEL', DEFAULT_OLLAMA_MODEL)}]")
            return {"chunks": chunks, "routing": routing, "answer": answer,
                    "elapsed": elapsed}
        except RuntimeError as e:
            print(f"\n[generation unavailable] {e}")
            print("Falling back to raw chunks:\n")

    for i, chunk in enumerate(chunks):
        meta, dist = chunk["meta"], chunk["distance"]
        relevance = round((1 - dist) * 100, 1)
        print(f"\n[{i+1}] Relevance: {relevance}% "
              f"| chunk_type: {meta.get('chunk_type', meta.get('type'))} "
              f"| source: {meta.get('source_doc_id', meta.get('source'))}")
        print(f"    {chunk['text']}")

    return {"chunks": chunks, "routing": routing, "answer": None}


# ─────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Rebar RAG DB Builder")
    parser.add_argument("--input", help="Path to extracted JSON file")
    parser.add_argument("--pdf", help="Path to PDF (runs extraction first)")
    parser.add_argument("--query", help="Query the database")
    parser.add_argument("--results", type=int, default=5, help="Number of results")
    parser.add_argument("--db", choices=["rebar", "salit"], default="rebar",
                        help="Which DB to query: rebar (single-PDF, default) "
                             "or salit (project-level N-doc DB)")
    parser.add_argument("--no-generate", action="store_true",
                        help="Skip LLM generation; print raw chunks")
    parser.add_argument("--backend", choices=["ollama", "groq"], default=None,
                        help="LLM backend for generation (default: ollama)")
    parser.add_argument("--doc-type", help="Restrict to one doc_type")
    parser.add_argument("--source-doc-id", help="Restrict to one source PDF")
    args = parser.parse_args()

    if args.query:
        query(args.query, n_results=args.results, db=args.db,
              generate=not args.no_generate, backend=args.backend,
              doc_type=args.doc_type, source_doc_id=args.source_doc_id)

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
