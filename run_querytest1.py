"""
RAG Pipeline — Query Test 1 Runner
Runs 10 benchmark queries through build_ragdb.py --db salit, captures output,
and writes a dated markdown report to test_runs/.

Usage:
    python run_querytest1.py
"""

import io
import sys
import os
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "test_runs")
os.makedirs(OUT_DIR, exist_ok=True)

TODAY = datetime.now().strftime("%Y-%m-%d")
OUT_FILE = os.path.join(OUT_DIR, f"query_test1_results_{TODAY}.md")

# ── The 10 queries, grouped ──────────────────────────────────────────────────
QUERY_GROUPS = [
    ("Table-Formatted Queries from Schedules", [
        "Give me the footing pad schedule with size and reinforcement.",
        "List all pier types with base plate pairing and tie spacing.",
        "Give me the concrete schedule by application.",
        "Show me the masonry wall reinforcement schedule by wall thickness.",
        "Give me the lintel schedule for non-load bearing block walls by opening size.",
        "List the steel angle lintel sizes by clear span.",
        "Give me the base plate schedule with bolt count and size.",
        "Show me the trenched footing specs by location (interior vs. exterior).",
        "Give me the grade beam specs by wall type.",
        "List all control joint (C.J.) specifications shown on the slab plan.",
    ])
]

# ── Helpers ──────────────────────────────────────────────────────────────────

_bragdb = None


def _get_build_ragdb():
    global _bragdb
    if _bragdb is None:
        sys.path.insert(0, os.path.join(ROOT, "src"))
        import build_ragdb as m
        _bragdb = m
    return _bragdb


def run_query(query_text):
    """Run one query IN-PROCESS; return (stdout, stderr, wall_seconds, returncode)."""
    m = _get_build_ragdb()
    out_buf, err_buf = io.StringIO(), io.StringIO()
    t0 = time.time()
    rc = 0
    try:
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            m.query(query_text, n_results=5, db="salit", generate=True)
    except Exception:
        err_buf.write(traceback.format_exc())
        rc = 1
    return out_buf.getvalue(), err_buf.getvalue(), time.time() - t0, rc


def extract_generation_time(stdout):
    """Pull the '[generated in Xs ...]' timing value from captured output."""
    for line in stdout.splitlines():
        if line.startswith("[generated in"):
            try:
                return float(line.split()[2].rstrip("s"))
            except Exception:
                pass
    return None


def detect_failure(stdout, stderr, returncode):
    """
    Return a failure reason string if the query failed, else None.
    Checks for: non-zero exit, fallback message, error keywords.
    """
    if returncode != 0:
        return f"Non-zero exit code {returncode}"
    combined = (stdout + stderr).lower()
    if "[generation unavailable]" in combined:
        return "Generation failed — fell back to raw chunks"
    if "falling back to raw chunks" in combined:
        return "Fell back to raw chunks"
    if "error" in combined and "traceback" in combined:
        return "Python exception / traceback"
    if stdout.strip() == "" and stderr.strip() != "":
        return "No stdout — possible crash"
    return None


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    suite_start = time.time()
    total_queries = sum(len(qs) for _, qs in QUERY_GROUPS)
    query_num = 0
    total_gen_time = 0.0
    failures = []   # list of (num, query, reason)

    lines = []  # markdown lines

    lines += [
        f"# RAG Pipeline — Query Test 1 Results",
        f"",
        f"**Date:** {TODAY}  ",
        f"**Database:** `salit` (Building C — Architectural + Structural)  ",
        f"**Model:** qwen3:14b via Ollama  ",
        f"**Total queries:** {total_queries}  ",
        f"",
        "---",
        "",
    ]

    print(f"\n{'='*65}")
    print(f"  RAG Query Test 1 — {total_queries} queries (in-process)")
    print(f"  Output: {OUT_FILE}")
    print(f"{'='*65}\n")

    # Fail fast ONCE if the generation backend is down (in-process query()
    # bypasses build_ragdb.main()'s per-invocation preflight), then warm the
    # embedding model + collection a single time up front.
    m = _get_build_ragdb()
    m._preflight_backend(None)
    print("  Loading embedding model + collection once ...")
    m._resolve_collection("salit").count()
    print("  Ready. Running queries in-process (model loaded once).\n")

    for group_name, queries in QUERY_GROUPS:
        lines += [f"## {group_name}", ""]

        for q in queries:
            query_num += 1
            cmd_str = f'python src/build_ragdb.py --query "{q}" --db salit'

            print(f"[{query_num:02d}/{total_queries}] {q[:70]}...")

            # run_query is in-process and never raises — it returns rc=1 with a
            # traceback in stderr on any error (including the ollama client's own
            # 600s generation timeout), which detect_failure() flags below.
            stdout, stderr, wall_s, rc = run_query(q)

            # Extract generation time from output
            gen_s = extract_generation_time(stdout)
            if gen_s:
                total_gen_time += gen_s

            # Detect failure
            reason = detect_failure(stdout, stderr, rc)
            if reason:
                failures.append((query_num, q, reason))
                status_tag = f"  ⚠ FAILED: {reason}"
            else:
                status_tag = f"  OK  ({gen_s:.1f}s)" if gen_s else f"  OK  ({wall_s:.1f}s wall)"

            print(status_tag)

            # Build clean output block: stdout, and stderr only if non-empty
            raw_output = stdout
            if stderr.strip():
                # Strip noisy HF loading-weights bars from stderr
                filtered_stderr = "\n".join(
                    ln for ln in stderr.splitlines()
                    if not any(skip in ln for skip in [
                        "Loading weights", "it/s]", "Warning: You are sending",
                        "HF_TOKEN", "huggingface",
                    ])
                )
                if filtered_stderr.strip():
                    raw_output += f"\n[stderr]\n{filtered_stderr}"

            lines += [
                f"### Query {query_num}",
                f"",
                f"**Command:**",
                f"```bash",
                f"{cmd_str}",
                f"```",
                f"",
                f"**Output:**",
                f"```",
                raw_output.rstrip(),
                f"```",
                f"",
                "---",
                "",
            ]

    # ── Summary section ───────────────────────────────────────────────────────
    suite_wall = time.time() - suite_start
    pass_count = total_queries - len(failures)

    lines += [
        "## Run Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total queries run | {total_queries} |",
        f"| Passed (no error / no fallback) | {pass_count} |",
        f"| Failed / fell back | {len(failures)} |",
        f"| Sum of LLM generation times | {total_gen_time:.1f} s |",
        f"| Total wall-clock time | {suite_wall:.1f} s ({suite_wall/60:.1f} min) |",
        f"| Run date | {TODAY} |",
        "",
    ]

    if failures:
        lines += ["### Failed / Degraded Queries", ""]
        for num, q, reason in failures:
            lines += [f"- **Query {num}** — *{reason}*", f"  > {q}", ""]
    else:
        lines += ["### All queries passed — no failures or fallbacks detected.", ""]

    lines += [
        "",
        "---",
        f"*Generated by `run_querytest1.py` at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
    ]

    # ── Write file ────────────────────────────────────────────────────────────
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # ── Console confirmation ──────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  DONE")
    print(f"  File   : {OUT_FILE}")
    print(f"  Queries: {total_queries}  |  Pass: {pass_count}  |  Fail: {len(failures)}")
    print(f"  LLM gen time : {total_gen_time:.1f} s")
    print(f"  Wall clock   : {suite_wall:.1f} s  ({suite_wall/60:.1f} min)")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
