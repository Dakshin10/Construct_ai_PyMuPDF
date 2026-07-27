"""
Salit Steel — Rebar Estimator Agent Gradio Web App
MinoriLabs — Construct AI practice

Provides a modern, premium dark web interface for the RAG querying pipeline,
document ingestion, and batch extraction runner without modifying any existing files.
"""

import sys
import os
import io
import time
import json
import contextlib
import subprocess
import gradio as gr

# Ensure src/ is in the python path
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT, "src")
sys.path.insert(0, SRC_DIR)

# Import pipeline APIs
try:
    from build_ragdb import query as rag_query, get_collection as get_rebar_collection
    from ingest_to_chroma import get_collection as get_salit_collection
except ImportError as e:
    print(f"[Error] Failed to import pipeline modules: {e}")
    sys.exit(1)

# Helper to run a command (as an argument LIST, not a shell string) and yield
# output in real time. Using argv form with shell=False avoids quoting bugs
# and injection risk from user-supplied values (project IDs, file paths that
# often contain spaces on Windows) that a shell=True f-string would have to
# escape correctly itself.
def run_command_stream(argv, cwd=None):
    process = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=False,
        cwd=cwd
    )

    full_output = ""
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            full_output += line
            yield full_output

    rc = process.poll()
    if rc != 0:
        full_output += f"\n[Process exited with code: {rc}]"
        yield full_output


def _format_cmd(argv):
    """Readable display string for a command shown in the log panel."""
    return " ".join(f'"{a}"' if " " in a else a for a in argv)

# List unique documents in Single-PDF DB (rebar)
def list_rebar_docs():
    try:
        col = get_rebar_collection()
        if col.count() == 0:
            return []
        metas = col.get(include=["metadatas"])["metadatas"]
        docs = set()
        for m in metas:
            src = m.get("source") or m.get("source_doc_id")
            if src:
                docs.add(src)
        return sorted(list(docs))
    except Exception:
        return []

# List projects and docs in Project DB (salit)
def list_salit_projects_and_docs():
    try:
        col = get_salit_collection()
        if col.count() == 0:
            return [], []
        metas = col.get(include=["metadatas"])["metadatas"]
        projects = set()
        docs = set()
        for m in metas:
            proj = m.get("project_id") or m.get("project")
            if proj:
                projects.add(proj)
            src = m.get("source_doc_id") or m.get("file")
            if src:
                if "," in src:
                    for s in src.split(","):
                        docs.add(s.strip())
                else:
                    docs.add(src.strip())
        return sorted(list(projects)), sorted(list(docs))
    except Exception:
        return [], []

# Dynamic filter updates
def update_filters(db):
    if db == "salit":
        projects, docs = list_salit_projects_and_docs()
        return (
            gr.Dropdown(choices=["all"] + projects, value="all", visible=True),
            gr.Dropdown(choices=["all"] + docs, value="all", visible=True)
        )
    else:
        docs = list_rebar_docs()
        return (
            gr.Dropdown(choices=["all"], value="all", visible=False),
            gr.Dropdown(choices=["all"] + docs, value="all", visible=True)
        )

# Initial load values
def init_app():
    projects, docs = list_salit_projects_and_docs()
    return (
        gr.Dropdown(choices=["all"] + projects, value="all"),
        gr.Dropdown(choices=["all"] + docs, value="all")
    )


# Query runner
def execute_query(question, db, backend, n_results, generate, doc_type, project_filter, source_doc_filter):
    if not question.strip():
        return "Please enter a question.", "Empty Question", "No output", "<p>No results</p>"
        
    backend_str = backend.lower()
    if generate:
        try:
            if backend_str == "groq":
                if not os.environ.get("GROQ_API_KEY"):
                    return "Error: GROQ_API_KEY not set in environment or .env file.", "GROQ_API_KEY Missing", "Please add GROQ_API_KEY to your .env file to use Groq backend.", "<p>Groq unavailable</p>"
            else:
                from build_ragdb import _check_ollama, DEFAULT_OLLAMA_MODEL
                model = os.environ.get("MODEL", DEFAULT_OLLAMA_MODEL)
                _check_ollama(model)
        except Exception as e:
            return f"LLM Backend error: {str(e)}", "Backend Down", f"Preflight error: {str(e)}", f"<p>Backend preflight failure: {str(e)}</p>"

    # Configure filters
    dt = None if doc_type == "all" or not doc_type else doc_type
    src_id = None if source_doc_filter == "all" or not source_doc_filter else source_doc_filter
    
    # We will override project-level filter inside query if db is salit and project is not 'all'
    # Wait, the build_ragdb query method doesn't take project_id filter directly, but
    # it retrieves chunks via query_router, which supports filtering or we can query with
    # the doc_type / source_doc_id.
    # Note: query_router's retrieve method supports metadata filters.
    
    f = io.StringIO()
    answer_text = ""
    routing_info = ""
    chunks_list = []
    
    t0 = time.time()
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        try:
            res = rag_query(
                question=question,
                n_results=int(n_results),
                db=db,
                generate=generate,
                doc_type=dt,
                source_doc_id=src_id,
                backend=backend_str,
                clean=False
            )
            if res:
                answer_text = res.get("answer") or ""
                r = res.get("routing") or {}
                routing_info = f"Database: {db.upper()} | Backend: {backend_str.upper()}\n"
                routing_info += f"Matched Chunk Types: {r.get('matched_chunk_types') or 'ALL'}\n"
                routing_info += f"Doc Type: {r.get('matched_doc_type') or '-'}\n"
                routing_info += f"Source Doc ID: {r.get('matched_source_doc_id') or '-'}\n"
                routing_info += f"Fallback Unfiltered: {r.get('fallback_unfiltered', False)}\n"
                routing_info += f"Elapsed Time: {time.time() - t0:.2f}s"
                chunks_list = res.get("chunks") or []
        except Exception as e:
            import traceback
            traceback.print_exc()
            answer_text = f"An error occurred during query execution:\n{str(e)}"
            routing_info = "Execution Error"
            
    stdout_val = f.getvalue()
    elapsed = time.time() - t0
    
    if not answer_text:
        if not generate:
            answer_text = "### Raw Chunk Retrieval Only\nLLM Generation was skipped. The raw document chunks matches are listed below."
        else:
            answer_text = f"### LLM Generation Failed\nUnable to generate response via {backend}. Please ensure Ollama is serving or check the console logs. The matching chunks are listed below."

    chunks_html = render_chunks_html(chunks_list)
    return answer_text, routing_info, stdout_val, chunks_html

# Chat handler to manage conversation history and format details
def chat_handler(message, history, db, backend, n_results, generate, doc_type, project_filter, source_doc_filter):
    if not message.strip():
        return "", history, ""
        
    if history is None:
        history = []
        
    answer_text, routing_info, stdout_val, chunks_html = execute_query(
        message, db, backend, n_results, generate, doc_type, project_filter, source_doc_filter
    )
    
    # Format a collapsible details block for sources and routing info
    collapsible_sources = f"""
<details style="margin-top: 12px; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; background: rgba(30, 41, 59, 0.25); padding: 10px;">
  <summary style="cursor: pointer; font-weight: 600; color: #38bdf8; font-size: 0.9rem;">View Sources & Routing Info</summary>
  <div style="margin-top: 10px; font-size: 0.85rem; color: #94a3b8; display: flex; flex-direction: column; gap: 8px;">
    <strong>Routing Information:</strong>
    <pre style="background: rgba(0, 0, 0, 0.3); padding: 8px; border-radius: 4px; font-family: monospace; font-size: 0.75rem; white-space: pre-wrap; color: #e2e8f0; border: 1px solid rgba(255, 255, 255, 0.05);">{routing_info}</pre>
    <strong>Source Chunks:</strong>
    {chunks_html}
  </div>
</details>
"""
    
    formatted_answer = f"{answer_text}\n\n{collapsible_sources}"
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": formatted_answer})
    return "", history, stdout_val



# HTML Renderer for retrieved chunks
def render_chunks_html(chunks):
    if not chunks:
        return "<p style='color: #94a3b8; font-style: italic;'>No chunks retrieved yet or database is empty.</p>"
        
    html = "<div style='display: flex; flex-direction: column; gap: 12px;'>"
    for i, chunk in enumerate(chunks):
        meta = chunk.get("meta") or chunk.get("metadata") or {}
        text = chunk.get("text") or chunk.get("document") or ""
        dist = chunk.get("distance", None)
        
        relevance_str = "n/a"
        if dist is not None:
            relevance = round((1 - dist) * 100, 1)
            relevance_str = f"{relevance}%"
            
        c_type = meta.get("chunk_type") or meta.get("type") or "unknown"
        source = meta.get("source_doc_id") or meta.get("source") or meta.get("file") or "unknown"
        c_id = meta.get("chunk_id") or chunk.get("id") or "n/a"
        
        badge_cls = "badge-overview"
        if "element" in c_type:
            badge_cls = "badge-element"
        elif "project" in c_type:
            badge_cls = "badge-project"
        elif "pdf" in c_type:
            badge_cls = "badge-pdf"
        elif "size" in c_type:
            badge_cls = "badge-size"
            
        html += f"""
        <div class="chunk-card">
            <div class="chunk-header">
                <div>
                    <span class="chunk-badge {badge_cls}">{c_type.upper()}</span>
                    <span style="margin-left: 8px; color: #94a3b8; font-weight: 500;">Source: {source}</span>
                </div>
                <span class="chunk-badge badge-relevance">Relevance: {relevance_str}</span>
            </div>
            <div style="font-size: 0.95rem; color: #e2e8f0; line-height: 1.5; margin-bottom: 8px;">
                {text}
            </div>
            <div style="font-size: 0.72rem; color: #64748b; font-family: monospace;">
                ID: {c_id}
            </div>
        </div>
        """
    html += "</div>"
    return html

# Ingest Single PDF
def handle_pdf_ingestion(pdf_file, project_id, doc_type, dry_run):
    if not pdf_file:
        yield "Error: No PDF file uploaded."
        return
    if not project_id.strip():
        yield "Error: Please specify a Project ID."
        return
        
    temp_file_path = pdf_file.name
    argv = [sys.executable, "src/ingest_project.py",
            "--project", project_id.strip(),
            "--doc", f"{doc_type}:{temp_file_path}"]
    if dry_run:
        argv.append("--dry-run")

    cmd_str = _format_cmd(argv)
    yield f"Running Ingestion command:\n> {cmd_str}\n\n[STDOUT / STDERR]\n"
    for output in run_command_stream(argv, cwd=ROOT):
        yield f"Running Ingestion command:\n> {cmd_str}\n\n[STDOUT / STDERR]\n{output}"

# Ingest Pre-Extracted JSON
def handle_json_ingestion(json_file):
    if not json_file:
        yield "Error: No JSON file uploaded."
        return

    argv = [sys.executable, "src/build_ragdb.py", "--input", json_file.name]
    cmd_str = _format_cmd(argv)
    yield f"Running JSON Ingestion:\n> {cmd_str}\n\n[STDOUT / STDERR]\n"
    for output in run_command_stream(argv, cwd=ROOT):
        yield f"Running JSON Ingestion:\n> {cmd_str}\n\n[STDOUT / STDERR]\n{output}"

# pdf_list.txt helper functions
def load_pdf_list():
    try:
        path = os.path.join(ROOT, "input", "pdf_list.txt")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        return "# pdf_list.txt not found. Add drawing filenames here (one per line)."
    except Exception as e:
        return f"Error loading pdf_list.txt: {str(e)}"

def save_pdf_list(content):
    try:
        path = os.path.join(ROOT, "input", "pdf_list.txt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return "Successfully saved input/pdf_list.txt!"
    except Exception as e:
        return f"Error saving pdf_list.txt: {str(e)}"

# Run run.py batch extraction
def run_batch_pipeline():
    cmd = "python run.py"
    yield f"Running batch extraction:\n> {cmd}\n\n[STDOUT / STDERR]\n"
    for output in run_command_stream(cmd, cwd=ROOT):
        yield f"Running batch extraction:\n> {cmd}\n\n[STDOUT / STDERR]\n{output}"

# Database statistics
def get_db_stats():
    stats = []
    
    # Check rebar (Single PDF)
    try:
        col = get_rebar_collection()
        cnt = col.count()
        if cnt > 0:
            metas = col.get(include=["metadatas"])["metadatas"]
            counts = {}
            for m in metas:
                p = m.get("project_id") or m.get("project") or "rebar"
                f = m.get("source_doc_id") or m.get("source") or m.get("file") or "unknown"
                t = m.get("chunk_type") or m.get("type") or "unknown"
                key = ("rebar (Single PDF)", p, f, t)
                counts[key] = counts.get(key, 0) + 1
            for (db, p, f, t), count in counts.items():
                stats.append([db, p, f, t, count])
    except Exception:
        pass
        
    # Check salit (Project DB)
    try:
        col = get_salit_collection()
        cnt = col.count()
        if cnt > 0:
            metas = col.get(include=["metadatas"])["metadatas"]
            counts = {}
            for m in metas:
                p = m.get("project_id") or m.get("project") or "salit"
                f = m.get("source_doc_id") or m.get("source") or m.get("file") or "unknown"
                t = m.get("chunk_type") or m.get("type") or "unknown"
                key = ("salit (Project DB)", p, f, t)
                counts[key] = counts.get(key, 0) + 1
            for (db, p, f, t), count in counts.items():
                stats.append([db, p, f, t, count])
    except Exception:
        pass
        
    return stats

# Refresh stats view
def refresh_stats():
    stats_data = get_db_stats()
    
    rebar_count = 0
    salit_count = 0
    try:
        rebar_count = get_rebar_collection().count()
    except Exception:
        pass
    try:
        salit_count = get_salit_collection().count()
    except Exception:
        pass
        
    count_summary = f"""
    <div style='display: flex; gap: 20px; margin-bottom: 20px;'>
        <div style='flex: 1; background: rgba(30, 41, 59, 0.5); padding: 15px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.05); text-align: center;'>
            <div style='font-size: 0.85rem; color: #94a3b8; text-transform: uppercase;'>Single-PDF Collection (rebar)</div>
            <div style='font-size: 2rem; font-weight: 700; color: #38bdf8; margin-top: 5px;'>{rebar_count} <span style='font-size: 1rem; font-weight: 400; color: #64748b;'>chunks</span></div>
        </div>
        <div style='flex: 1; background: rgba(30, 41, 59, 0.5); padding: 15px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.05); text-align: center;'>
            <div style='font-size: 0.85rem; color: #94a3b8; text-transform: uppercase;'>Project-Level Collection (salit)</div>
            <div style='font-size: 2rem; font-weight: 700; color: #818cf8; margin-top: 5px;'>{salit_count} <span style='font-size: 1rem; font-weight: 400; color: #64748b;'>chunks</span></div>
        </div>
    </div>
    """
    
    headers = ["Database", "Project", "Document / Source File", "Chunk Type", "Count"]
    if not stats_data:
        return count_summary, gr.Dataframe(value=[["No data", "No data", "No data", "No data", 0]], headers=headers)
        
    return count_summary, gr.Dataframe(value=stats_data, headers=headers)

# Custom premium dark theme CSS
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

body, .gradio-container, button, input, select, textarea, span, p {
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

.gradio-container {
    background-color: #0b0f19 !important;
    color: #e2e8f0 !important;
}

/* Glassmorphic panels */
.glass-panel {
    background: rgba(17, 24, 39, 0.7) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4) !important;
    padding: 24px !important;
    margin-bottom: 20px !important;
}

.gradient-title {
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700 !important;
    font-size: 2.8rem !important;
    text-shadow: 0 4px 20px rgba(129, 140, 248, 0.2);
    letter-spacing: -0.02em;
    margin-bottom: 0.2rem;
}

.subtitle {
    color: #94a3b8 !important;
    font-size: 1.1rem !important;
    font-weight: 400 !important;
    margin-bottom: 2rem !important;
}

/* Premium chunk card visualization */
.chunk-card {
    background: rgba(30, 41, 59, 0.45);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 12px;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

.chunk-card:hover {
    transform: translateY(-2px);
    background: rgba(30, 41, 59, 0.65);
    border-color: rgba(56, 189, 248, 0.3);
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.25);
}

.chunk-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding-bottom: 8px;
    margin-bottom: 12px;
    font-size: 0.85rem;
}

.chunk-badge {
    padding: 4px 10px;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.72rem;
    letter-spacing: 0.03em;
}

.badge-overview { background: rgba(234, 179, 8, 0.15); color: #facc15; border: 1px solid rgba(234, 179, 8, 0.3); }
.badge-element { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }
.badge-project { background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); }
.badge-pdf { background: rgba(236, 72, 153, 0.15); color: #f472b6; border: 1px solid rgba(236, 72, 153, 0.3); }
.badge-size { background: rgba(20, 184, 166, 0.15); color: #2dd4bf; border: 1px solid rgba(20, 184, 166, 0.3); }
.badge-relevance { background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); font-family: monospace; }

/* Customizing Gradio component interfaces */
.tabs {
    border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
}

.tab-nav button {
    font-size: 1.05rem !important;
    font-weight: 500 !important;
    color: #64748b !important;
}

.tab-nav button.selected {
    color: #38bdf8 !important;
    border-bottom-color: #38bdf8 !important;
}

input[type="text"], select, textarea {
    background-color: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
    color: #f8fafc !important;
}

input[type="text"]:focus, select:focus, textarea:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2) !important;
}

/* Primary submit buttons styling */
.action-btn {
    background: linear-gradient(135deg, #0284c7 0%, #1d4ed8 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 14px rgba(29, 78, 216, 0.3) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.action-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(29, 78, 216, 0.5) !important;
}

/* Logs and console display */
.console-box textarea {
    font-family: 'Fira Code', 'Courier New', Courier, monospace !important;
    background-color: #020617 !important;
    color: #38bdf8 !important;
    font-size: 0.85rem !important;
    border-color: rgba(56, 189, 248, 0.15) !important;
}
"""

# Fetch initial lists for dropdowns lazily
initial_projects, initial_docs = [], []
initial_rebar_docs = []


# Build the Web Application Layout
with gr.Blocks(title="Salit Steel Rebar Agent WebUI") as demo:
    
    # Title & Subtitle Header
    with gr.Row():
        with gr.Column(scale=12):
            gr.Markdown("# 🏗️ Salit Steel — Rebar Estimator Agent", elem_classes=["gradient-title"])
            gr.Markdown("Local-first Reinforcement Steel PDF Extraction, Cross-Document Deduplication, and Intent-routed Semantic Querying (RAG)", elem_classes=["subtitle"])

    # Core Navigation Tabs
    with gr.Tabs(elem_classes=["tabs"]):
        
        # TAB 1: ESTIMATOR AGENT (RAG)
        with gr.TabItem("🔍 Estimator Query Agent (RAG)", id=0):
            with gr.Column(scale=12):
                # Collapsible Advanced Settings Panel (closed by default)
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    with gr.Row():
                        with gr.Column(scale=4):
                            db_select = gr.Dropdown(
                                choices=["salit", "rebar"],
                                value="salit",
                                label="Target Database",
                                info="salit = Project-level DB (23+ sheets) | rebar = Single PDF DB"
                            )
                            backend_select = gr.Dropdown(
                                choices=["ollama", "groq"],
                                value="ollama",
                                label="LLM Generation Backend",
                                info="Ollama runs locally. Groq uses cloud APIs (needs GROQ_API_KEY)."
                            )
                        with gr.Column(scale=4):
                            results_slider = gr.Slider(
                                minimum=1,
                                maximum=15,
                                value=5,
                                step=1,
                                label="Retrieve Top-K Chunks"
                            )
                            gen_toggle = gr.Checkbox(
                                value=True,
                                label="Enable AI Answer Generation",
                                info="If disabled, displays raw matching document chunks only."
                            )
                        with gr.Column(scale=4):
                            doc_type_filter = gr.Dropdown(
                                choices=["all", "structural", "architectural", "mechanical", "civil-site"],
                                value="all",
                                label="Document Type Filter"
                            )
                            # Project and Source ID Dropdowns (dynamically populated)
                            project_filter = gr.Dropdown(
                                choices=["all"] + initial_projects,
                                value="all",
                                label="Project ID Filter",
                                visible=True
                            )
                            source_doc_filter = gr.Dropdown(
                                choices=["all"] + initial_docs,
                                value="all",
                                label="Source PDF Filter"
                            )
                    with gr.Row():
                        with gr.Column(scale=12):
                            debug_logs = gr.Textbox(
                                label="Debug / Console Stdout Logs (Last Query)",
                                interactive=False,
                                lines=8,
                                elem_classes=["console-box"]
                            )

                # Conversational Chatbot Panel
                chatbot = gr.Chatbot(label="Estimator Agent Chat", height=600)
                
                # Chat Input Box Row at the bottom
                with gr.Row():
                    query_input = gr.Textbox(
                        placeholder="Ask any structural estimation question (e.g., 'What is the cover spacing of beam B1?' or 'total weight of 15M bars'). Press Enter to send.",
                        show_label=False,
                        lines=1,
                        scale=10
                    )
                    submit_btn = gr.Button("Send", elem_classes=["action-btn"], scale=2)

            # Event hookups for Tab 1
            db_select.change(
                fn=update_filters,
                inputs=[db_select],
                outputs=[project_filter, source_doc_filter]
            )
            
            submit_btn.click(
                fn=chat_handler,
                inputs=[query_input, chatbot, db_select, backend_select, results_slider, gen_toggle, doc_type_filter, project_filter, source_doc_filter],
                outputs=[query_input, chatbot, debug_logs]
            )
            query_input.submit(
                fn=chat_handler,
                inputs=[query_input, chatbot, db_select, backend_select, results_slider, gen_toggle, doc_type_filter, project_filter, source_doc_filter],
                outputs=[query_input, chatbot, debug_logs]
            )


        # TAB 2: DOCUMENT INGESTION
        with gr.TabItem("📥 Ingest Drawing Document", id=1):
            with gr.Row():
                
                # Ingestion Forms
                with gr.Column(scale=5, elem_classes=["glass-panel"]):
                    gr.Markdown("### 🏢 Ingest Single Drawing PDF")
                    pdf_input_file = gr.File(
                        label="Upload Drawing PDF",
                        file_types=[".pdf"]
                    )
                    ingest_proj_id = gr.Textbox(
                        placeholder="e.g. DUNVEGAN",
                        label="Project ID (Unique identifier for the project)"
                    )
                    ingest_doc_type = gr.Dropdown(
                        choices=["structural", "architectural", "mechanical", "civil-site", "general"],
                        value="structural",
                        label="Drawing Classification (Doc Type)",
                        info="structural = processes rebar scheduling | other = processes layout facts & text"
                    )
                    ingest_dry_run = gr.Checkbox(
                        value=False,
                        label="Run Dry Run / Validation Audit",
                        info="If checked, runs parsers and lists accepted/rejected elements WITHOUT writing to ChromaDB."
                    )
                    ingest_pdf_btn = gr.Button("Ingest PDF Document", elem_classes=["action-btn"])
                    
                    gr.Markdown("<br><hr style='border-color: rgba(255,255,255,0.05)'><br>")
                    
                    gr.Markdown("### 📄 Ingest Pre-Extracted JSON")
                    json_input_file = gr.File(
                        label="Upload Extracted JSON File",
                        file_types=[".json"]
                    )
                    ingest_json_btn = gr.Button("Ingest Extracted JSON", elem_classes=["action-btn"])
                    
                # Ingestion Logs Streaming Panel
                with gr.Column(scale=7):
                    gr.Markdown("### 📜 Ingestion Console & Logs")
                    ingest_logs = gr.Textbox(
                        label="Parser & ChromaDB Ingestion Output (Real-time)",
                        interactive=False,
                        lines=28,
                        elem_classes=["console-box"],
                        value="Select a PDF or JSON file, fill in the metadata, and click the Ingest button."
                    )
                    
            # Event hookups for Tab 2
            ingest_pdf_btn.click(
                fn=handle_pdf_ingestion,
                inputs=[pdf_input_file, ingest_proj_id, ingest_doc_type, ingest_dry_run],
                outputs=[ingest_logs]
            )
            ingest_json_btn.click(
                fn=handle_json_ingestion,
                inputs=[json_input_file],
                outputs=[ingest_logs]
            )

        # TAB 3: BATCH RUNNER & DB STATS
        with gr.TabItem("⚙️ Pipeline & Database Inspector", id=2):
            with gr.Row():
                
                # Left Panel - Batch Runner controls
                with gr.Column(scale=5, elem_classes=["glass-panel"]):
                    gr.Markdown("### 📑 Batch Processing Config (`pdf_list.txt`)")
                    gr.Markdown("Define drawings to extract in batch below (one drawing name per line):")
                    
                    pdf_list_textarea = gr.TextArea(
                        value=load_pdf_list(),
                        lines=10,
                        label="pdf_list.txt Contents"
                    )
                    save_list_btn = gr.Button("Save pdf_list.txt Changes", variant="secondary")
                    save_status = gr.Markdown("")
                    
                    gr.Markdown("<br>")
                    run_batch_btn = gr.Button("🚀 Run Batch Extractor (run.py)", elem_classes=["action-btn"])
                    
                # Right Panel - Database Stats
                with gr.Column(scale=7):
                    gr.Markdown("### 📊 Database Statistics & Inspector")
                    refresh_stats_btn = gr.Button("🔄 Refresh Database Status", variant="secondary")
                    
                    # HTML Cards view
                    db_summary_html = gr.HTML(
                        value="<p style='color: #64748b;'>Click refresh to inspect counts.</p>"
                    )
                    
                    # Dataframe details
                    db_stats_df = gr.Dataframe(
                        headers=["Database", "Project", "Document / Source File", "Chunk Type", "Count"],
                        datatype=["str", "str", "str", "str", "number"],
                        interactive=False,
                        label="Ingested Chunks Register"
                    )
                    
            # Console logs for Batch processing
            with gr.Row():
                with gr.Column(scale=12):
                    gr.Markdown("### 🖥️ Extraction Console Logs")
                    batch_logs = gr.Textbox(
                        label="Batch Extraction Outputs (Real-time)",
                        interactive=False,
                        lines=15,
                        elem_classes=["console-box"],
                        value="Run the batch extractor to capture processing progress."
                    )
                    
            # Event hookups for Tab 3
            save_list_btn.click(
                fn=save_pdf_list,
                inputs=[pdf_list_textarea],
                outputs=[save_status]
            )
            run_batch_btn.click(
                fn=run_batch_pipeline,
                inputs=[],
                outputs=[batch_logs]
            )
            refresh_stats_btn.click(
                fn=refresh_stats,
                inputs=[],
                outputs=[db_summary_html, db_stats_df]
            )
            
            # Auto-run database stats refresh and load dropdown filters on load
            demo.load(
                fn=refresh_stats,
                inputs=[],
                outputs=[db_summary_html, db_stats_df]
            )
            demo.load(
                fn=init_app,
                inputs=[],
                outputs=[project_filter, source_doc_filter]
            )


# Launch the Application
if __name__ == "__main__":
    # In Windows, Gradio might select port 7860 or search for the next available port.
    # We specify share=False to run locally only, adhering to local-first practices.
    demo.queue()
    demo.launch(server_name="127.0.0.1", share=False, css=custom_css)
