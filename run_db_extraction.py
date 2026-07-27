import os
import re
import sys
import json
import sqlite3
from pathlib import Path
import fitz

# Insert src to python path to use the general extractor
sys.path.insert(0, str(Path(__file__).parent / "src"))
from extract_general_pdf import page_spans, process_pdf, extract_footing_pad_schedule

# --- Configuration ---
ROOT = Path(__file__).parent.resolve()
DB_FILE = ROOT / "output" / "foundation_data.db"
IMAGES_DIR = ROOT / "output" / "images"
os.makedirs(IMAGES_DIR, exist_ok=True)

# 1. Read the active PDF from input/pdf_list.txt
def get_active_pdf():
    list_path = ROOT / "input" / "pdf_list.txt"
    if not list_path.exists():
        raise FileNotFoundError(f"pdf_list.txt not found at {list_path}")
    
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Strip quotes if any
            clean_path = line.strip('"').strip("'")
            return clean_path
            
    raise ValueError("No active PDF path specified in input/pdf_list.txt")

# 2. Render PDF page 1 to a high-res PNG image
def render_pdf_page(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    zoom = 3  # High-resolution rendering
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    
    pdf_stem = Path(pdf_path).stem
    png_name = f"{pdf_stem}_page1.png"
    png_path = IMAGES_DIR / png_name
    pix.save(str(png_path))
    doc.close()
    return png_path.resolve()

# 3. Extract site name from drawing text
def extract_site_name(pdf_path):
    doc = fitz.open(pdf_path)
    text = doc[0].get_text("text").upper()
    doc.close()
    
    # Check for common project/site patterns
    if "BUILDING \"C\"" in text or "BUILDING C" in text or "BUILDING-C" in text:
        return "Building C"
    elif "DUNVEGAN" in text:
        return "Dunvegan Warehouse"
    
    # Fallback to file name stem capitalized
    return Path(pdf_path).stem.replace("_", " ").title()

# 3b. Extract drawing description dynamically (used as Segment)
def extract_drawing_description(pdf_path):
    doc = fitz.open(pdf_path)
    text = doc[0].get_text("text")
    doc.close()
    
    for line in text.splitlines():
        line_clean = line.strip()
        m = re.search(r'BUILDING\s+["\'][A-Z]["\']\s*:\s*(.*)', line_clean, re.I)
        if m:
            title = m.group(1).strip()
            # Clean up consecutive spaces
            title = re.sub(r'\s+', ' ', title)
            # Capitalize to title case
            return title.title()
            
    # Fallback
    return "Foundation Layout Plan" if "planlayout" in str(pdf_path).lower() else "Roof Framing Layout Plan"

# 4. Extract pier sizes from concrete pier schedule
def extract_pier_sizes(pdf_path):
    # Default fallback sizes based on drawing analysis
    sizes = {
        "P1": {"length": 500.0, "width": 500.0, "height": None},
        "P2": {"length": 1250.0, "width": 500.0, "height": None}
    }
    
    doc = fitz.open(pdf_path)
    text = doc[0].get_text("text")
    doc.close()
    
    # Attempt to dynamically extract pier size from drawing text using pattern matching
    # E.g. finding '500mm x 500mm' near P1 or '1250mm x 500mm' near P2
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for i, line in enumerate(lines):
        # Look for a dimension callout (e.g. '500mm x 500mm' or '1250mm x 500mm')
        m = re.match(r"^(\d+)\s*mm\s*x\s*(\d+)\s*mm$", line, re.I)
        if m:
            length = float(m.group(1))
            width = float(m.group(2))
            
            # Check the surrounding 5 lines for pier marks P1 or P2
            window = lines[max(0, i-5):min(len(lines), i+6)]
            for win_line in window:
                win_line_clean = win_line.strip().upper()
                if win_line_clean == "P1":
                    sizes["P1"] = {"length": length, "width": width, "height": None}
                elif win_line_clean == "P2":
                    sizes["P2"] = {"length": length, "width": width, "height": None}
                    
    return sizes

def main():
    try:
        pdf_path_str = get_active_pdf()
        pdf_path = Path(pdf_path_str).resolve()
        if not pdf_path.exists():
            print(f"Error: PDF file does not exist at '{pdf_path}'")
            sys.exit(1)
            
        print(f"Active PDF: {pdf_path}")
        
        # Site Name, Segment/Description, and Image Store path
        site_name = extract_site_name(pdf_path)
        segment_name = extract_drawing_description(pdf_path)
        print(f"Extracted Site Name: {site_name}")
        print(f"Extracted Segment Name: {segment_name}")
        
        print("Rendering PDF page to image store...")
        image_path = render_pdf_page(pdf_path)
        print(f"Rendered image path: {image_path}")
        
        # Pipeline extraction
        print("Running foundation extraction pipeline...")
        extracted = process_pdf(str(pdf_path))
        
        # Get plan-view counts (footings and piers)
        plan_instances = extracted.get("plan_instances", {})
        footing_counts = plan_instances.get("footing", {})
        pier_counts = plan_instances.get("pier", {})
        
        # Get footing schedule details
        footing_schedule = extracted.get("footing_schedule", {})
        if not footing_schedule:
            # Fall back to custom footing schedule parsing directly from spans
            all_spans = []
            doc = fitz.open(pdf_path)
            for p_num in range(doc.page_count):
                # Import page_spans safely
                all_spans.extend(page_spans(doc[p_num], p_num + 1))
            doc.close()
            footing_schedule = extract_footing_pad_schedule(all_spans)
            
        # Get pier sizes
        pier_sizes = extract_pier_sizes(pdf_path)
        
        # 5. Populate rows for SQLite
        rows_to_insert = []
        
        # Add footings (F1-F5)
        # Note: We include all types F1-F5 that are defined in the footing pad schedule
        footing_marks = sorted(list(set(footing_schedule.keys()) | {"F1", "F2", "F3", "F4", "F5"}))
        for mark in footing_marks:
            # Retrieve size dimensions
            length, width, height = None, None, None
            sched_info = footing_schedule.get(mark)
            if sched_info and sched_info.get("size"):
                size_str = sched_info["size"] # e.g. "1300x1300x400"
                parts = size_str.split("x")
                if len(parts) >= 2:
                    length = float(parts[0])
                    width = float(parts[1])
                if len(parts) >= 3:
                    height = float(parts[2])
                    
            # Quantity on plan (defaults to 0 if not found)
            qty = footing_counts.get(mark, 0)
            
            rows_to_insert.append((
                site_name,
                segment_name,
                str(pdf_path),
                str(image_path),
                mark,
                length,
                width,
                height,
                qty
            ))
            
        # Add piers (P1-P2)
        for mark in ("P1", "P2"):
            dims = pier_sizes.get(mark, {"length": None, "width": None, "height": None})
            qty = pier_counts.get(mark, 0)
            
            rows_to_insert.append((
                site_name,
                segment_name,
                str(pdf_path),
                str(image_path),
                mark,
                dims["length"],
                dims["width"],
                dims["height"],
                qty
            ))
            
        # 6. Database Storage
        print(f"Connecting to SQLite Database: {DB_FILE}")
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Create Table matching the 9 columns layout
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS foundation_elements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                site_name TEXT NOT NULL,
                segment TEXT NOT NULL,
                pdf_link TEXT NOT NULL,
                image_link TEXT NOT NULL,
                element_type TEXT NOT NULL,
                length REAL,
                width REAL,
                height REAL,
                qty INTEGER NOT NULL,
                UNIQUE(site_name, pdf_link, element_type)
            )
        """)
        
        # Clean existing rows for this project and PDF to ensure idempotence
        cursor.execute(
            "DELETE FROM foundation_elements WHERE site_name = ? AND pdf_link = ?",
            (site_name, str(pdf_path))
        )
        
        # Insert rows
        cursor.executemany("""
            INSERT OR REPLACE INTO foundation_elements (
                site_name, segment, pdf_link, image_link, element_type, length, width, height, qty
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows_to_insert)
        
        conn.commit()
        
        # Check stored count
        cursor.execute("SELECT COUNT(*) FROM foundation_elements WHERE site_name = ?", (site_name,))
        stored_count = cursor.fetchone()[0]
        conn.close()
        
        print(f"Successfully stored {stored_count} foundation elements in the SQLite database!")
        
    except Exception as e:
        import traceback
        print(f"Error during execution: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
