import os
import re
import sys
import sqlite3
from pathlib import Path
import fitz

# Insert src to python path to use the general extractor helper
sys.path.insert(0, str(Path(__file__).parent / "src"))
from extract_general_pdf import process_pdf

# --- Configuration ---
ROOT = Path(__file__).parent.resolve()
DB_FILE = ROOT / "output" / "roof_data.db"
IMAGES_DIR = ROOT / "output" / "images"
PDF_PATH = Path(r"C:\Users\daksh\OneDrive - MinoriLabs\Construction_ai_assitant\rooflayout.pdf")
os.makedirs(IMAGES_DIR, exist_ok=True)

# 1. Render PDF page 1 to a high-res PNG image
def render_pdf_page(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    zoom = 3  # High-resolution rendering
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    
    png_path = IMAGES_DIR / "rooflayout_page1.png"
    pix.save(str(png_path))
    doc.close()
    return png_path.resolve()

# 2. Extract site name from drawing text
def extract_site_name(pdf_path):
    doc = fitz.open(pdf_path)
    text = doc[0].get_text("text").upper()
    doc.close()
    
    if "BUILDING \"C\"" in text or "BUILDING C" in text or "BUILDING-C" in text:
        return "Building C"
    return "Building C"  # Default site name

# 2b. Extract drawing description dynamically (used as Segment)
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

# 3. Parse column sizes from schedule elements
def parse_column_dimensions(schedule_elements):
    # Default fallbacks from drawing schedule
    dims = {
        "C1": {"length": 203.0, "width": 203.0, "height": 7.9},
        "C2": {"length": 203.0, "width": 203.0, "height": 9.5}
    }
    
    for elem in schedule_elements:
        if elem.get("element_type") == "column" and elem.get("raw_lines"):
            raw_lines = elem["raw_lines"]
            for i, line in enumerate(raw_lines):
                # Search for HSS dimensions (e.g. 'HSS 203mm x 203mm x 7.9mm')
                m = re.search(r"HSS\s*(\d+)\s*mm\s*x\s*(\d+)\s*mm\s*x\s*([\d\.]+)\s*mm", line, re.I)
                if m:
                    length = float(m.group(1))
                    width = float(m.group(2))
                    height = float(m.group(3))
                    
                    # Look for mark (C1/C2) in adjacent lines
                    assigned = False
                    for offset in [1, -1]:
                        if 0 <= i + offset < len(raw_lines):
                            val = raw_lines[i + offset].strip().upper()
                            if val in ["C1", "C2"]:
                                dims[val] = {"length": length, "width": width, "height": height}
                                assigned = True
                                break
                    if not assigned:
                        # Fallback to window search if not directly adjacent
                        window = raw_lines[max(0, i-2):min(len(raw_lines), i+3)]
                        for win_line in window:
                            win_line_clean = win_line.strip().upper()
                            if win_line_clean in ["C1", "C2"]:
                                dims[win_line_clean] = {"length": length, "width": width, "height": height}
                            
    return dims

def main():
    try:
        if not PDF_PATH.exists():
            print(f"Error: PDF file does not exist at '{PDF_PATH}'")
            sys.exit(1)
            
        print(f"Target PDF: {PDF_PATH}")
        site_name = extract_site_name(PDF_PATH)
        segment_name = extract_drawing_description(PDF_PATH)
        print(f"Extracted Site Name: {site_name}")
        print(f"Extracted Segment Name: {segment_name}")
        
        print("Rendering PDF page to image store...")
        image_path = render_pdf_page(PDF_PATH)
        print(f"Rendered image path: {image_path}")
        
        # Use full core extraction pipeline (process_pdf)
        print("Running core extraction pipeline...")
        extracted_data = process_pdf(str(PDF_PATH))
        
        plan_instances = extracted_data.get("plan_instances", {})
        column_counts = plan_instances.get("column", {})
        
        # Extract C1 and C2 quantities
        c1_count = column_counts.get("C1", 0)
        c2_count = column_counts.get("C2", 0)
        print(f"Extracted: C1 Qty = {c1_count}, C2 Qty = {c2_count}")
        
        # Parse column dimensions
        schedule_elements = extracted_data.get("schedule_elements", [])
        col_dims = parse_column_dimensions(schedule_elements)
        print(f"Parsed Column Dimensions: {col_dims}")
        
        # SQLite Database Storage
        print(f"Connecting to SQLite Database: {DB_FILE}")
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Create Table matching the requested format:
        # site_name, segment, pdf_link, image_link, type, length, width, height, qty
        cursor.execute("DROP TABLE IF EXISTS roof_elements")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS roof_elements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                site_name TEXT NOT NULL,
                segment TEXT NOT NULL,
                pdf_link TEXT NOT NULL,
                image_link TEXT NOT NULL,
                type TEXT NOT NULL,
                length REAL,
                width REAL,
                height REAL,
                qty INTEGER NOT NULL,
                UNIQUE(site_name, pdf_link, type)
            )
        """)
        
        # Prepare rows to insert
        rows_to_insert = [
            (
                site_name, segment_name, str(PDF_PATH), str(image_path), "C1", 
                col_dims["C1"]["length"], col_dims["C1"]["width"], col_dims["C1"]["height"], c1_count
            ),
            (
                site_name, segment_name, str(PDF_PATH), str(image_path), "C2", 
                col_dims["C2"]["length"], col_dims["C2"]["width"], col_dims["C2"]["height"], c2_count
            )
        ]
        
        # Insert rows
        cursor.executemany("""
            INSERT OR REPLACE INTO roof_elements (
                site_name, segment, pdf_link, image_link, type, length, width, height, qty
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows_to_insert)
        
        conn.commit()
        conn.close()
        
        print("Successfully stored roof column elements in the SQLite database!")
        
    except Exception as e:
        import traceback
        print(f"Error during execution: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
