import os
import sqlite3
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
DB_FILE = ROOT / "output" / "roof_data.db"

def main():
    if not DB_FILE.exists():
        print(f"Error: Database file not found at '{DB_FILE}'. Run python run_db_extraction_roof.py first.")
        return

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='roof_elements'")
        if not cursor.fetchone():
            print("Error: Table 'roof_elements' does not exist in database.")
            conn.close()
            return
            
        cursor.execute("""
            SELECT site_name, segment, pdf_link, image_link, type, length, width, height, qty
            FROM roof_elements
            ORDER BY type
        """)
        rows = cursor.fetchall()
        
        if not rows:
            print("No column elements stored in the database yet.")
            conn.close()
            return
            
        # Define column headers
        headers = [
            "Site Name",
            "Segment",
            "PDF Link",
            "Image Link",
            "Type",
            "Length (mm)",
            "Width (mm)",
            "Height (mm)",
            "QTY"
        ]
        
        # Calculate max widths for nice formatting
        col_widths = [len(h) for h in headers]
        for row in rows:
            for idx, val in enumerate(row):
                val_str = str(val)
                # Shorten long absolute paths for better terminal readability
                if idx in (2, 3) and len(val_str) > 35:
                    val_str = "..." + val_str[-32:]
                col_widths[idx] = max(col_widths[idx], len(val_str))
                
        # Separator line
        sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
        
        print("\n" + "=" * len(sep))
        print("                 ROOF COLUMNS DATABASE CONTENTS")
        print("=" * len(sep))
        
        # Print headers
        print(sep)
        header_row = "|" + "|".join(f" {headers[i]:<{col_widths[i]}} " for i in range(len(headers))) + "|"
        print(header_row)
        print(sep)
        
        # Print data rows
        for row in rows:
            formatted_vals = []
            for idx, val in enumerate(row):
                val_str = str(val)
                if idx in (5, 6, 7) and val is not None:
                    val_str = f"{float(val):.1f}" if val % 1 != 0 else f"{float(val):.0f}"
                if idx in (2, 3) and len(val_str) > 35:
                    val_str = "..." + val_str[-32:]
                formatted_vals.append(val_str)
                
            data_row = "|" + "|".join(f" {formatted_vals[i]:<{col_widths[i]}} " for i in range(len(headers))) + "|"
            print(data_row)
            
        print(sep)
        print(f"Total elements retrieved: {len(rows)}\n")
        
    except Exception as e:
        print(f"Error querying database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
