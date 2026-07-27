import os
import sys
import sqlite3
import argparse
from pathlib import Path
import gspread
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow

ROOT = Path(__file__).parent.resolve()
DB_FOUNDATION = ROOT / "output" / "foundation_data.db"
DB_ROOF = ROOT / "output" / "roof_data.db"

CLIENT_SECRET_FILE = Path("D:/MinoriLabs-intern/minori-hrms-mcp/credentials/client_secret.json")
TOKEN_FILE = ROOT / "output" / "google_token.json"

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

def get_google_client():
    """Authorize with gspread using InstalledAppFlow for browser login."""
    if not CLIENT_SECRET_FILE.exists():
        raise FileNotFoundError(
            f"Google client secret file not found at {CLIENT_SECRET_FILE}. "
            "Ensure the minori-hrms-mcp project has a valid client_secret.json."
        )

    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("Refreshing expired Google API credentials...")
            try:
                creds.refresh(Request())
            except Exception:
                creds = None
                
        if not creds:
            print("\nAuthorization required. Opening browser window for Google Login...")
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CLIENT_SECRET_FILE),
                SCOPES
            )
            creds = flow.run_local_server(port=0)
            
        os.makedirs(TOKEN_FILE.parent, exist_ok=True)
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())
            
    return gspread.authorize(creds)

def main():
    parser = argparse.ArgumentParser(description="Export both foundation and roof databases combined into a single tab in the Google Sheet")
    parser.add_argument("--url", required=True, help="Google Sheets URL to write to")
    args = parser.parse_args()

    # 1. Fetch Foundation Data (9 columns)
    foundation_rows = []
    if DB_FOUNDATION.exists():
        conn = sqlite3.connect(DB_FOUNDATION)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='foundation_elements'")
            if cursor.fetchone():
                cursor.execute("""
                    SELECT site_name, segment, pdf_link, image_link, element_type, length, width, height, qty
                    FROM foundation_elements
                    ORDER BY 
                        CASE WHEN element_type LIKE 'F%' THEN 1 ELSE 2 END,
                        element_type
                """)
                foundation_rows = cursor.fetchall()
        except Exception as e:
            print(f"Error fetching foundation database: {e}")
        finally:
            conn.close()

    # 2. Fetch Roof Data (9 columns: site_name, segment, pdf_link, image_link, type, length, width, height, qty)
    roof_rows = []
    if DB_ROOF.exists():
        conn = sqlite3.connect(DB_ROOF)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='roof_elements'")
            if cursor.fetchone():
                cursor.execute("""
                    SELECT site_name, segment, pdf_link, image_link, type, length, width, height, qty
                    FROM roof_elements
                    ORDER BY type
                """)
                roof_rows = cursor.fetchall()
        except Exception as e:
            print(f"Error fetching roof database: {e}")
        finally:
            conn.close()

    if not foundation_rows and not roof_rows:
        print("No elements found in either foundation or roof database to export.")
        return

    # Unified 9-column headers
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

    # Combine all records
    combined_rows = list(foundation_rows) + list(roof_rows)
    
    # Map None values to empty strings for Google Sheets
    values = [[("" if val is None else val) for val in r] for r in combined_rows]

    # 3. Authorize Google Client
    try:
        client = get_google_client()
    except Exception as e:
        print(f"Google Authorization failed: {e}", file=sys.stderr)
        sys.exit(1)

    # 4. Open Spreadsheet and write to the first sheet
    try:
        print(f"Opening Google Sheet: {args.url}")
        sh = client.open_by_url(args.url)
        
        # Get the first sheet
        worksheet = sh.get_worksheet(0)
        worksheet.clear()
        
        print("Writing combined data to worksheet...")
        worksheet.update([headers] + values)

        print("\n" + "="*80)
        print("                   COMBINED FLAT GOOGLE SHEETS EXPORT SUCCESSFUL")
        print("="*80)
        print(f"Spreadsheet Title: {sh.title}")
        print(f"Link to Sheet:     {sh.url}")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"Failed to write to Google Sheets: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
