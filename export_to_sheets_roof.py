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
DB_FILE = ROOT / "output" / "roof_data.db"
CLIENT_SECRET_FILE = Path("D:/MinoriLabs-intern/minori-hrms-mcp/credentials/client_secret.json")
TOKEN_FILE = ROOT / "output" / "google_token.json"

# Full write scopes for Sheets and Drive
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
    parser = argparse.ArgumentParser(description="Export roof column database to Google Sheets")
    parser.add_argument("--url", help="Google Sheets URL to write to (creates new sheet if omitted)")
    args = parser.parse_args()

    if not DB_FILE.exists():
        print(f"Error: Database file not found at '{DB_FILE}'. Run python run_db_extraction_roof.py first.")
        return

    # 1. Fetch data from SQLite
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='roof_elements'")
        if not cursor.fetchone():
            print("Error: Table 'roof_elements' does not exist in database.")
            conn.close()
            return
            
        cursor.execute("""
            SELECT site_name, segment, pdf_link, image_link, type, qty
            FROM roof_elements
            ORDER BY type
        """)
        rows = cursor.fetchall()
        
        if not rows:
            print("No column elements stored in the database to export.")
            conn.close()
            return
            
        site_name = rows[0][0]
    except Exception as e:
        print(f"Database error: {e}")
        conn.close()
        return
    finally:
        conn.close()

    # Define headers without 'Col X:' prefix
    headers = [
        "Site Name",
        "Segment",
        "PDF Link",
        "Image Link",
        "Type",
        "QTY"
    ]

    # Convert rows to lists
    values = []
    for r in rows:
        values.append(list(r))

    # 2. Authorize Google Client
    try:
        client = get_google_client()
    except Exception as e:
        print(f"Google Authorization failed: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Open or create Spreadsheet
    try:
        if args.url:
            print(f"Opening existing Google Sheet by URL: {args.url}")
            sh = client.open_by_url(args.url)
        else:
            title = f"Roof Elements - {site_name}"
            print(f"Creating new Google Sheet: '{title}'...")
            sh = client.create(title)
            
        worksheet = sh.get_worksheet(0)
        worksheet.clear()
        
        # Write headers + values
        print("Writing data to worksheet...")
        worksheet.update([headers] + values)
        
        print("\n" + "="*80)
        print("                       GOOGLE SHEETS EXPORT SUCCESSFUL")
        print("="*80)
        print(f"Spreadsheet Title: {sh.title}")
        print(f"Link to Sheet:     {sh.url}")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"Failed to write to Google Sheets: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
