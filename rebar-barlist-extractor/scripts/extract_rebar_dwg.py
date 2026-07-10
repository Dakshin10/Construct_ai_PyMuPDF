"""
Generic rebar shop-drawing extractor (PyMuPDF).

Many CAD-exported rebar drawings render the "Bar List" schedule as scattered
text with no real table grid -- sometimes rotated 90 degrees, sometimes not,
and title-block LABELS are frequently vector graphics (not text), so only the
filled-in VALUES are extractable as text.

Strategy, per page:
  1. Try PyMuPDF's native table detector (page.find_tables()). Works when the
     drawing has real gridlines around the bar list.
  2. If that fails, fall back to a spatial reconstruction: find the text block
     containing bar-list header keywords (Mark/Qty/Size/Length/Type, in any
     wording), auto-detect whether the table reads normally (dir=(1,0)) or
     rotated (dir=(0,1)), cluster the remaining spans in that block into
     columns/rows by position (not hardcoded pixel offsets), and match shape
     dimensions to whatever row-labels the table itself defines (not a fixed
     A-R list).
  3. Title block: auto-pair any "Label : Value" text found in a single span.
     Everything else unclaimed is dumped into an "Unmapped Text" sheet with
     page + coordinates so nothing is silently lost -- use it to spot new
     fields on an unfamiliar template.

This is heuristic. Always spot-check output against the source drawing,
especially rows flagged in the "Issues" column and anything in "Unmapped Text".
"""

import re
import sys

import fitz
import pandas as pd

# ---------------------------------------------------------------- tunables

LABEL_MATCH_TOL = 20.0   # px: max distance to match a dimension value to a row-label
MIN_HEADER_HITS = 3      # how many of the 5 canonical headers must be found to accept a block as the bar list

MARK_WORDS = {"bar mark", "mark"}
QTY_WORDS = {"qty", "no.", "no", "number", "quantity", "pcs"}
SIZE_WORDS = {"size", "dia", "diameter", "bar size"}
LENGTH_WORDS = {"length", "total length", "cut length", "cutting length"}
TYPE_WORDS = {"type", "shape", "shape code", "shape type"}

LABEL_TOKEN_RE = re.compile(r"^['\"]?[A-Za-z0-9]{1,3}['\"]?$")
LABEL_VALUE_RE = re.compile(r"^\s*(.+?)\s*[:：]\s*(\S.*)$")


# ---------------------------------------------------------------- low level

def get_blocks(page):
    """Return text blocks as list of {bbox, spans:[{text,x0,y0,x1,y1,xc,yc,dir}]}."""
    blocks = []
    for b in page.get_text("dict")["blocks"]:
        if b["type"] != 0:
            continue
        spans = []
        for line in b["lines"]:
            txt = "".join(s["text"] for s in line["spans"]).strip()
            if not txt:
                continue
            x0, y0, x1, y1 = line["bbox"]
            spans.append({
                "text": txt, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "xc": (x0 + x1) / 2, "yc": (y0 + y1) / 2,
                "dir": line["dir"],
            })
        if spans:
            blocks.append({"bbox": b["bbox"], "spans": spans})
    return blocks


def cluster_1d(spans, key, tol):
    """Cluster spans into groups where consecutive key(span) values are within tol."""
    ordered = sorted(spans, key=key)
    clusters = []
    for s in ordered:
        if clusters and key(s) - key(clusters[-1][-1]) <= tol:
            clusters[-1].append(s)
        else:
            clusters.append([s])
    return clusters


def find_header_hits(block_spans):
    """Find which of the 5 canonical bar-list header roles appear in this block."""
    hits = {"mark": None, "qty": None, "size": None, "length": None, "type": None}
    for s in block_spans:
        t = s["text"].strip().lower()
        if hits["mark"] is None and t in MARK_WORDS:
            hits["mark"] = s
        if hits["qty"] is None and t in QTY_WORDS:
            hits["qty"] = s
        if hits["size"] is None and any(w in t for w in SIZE_WORDS):
            hits["size"] = s
        if hits["length"] is None and any(w in t for w in LENGTH_WORDS):
            hits["length"] = s
        if hits["type"] is None and re.search(r"\btype\b|\bshape\b", t):
            hits["type"] = s
    return hits


# ---------------------------------------------------------- native tables

def _keyword_category(text):
    t = text.strip().lower()
    if t in MARK_WORDS:
        return "mark"
    if t in QTY_WORDS:
        return "qty"
    if any(w in t for w in SIZE_WORDS):
        return "size"
    if any(w in t for w in LENGTH_WORDS):
        return "length"
    if re.search(r"\btype\b|\bshape\b", t):
        return "type"
    return None


def try_native_tables(page):
    """Use PyMuPDF's built-in (gridline/whitespace-inferred) table finder.

    Only trusted when at least MIN_HEADER_HITS canonical fields each land in
    their OWN column (not several keywords glued into one garbage cell) and
    no cell contains embedded newlines -- both are telltale signs of a
    misdetected table on a borderless CAD drawing, which happens often on
    rotated/dense technical drawings. Returns a DataFrame or None.
    """
    try:
        tabs = page.find_tables()
    except Exception:
        return None
    best = None
    for t in tabs.tables:
        try:
            df = t.to_pandas()
        except Exception:
            continue
        if df.astype(str).apply(lambda col: col.str.contains("\n")).any().any():
            continue
        categories = {cat for c in df.columns if (cat := _keyword_category(str(c)))}
        if len(categories) < MIN_HEADER_HITS:
            continue
        if best is None or len(df.columns) > len(best.columns):
            best = df
    return best


# ------------------------------------------------------- spatial fallback

def build_dimension_labels(block_spans, header_hits, order_key):
    """Row/column labels beyond the 5 fixed fields (e.g. A,B,C... shape dimensions),
    discovered from the drawing itself rather than assumed."""
    claimed_ids = {id(s) for s in header_hits.values() if s}
    anchor_positions = [order_key(s) for s in header_hits.values() if s]
    if not anchor_positions:
        return []
    after = max(anchor_positions)
    labels = []
    for s in block_spans:
        if id(s) in claimed_ids:
            continue
        if order_key(s) <= after:
            continue
        if LABEL_TOKEN_RE.match(s["text"]):
            labels.append((s["text"].strip("'\""), order_key(s)))
    labels.sort(key=lambda t: t[1])
    return labels


def parse_record(col_spans, header_hits, labels, order_key):
    """Parse one bar-mark's worth of spans (already sorted along the reading axis)
    into a record dict, using header positions (not hardcoded pixel bands) as anchors."""
    col_spans = sorted(col_spans, key=order_key)
    rec = {"BarMark": None, "Qty": None, "Size": None, "TotalLength": None, "Type": None, "Issues": ""}
    for lbl, _ in labels:
        rec[lbl] = None

    idx = 0
    mark_hit, qty_hit = header_hits.get("mark"), header_hits.get("qty")
    if mark_hit and qty_hit and col_spans:
        boundary = (order_key(mark_hit) + order_key(qty_hit)) / 2
        if order_key(col_spans[0]) < boundary:
            rec["BarMark"] = col_spans[0]["text"]
            idx = 1

    # CRSI/CSA bar-list convention: a shape Type code is present iff the bar has a Mark
    # (straight bars carry no mark and no type). Use that instead of a pixel cutoff,
    # which is fragile when Type values sit only ~1px inside the dimension-label zone.
    fixed_count = 3 if rec["BarMark"] else 2
    first_label_pos = labels[0][1] if labels else float("inf")
    pre_dim = col_spans[idx:idx + fixed_count]
    if pre_dim and order_key(pre_dim[-1]) > first_label_pos - LABEL_MATCH_TOL:
        rec["Issues"] += "Type/Length field sits close to the dimension zone; verify manually; "

    if len(pre_dim) >= 1:
        parts = pre_dim[0]["text"].split(None, 1)
        rec["Qty"] = parts[0] if parts else None
        rec["Size"] = parts[1] if len(parts) > 1 else None
    if len(pre_dim) >= 2:
        rec["TotalLength"] = pre_dim[1]["text"]
    if len(pre_dim) >= 3:
        rec["Type"] = pre_dim[2]["text"]

    idx += len(pre_dim)
    for s in col_spans[idx:]:
        if not labels:
            rec["Issues"] += f"unmatched value '{s['text']}' (no dimension labels found); "
            continue
        nearest = min(labels, key=lambda t: abs(t[1] - order_key(s)))
        if abs(nearest[1] - order_key(s)) > LABEL_MATCH_TOL * 3:
            rec["Issues"] += f"'{s['text']}' too far from any dimension label; "
        rec[nearest[0]] = s["text"]

    return rec


def extract_barlist_spatial(blocks, page_no):
    for block in blocks:
        header_hits = find_header_hits(block["spans"])
        if sum(v is not None for v in header_hits.values()) < MIN_HEADER_HITS:
            continue

        # orientation: which axis dominates the header spans' writing direction
        dirs = [s["dir"] for s in block["spans"] if s in header_hits.values()]
        vertical = sum(abs(d[1]) for d in dirs) >= sum(abs(d[0]) for d in dirs)

        if vertical:
            group_key = lambda s: s["xc"]
            order_key = lambda s: s["y0"]
        else:
            group_key = lambda s: s["yc"]
            order_key = lambda s: s["x0"]

        # the header/label column|row = the cluster that actually contains the header-hit spans
        header_hit_ids = {id(v) for v in header_hits.values() if v}
        groups = cluster_1d(block["spans"], group_key, tol=LABEL_MATCH_TOL / 4)
        header_group = next((g for g in groups if any(id(s) in header_hit_ids for s in g)), None)
        if header_group is None:
            continue
        data_groups = [g for g in groups if g is not header_group]

        if not data_groups:
            continue

        labels = build_dimension_labels(header_group, header_hits, order_key)

        records = []
        for g in data_groups:
            rec = parse_record(g, header_hits, labels, order_key)
            rec["Page"] = page_no
            records.append(rec)

        label_cols = [l for l, _ in labels]
        df = pd.DataFrame(records, columns=["Page", "BarMark", "Qty", "Size", "TotalLength", "Type"]
                           + label_cols + ["Issues"])
        consumed_ids = {id(s) for g in data_groups for s in g} | {id(v) for v in header_hits.values() if v} \
            | {id(s) for s in block["spans"] if LABEL_TOKEN_RE.match(s["text"])}
        return df, consumed_ids
    return None, set()


# ------------------------------------------------------------- title block

def extract_title_block(blocks, consumed_ids):
    pairs = []
    unmapped = []
    for block in blocks:
        for s in block["spans"]:
            if id(s) in consumed_ids:
                continue
            m = LABEL_VALUE_RE.match(s["text"])
            if m:
                pairs.append({"Field": m.group(1), "Value": m.group(2), "x": round(s["x0"], 1), "y": round(s["y0"], 1)})
            else:
                unmapped.append({"Text": s["text"], "x": round(s["x0"], 1), "y": round(s["y0"], 1)})
    return pairs, unmapped


# --------------------------------------------------------------------- main

def main(pdf_path, out_path):
    doc = fitz.open(pdf_path)
    all_barlist = []
    all_titleblock = []
    all_unmapped = []

    for page_no, page in enumerate(doc, start=1):
        blocks = get_blocks(page)

        df = try_native_tables(page)
        strategy = "native-table"
        consumed_ids = set()
        if df is None:
            df, consumed_ids = extract_barlist_spatial(blocks, page_no)
            strategy = "spatial-reconstruction" if df is not None else "none"
        # Note: when native-table succeeds, consumed_ids stays empty (it isn't tied to
        # our span objects), so those cells will also appear in "Unmapped Text" -- harmless
        # duplication, just don't be surprised to see bar-list values there in that case.
        if df is not None and "Page" not in df.columns:
            df.insert(0, "Page", page_no)
        print(f"Page {page_no}: bar-list strategy = {strategy}"
              f"{f', {len(df)} rows' if df is not None else ''}")
        if df is not None:
            all_barlist.append(df)

        pairs, unmapped = extract_title_block(blocks, consumed_ids)
        for p in pairs:
            p["Page"] = page_no
        for u in unmapped:
            u["Page"] = page_no
        all_titleblock.extend(pairs)
        all_unmapped.extend(unmapped)

    bar_df = pd.concat(all_barlist, ignore_index=True) if all_barlist else pd.DataFrame()
    title_df = pd.DataFrame(all_titleblock, columns=["Page", "Field", "Value", "x", "y"])
    unmapped_df = pd.DataFrame(all_unmapped, columns=["Page", "Text", "x", "y"])

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        bar_df.to_excel(writer, sheet_name="Bar List", index=False)
        title_df.to_excel(writer, sheet_name="Title Block (auto-paired)", index=False)
        unmapped_df.to_excel(writer, sheet_name="Unmapped Text", index=False)

    n_issues = int((bar_df["Issues"] != "").sum()) if "Issues" in bar_df.columns and not bar_df.empty else 0
    print(f"\nBar list: {len(bar_df)} rows total, {n_issues} flagged with issues for manual review")
    print(f"Title block: {len(title_df)} auto-paired fields, {len(unmapped_df)} unmapped text spans")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else None
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    if not pdf_path:
        raise SystemExit("Usage: python extract_rebar_dwg.py <input.pdf> [output.xlsx]")
    if not out_path:
        out_path = pdf_path.rsplit(".", 1)[0] + "_extracted.xlsx"
    main(pdf_path, out_path)
