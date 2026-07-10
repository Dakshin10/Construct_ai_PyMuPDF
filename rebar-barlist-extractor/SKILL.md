---
name: rebar-barlist-extractor
description: Extracts the Bar List (rebar schedule) and title-block metadata from CAD-exported rebar/reinforcement shop drawing PDFs into an Excel workbook. Use this whenever the user wants to pull data out of a rebar shop drawing, reinforcement detail drawing, or bar bending schedule PDF -- especially CRSI/CSA-style drawings (e.g. from CADS-USA, or similar rebar detailing software) where the bar list shows Mark/Qty/Size/Length/Type/shape-dimension columns. Trigger on mentions of "bar list", "bar schedule", "rebar drawing", "reinforcement drawing", "bar bending schedule", "CRSI", "CSA G30.18", or a request to tabulate/extract data from a structural/rebar PDF -- even if the user just says "pull the data out of this drawing" while attaching a rebar shop drawing PDF. Do NOT use for general PDF text extraction, architectural drawings without a bar list, or PDFs that aren't rebar/reinforcement shop drawings.
---

# Rebar Bar List Extractor

## What this is for

Rebar shop drawings (reinforcement/bar-bending-schedule PDFs) usually carry their
real data -- the Bar List -- as a scattered grid of independent text objects with
no underlying table structure. Many are exported from CAD tools (e.g. CADS-USA)
with the whole schedule rotated 90 degrees, and the title-block *labels*
("Revised:", "Drawn by:") are often vector graphics rather than text, so only
the filled-in *values* are extractable. A naive `pdf.get_text()` dump turns
this into an unreadable wall of numbers with no idea which value belongs to
which bar mark.

This skill wraps a script (`scripts/extract_rebar_dwg.py`) that reconstructs the
table from the PDF's text-layer coordinates and writes a clean Excel workbook.
It was built and validated against a real CADS-USA drawing, cross-checked
row-by-row against that drawing's own bar-list report, with a 100% field match.

## When to use it

Reach for this skill whenever the user hands you a rebar/reinforcement shop
drawing PDF and wants the bar list (or the whole drawing's data) turned into a
spreadsheet, table, or otherwise "extracted." This includes requests phrased
around the drawing itself ("tabulate this wall drawing", "get me the rebar
quantities from this PDF") even if the user doesn't use the words "bar list."

It is **not** a general PDF-to-Excel tool -- it's specifically shaped around
the CRSI/CSA bar-list convention (Mark, Qty, Size, Total Length, Type, then
shape dimensions lettered A, B, C...). If the PDF isn't a rebar shop drawing,
use ordinary PDF extraction instead.

## How to run it

```
python scripts/extract_rebar_dwg.py "<input.pdf>" ["<output.xlsx>"]
```

If the output path is omitted, it defaults to `<input>_extracted.xlsx` next to
the source PDF. Dependencies: `pymupdf`, `pandas`, `openpyxl` -- install with
`pip install pymupdf pandas openpyxl` if the run fails with `ModuleNotFoundError`.

The script prints, per page, which strategy it used (`native-table` or
`spatial-reconstruction`) and a row/issue count. Read that output -- it tells
you how much to trust the result before you even open the spreadsheet.

## Strategy the script uses (so you know what to check)

1. **Try PyMuPDF's native table detector first** (`page.find_tables()`). This
   only wins when the drawing has real gridlines, and it's validated strictly
   (every one of Mark/Qty/Size/Length/Type must land in its own column, no
   cell may contain an embedded newline) because on borderless CAD drawings
   this detector can produce a plausible-looking but subtly wrong table --
   e.g. silently dropping a row or merging two columns.
2. **Fall back to spatial reconstruction.** It finds the text block containing
   bar-list header keywords (in any common wording), auto-detects whether the
   table reads normally or is rotated 90 degrees (from PyMuPDF's line
   direction vector), clusters the remaining text into one group per bar mark
   by position, and matches shape-dimension values to whatever row-labels
   *this* drawing defines -- not a hardcoded A-R list, since some shapes only
   use a few letters and some go further.
3. **Title block**: any text that reads as `"Label : Value"` in a single span
   gets auto-paired. Everything else unclaimed (which, on many CAD exports, is
   most of the title block -- since the labels are graphics, not text) goes
   into an "Unmapped Text" sheet with page + coordinates, rather than being
   silently dropped or guessed at.
4. **Ambiguous rows get flagged**, not silently resolved. If a Type value sits
   right at the boundary of the dimension zone, the row's `Issues` column says
   so instead of guessing.

## Output

An Excel workbook with three sheets:

- **Bar List** -- one row per bar mark: `Page, BarMark, Qty, Size,
  TotalLength, Type`, then one column per shape-dimension label the drawing
  actually uses, then `Issues`.
- **Title Block (auto-paired)** -- `Page, Field, Value, x, y` for any
  `Label : Value` text found directly.
- **Unmapped Text** -- `Page, Text, x, y` for everything else on the page not
  consumed by the bar list. This is where most title-block values (project
  name, drawing number, date, drafter, scale...) usually end up, since their
  labels are graphics rather than text on many CAD exports.

## Verifying the output -- do this every time

Don't just hand back the spreadsheet. Before presenting results:

1. **Check the console strategy line** for each page -- if it says
   `spatial-reconstruction`, that's the expected/well-tested path for
   CAD-rotated drawings; if it says `native-table`, double check a row or two
   against the drawing since that path is newer and less exercised.
2. **Open the Bar List sheet and count rows** against the number of distinct
   bar marks visible in the drawing's own bar list region. A mismatch usually
   means the header block wasn't found cleanly, or there's more than one
   bar-list table on the page (see limitations below).
3. **Check the `Issues` column** for any flagged rows and manually confirm
   those specific values against the PDF.
4. **Skim the Unmapped Text sheet** for anything that looks like it should
   have been part of the Bar List (e.g. a stray Mark or Qty value) -- that
   would indicate the spatial clustering missed a column.
5. **If the user has a reference bar-list report** (many rebar detailing tools,
   like CADS-USA, can print one separately from the drawing), cross-check
   your output against it row by row before treating the extraction as final
   -- this is exactly how the script itself was validated.

## Known limitations

- Built specifically around the CRSI/CSA bar-list convention (Mark, Qty, Size,
  Length, Type, then lettered shape dimensions). A drawing using a genuinely
  different schedule layout (different field order, no shape-dimension
  columns, etc.) will need new parsing logic, not just a tweak.
- Only extracts the *first* bar-list-like block found per page. A sheet with
  two separate bar-list tables (e.g. one per wall segment) needs a small
  extension to loop over all matching blocks instead of returning the first.
- The "Type code present iff Mark present" rule is a domain assumption from
  the CRSI/CSA convention, not a geometric fact -- if a future drawing breaks
  this convention, affected rows should still surface via the `Issues` column
  since the field boundary would land somewhere odd.
- Multi-page PDFs are supported (each page is processed independently and
  results are concatenated with a `Page` column), but the script has only
  been exercised against a single-page drawing so far.
