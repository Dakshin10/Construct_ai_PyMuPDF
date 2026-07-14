# Known Limitations & Source-Document Constraints

These are **not pipeline bugs** — they are inherent limits of what the source
drawing sets provide, plus a documented failure-mode class the extractor now
guards against. Recorded here so they aren't re-investigated as new "bugs" in a
future session.

---

## 1. Per-element length / stock-length is not reliably extractable

**Status: source-document limitation, not a pipeline gap. Do not treat as fixable.**

MTO (material take-off) and rebar-weight calculation require, per element:
`bar length × count × unit weight`. The Building-C drawing set — and similar
SOSCIA-style structural sets — **do not provide bar cut-lengths or member
lengths in a structured, mark-mappable way.**

What the drawings *do* give and the pipeline *does* capture:
- bar **size** (e.g. 15M, 20M), **count** (e.g. 16), and **spacing** (@ c/c)
- member **cross-sections** (e.g. HSS 203×203×7.9, "350mm WIDE × 300mm Dp.")

What is **absent or unusable** for length:
- No `STOCK`-length values anywhere in the text (0 occurrences).
- A "LENGTH" schedule column header exists (foundation sheet) but its values
  live inside the geometrically-scrambled schedule matrix and cannot be
  reliably mapped back to a specific mark.
- Only **sporadic, heterogeneous** length data exists — dowel cut-lengths
  ("15M DOWELS × 915mm Lg."), anchor-bolt lengths ("19mm DIA. × 610mm Long") —
  none tied to the pier/footing/column marks.

**Consequence:** weight/MTO queries cannot be answered from these drawings. The
pipeline correctly reports length/weight as *not available* rather than
guessing. Supplying length would require a different input (a bar bending
schedule / BBS, or an estimator-provided cut-list), not a parser change.

---

## 2. Steel↔concrete mis-anchoring of schedule callouts (guarded)

**Status: root-caused and guarded. Documented so the same failure mode is
recognised on new drawing sets.**

### The failure mode
On these sheets, **steel-member schedules and concrete-reinforcement schedules
sit close together on the same drawing**. Extraction assigns each text callout
to its spatially-nearest schedule heading. When a concrete-rebar callout sits
nearer a steel heading (or a steel section/hardware token reuses a mark like
`C1`), the rebar data gets **mis-anchored to the wrong element type** — a steel
member silently absorbs concrete rebar/tie data, or a non-element (a bolt grade,
a drawing cross-reference) is ingested as a structural element.

### Confirmed instances (both fixed)
| Symptom | Reality | Fix |
|---|---|---|
| `Column C1,C2` reported `16×20M rebar, 10M ties` | Steel **HSS 203×203** columns (no rebar). The rebar was the **pier** schedule's `16-20M VERT.` mis-anchored to a column heading. | Columns capture their HSS section + `material=steel`; never accept concrete rebar/ties; the pier keeps its own rebar. |
| `Beam L76 / L15 / A325 / beam-12` | `L76`/`L15` = steel **angle**/deck-fastener callouts; `A325` = ASTM **bolt grade**; `beam-12` = general-notes prose fragment. **No concrete beams exist** (structure is steel-framed). | All rejected from `element_summary`. |
| `Slab S3 / MW18` | `S3` = "SECTION ON DWG. **No. S3**" drawing cross-reference; `MW18` = "152×152-**MW18.7**" welded-wire-mesh spec. | Both rejected; the genuine slab-on-grade is kept as concrete. |

### The reusable guard
`extract_general_pdf.classify_member(label, context)` classifies every candidate
element by its mark + nearby text into **`reject` / `steel` / `concrete`**:
- **reject** — ASTM bolt grades (`A325`/`A307`), drawing cross-references
  (`DWG. No. …`, `SECTION ON DWG`), and angle/mesh/fastener marks (`L##`, `MW##`).
- **steel** — HSS / W-shape / channel members (or `Wall Thk.`, `O.W.S.J.`, deck,
  gauge context): `material=steel`, rebar/tie fields cleared (inapplicable, not
  missing).
- **concrete** — genuine concrete element; rebar/tie data kept.

It is applied at both extraction sources (the schedule parser already special-
cases HSS columns; the plan-view element parser routes every candidate through
`classify_member`) plus an implausible-dimension guard (no real member has a
<50 mm plan dimension — catches notes-prose garbage like a "4×37" beam).

### If a future drawing set regresses
Look first for a **steel schedule adjacent to a rebar schedule** on the same
sheet. Extend `classify_member`'s pattern tables (`_STEEL_MARK`, `_STEEL_CTX`,
`_BOLT_GRADE`, `_DWG_REF`) rather than adding element-type-specific patches.

### Concrete grade beam — now captured (was a documented gap)
The concrete grade beam is a free-text foundation-plan callout ("350mm WIDE ×
300mm Dp. CONT. CONC. (FORMED) GRADE BEAM REINF. w/ 2-15M CONT. BARS AT TOP OF
WALL"), not a schedule row, so the generic keyword parser mis-grouped its rebar
onto a neighbouring footing. It is now captured by a **targeted extractor**
(`extract_general_pdf.extract_grade_beams`): it keys on the "GRADE BEAM" callout,
takes the size from "N mm WIDE x N mm Dp" and the rebar **only** from the
grade-beam-specific phrase "N-NNM CONT. BARS AT TOP OF WALL" (so it doesn't
absorb the adjacent trench-footing's "…TOP & BOTTOM" bars), and confirms concrete
via `classify_member`. Building-C yields two variants — **350×300** and
**350×400**, both **2-15M** continuous.

If a future set uses different grade-beam phrasing, extend `_GB_SIZE` / `_GB_REBAR`
rather than widening the scan window (a wide window re-absorbs neighbouring
footing rebar).

---

## 3. "Stirrups" terminology — this drawing set uses "TIES" instead

**Status: source-document terminology, not a pipeline gap.**

The Building-C set never issues a sized stirrup callout ("STIRRUPS ##M"). The
bare word "stirrups" appears exactly twice in the whole corpus, both boilerplate
CSA-spec sentences with no associated size (e.g. "STIRRUPS AND TIES TO BE
DEFORMED BARS TO CONFORM TO CSA G30.18"). All actual transverse/hoop
reinforcement on these drawings is called out as **TIES** instead (e.g. "10M
TIES @ 300mm c/c.", captured by the working `ties_type` / `ties_spacing`
fields).

**Consequence:** `element_summary` chunks surface the extracted `ties_type` /
`ties_spacing` value under `stirrups` too (see `ingest_to_chroma.py`,
`build_element_summary_chunks`), with an explicit note in the chunk text
("this drawing set records transverse reinforcement as \"ties\", not a sized
\"stirrups\" callout") and a `stirrups_source: ties` metadata flag — so this
reads as *this project's terminology*, not fabricated or double-counted data.
The original `ties_type`/`ties_spacing` fields are left untouched.

**If a future drawing set genuinely uses distinct, sized "STIRRUPS ##M"
callouts** (different from ties), do not assume this derivation still applies —
check whether `stirrups_source` is being set from a real per-project regex
match vs. this ties-derived fallback, and extend `ATTR_PATTERNS["stirrups"]`
for that project rather than relying on the ties surfacing.
