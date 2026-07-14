"""
Unit + regression tests for the steel/concrete mis-anchoring guard (C2).

`classify_member()` and the grade-beam extractor have each caused and then fixed
a real data-integrity bug this project (columns absorbing pier rebar; the slab
steel-misclassification regression; grade-beam footing-rebar bleed). These tests
lock those fixes in.

Run:  python -m pytest tests/ -q     (or:  python tests/test_classification.py)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from extract_general_pdf import classify_member, extract_grade_beams, extract_elements


# ── classify_member() ─────────────────────────────────────────────────────────

def test_hss_steel_section_is_steel_not_rebar():
    # HSS column heading → steel member; rebar must not be attached to it
    assert classify_member("C1", "HSS 203mm x 203mm x 7.9mm Wall Thk.") == "steel"
    assert classify_member(None, "HSS 305x152x9.5 (L.S.V.)") == "steel"


def test_bolt_grades_rejected_entirely():
    assert classify_member("A325", "connections made with 20mm DIA. A325 bolts") == "reject"
    assert classify_member("A307", "A307 anchor bolts") == "reject"
    # bolt grade in context, no rebar signal, unmarked → still reject
    assert classify_member(None, "All connections shall be made with A325 bolts.") == "reject"


def test_angle_sections_rejected_entirely():
    assert classify_member("L76", "PROVIDE L76 x 76 x 6.4 WELDED ON TOP OF JOIST") == "reject"
    assert classify_member("L15", "DECK TO STEEL BEAM = X-ENP-19 L15") == "reject"


def test_deck_fastener_context_is_not_concrete():
    # A deck-fastener callout must never be accepted as a concrete element
    assert classify_member(None, "DECK TO STEEL BEAM = X-ENP-19; SIDE LAP @ 300 c/c") != "concrete"


def test_genuine_concrete_rebar_accepted():
    assert classify_member("F1", "REINF. w/ 3-15M CONT. BARS TOP & BOTTOM") == "concrete"
    assert classify_member("P1", "16-20M BARS with 10M TIES @ 300mm c/c") == "concrete"


def test_welded_wire_mesh_is_concrete_not_steel():
    # A WWM-reinforced slab-on-grade is CONCRETE, not a steel member.
    assert classify_member(None, "SLAB ON GRADE THICKNESS 152mm REINFORCING") == "concrete"
    assert classify_member(None, "SLAB REINF. w/ 152x152-MW18.7 WELDED WIRE MESH") == "concrete"
    # ...but a mesh SPEC used as a MARK is rejected (it is not a slab mark)
    assert classify_member("MW18", "152x152-MW18.7 WELDED WIRE MESH") == "reject"


def test_drawing_cross_reference_rejected():
    assert classify_member("S3", "SLAB-ON-GRADE SECTION ON DWG. No.S3") == "reject"


# ── grade-beam extractor ────────────────────────────────────────────────────────

def test_grade_beam_top_of_wall_captured_with_size_and_rebar():
    lines = [
        "350mm WIDE x 300mm Dp.",
        "CONT. CONC. (FORMED)",
        "GRADE BEAM REINF. w/",
        "2-15M CONT. BARS AT TOP OF WALL",
    ]
    gb = extract_grade_beams(lines)
    assert len(gb) == 1, gb
    b = gb[0]
    assert b["element_type"] == "beam"
    assert b["size"] == "350x300"
    assert b["num_rebars"] == 2
    assert b["rebar_size"] == "15M"
    assert b["material"] == "concrete"


def test_grade_beam_no_bleed_from_neighbouring_footing_rebar():
    # Regression: a footing's "3-15M ... TOP & BOTTOM" sits right beside the grade
    # beam. The grade beam must take ONLY its own "2-15M ... AT TOP OF WALL" bars.
    lines = [
        "REINF. w/ 3-15M CONT. BARS TOP & BOTTOM",           # <- footing, must NOT leak
        "(NOTE: TOP OF TRENCH FOOTING IS 300mm BELOW FIN.)",
        "350mm WIDE x 300mm Dp.",
        "CONT. CONC. (FORMED)",
        "GRADE BEAM REINF. w/",
        "2-15M CONT. BARS AT TOP OF WALL",
    ]
    gb = extract_grade_beams(lines)
    assert len(gb) == 1, gb
    assert gb[0]["num_rebars"] == 2 and gb[0]["rebar_size"] == "15M"


def test_grade_beam_requires_top_of_wall_phrase():
    # A "GRADE BEAM (SEE TYPICAL SECTION)" note with no own-rebar phrase → not
    # emitted (we don't fabricate rebar from a neighbour).
    lines = [
        "150 SLAB",
        "350mm WIDE x 300mm Dp. CONC. (FORMED) GRADE BEAM (SEE TYPICAL SECTION)",
        "3-15M CONT. BARS TOP & BOTTOM",   # footing phrasing, not grade-beam
    ]
    assert extract_grade_beams(lines) == []


# ── extract_elements: immediate-context-only (slab regression) ──────────────────

def test_immediate_context_not_greedy_tail():
    """
    A clean concrete slab-on-grade callout followed by unrelated steel/general-
    notes lines must classify from its IMMEDIATE callout, not the greedy tail —
    locks in the slab-steel-misclassification regression fix.
    """
    lines = [
        "SLAB ON GRADE",            # 0: element keyword — starts the slab
        "THICKNESS 152mm",          # 1
        "152 x 152",                # 2: size lives in the immediate callout
        "GRANULAR BASE",            # 3
        "REINFORCING SEE PLAN",     # 4
        # greedy tail (line 5+): unrelated steel / general-notes text that used
        # to leak into classification and wrongly flag the slab as steel
        "All steel design to CAN/CSA-S16-14",
        "HSS COLUMNS EXTENDED ABOVE THE ROOF LEVEL",
        "O.W.S.J. AND METAL DECK",
    ]
    slabs = [e for e in extract_elements(lines) if e["element_type"] == "slab"]
    assert slabs, "slab-on-grade should be captured"
    assert slabs[0].get("material") != "steel", \
        "slab must stay concrete — steel words in the greedy tail must not reclassify it"


# ── standalone runner (no pytest required) ─────────────────────────────────────

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} tests passed")
    sys.exit(0 if passed == len(fns) else 1)
