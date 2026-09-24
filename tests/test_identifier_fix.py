"""Tests for the identifier NFKC collision fix preview.

Covers the pure ``preview_collision_fix`` helper and the
``POST /api/v1/identifiers/normalization/fix-preview`` endpoint: only
non-canonical spellings of the targeted group are rewritten (canonical
spellings and other groups untouched), strings/comments/other characters
are preserved verbatim, edit offsets slice back to the original spelling,
BOM/CRLF and edit ordering survive, the fixed source rescans with the
targeted collision gone, 422 on a missing group or lexical failure, and the
preview is never persisted.

Non-ASCII characters are written as \\uXXXX escapes so the file's exact
codepoint composition is independent of editor normalisation.
"""

import pytest

from app.identifiers import (
    CollisionGroupNotFound,
    IdentifierLexError,
    preview_collision_fix,
    scan_identifiers,
)

ACUTE = "é"                       # precomposed e-acute (canonical)
E_COMBINING = "e" + chr(0x0301)   # 'e' + combining acute U+0301
FW_X = "Ｘ"                       # fullwidth Latin capital X
MATH_X = "𝐗"                      # mathematical bold capital X (astral)
FW_A = "ａ"                        # fullwidth Latin small a
FW_B = "ｂ"                        # fullwidth Latin small b
MATH_A = "𝐚"                       # mathematical bold small a
MATH_B = "𝐛"                       # mathematical bold small b
FI_LIGATURE = "ﬁ"                 # U+FB01 LATIN SMALL LIGATURE FI
FW_FI = "ｆｉ"                    # fullwidth f + fullwidth i, NFKC -> 'fi'
BOM = "﻿"


def _apply(source, edits):
    """Apply an edit list to *source* (mirrors the endpoint contract)."""
    pieces = []
    cursor = 0
    for edit in sorted(edits, key=lambda e: e["offset"]):
        assert source[edit["offset"]:edit["end_offset"]] == edit["original"]
        pieces.append(source[cursor:edit["offset"]])
        pieces.append(edit["replacement"])
        cursor = edit["end_offset"]
    pieces.append(source[cursor:])
    return "".join(pieces)


# ------------------------------------------------------- unit: the repair --

def test_only_non_canonical_spellings_are_replaced():
    # Canonical spelling 'fi' is ASCII; the ligature is the odd spelling.
    src = f"fi = 1\nx = {FI_LIGATURE}\nfi2 = {FI_LIGATURE} + fi\n"
    preview = preview_collision_fix(src, "fi")
    assert preview.normalised == "fi"
    assert len(preview.edits) == 2
    # edits carry original-source codepoint offsets and both texts
    for edit in preview.edits:
        assert edit.original == FI_LIGATURE
        assert edit.replacement == "fi"
        assert src[edit.offset:edit.end_offset] == FI_LIGATURE
    # source order
    assert [e.offset for e in preview.edits] == sorted(e.offset for e in preview.edits)
    assert _apply(src, [e.to_dict() for e in preview.edits]) == preview.fixed_source
    assert FI_LIGATURE not in preview.fixed_source
    assert preview.fixed_source == "fi = 1\nx = fi\nfi2 = fi + fi\n"


def test_canonical_spelling_needs_no_edit_when_other_variants_exist():
    # The canonical form X is present as ASCII; fullwidth and math bold are
    # both replaced, the ASCII occurrences are not themselves edited.
    src = f"X = 1\nx2 = {FW_X}\nx3 = {MATH_X} + X\n"
    preview = preview_collision_fix(src, "X")
    originals = [edit.original for edit in preview.edits]
    assert sorted(originals) == sorted([FW_X, MATH_X])
    assert all(edit.replacement == "X" for edit in preview.edits)
    assert preview.fixed_source == "X = 1\nx2 = X\nx3 = X + X\n"


def test_canonical_spelling_itself_need_not_be_present():
    # No raw spelling equals 'fi'; the two odd spellings (ligature and
    # fullwidth) both normalise to it and are both rewritten.
    src = f"{FI_LIGATURE} = 1\ny = {FW_FI}\n"
    preview = preview_collision_fix(src, "fi")
    assert len(preview.edits) == 2
    assert {e.original for e in preview.edits} == {FI_LIGATURE, FW_FI}
    assert all(e.replacement == "fi" for e in preview.edits)
    assert preview.fixed_source == "fi = 1\ny = fi\n"


def test_three_way_group_collapses_to_canonical():
    src = f"{FW_X} = 1\n{MATH_X} = 2\nX = 3\n"
    preview = preview_collision_fix(src, "X")
    assert [edit.original for edit in preview.edits] == [FW_X, MATH_X]
    assert preview.fixed_source == "X = 1\nX = 2\nX = 3\n"


def test_other_collision_groups_are_left_untouched():
    src = f"{MATH_A} = 1\n{MATH_B} = 2\n{FW_A} = 3\n{FW_B} = 4\n"
    preview = preview_collision_fix(src, "a")
    assert all(edit.replacement == "a" for edit in preview.edits)
    assert len(preview.edits) == 2  # math A and fullwidth A, not the Bs
    # the b group survives verbatim in the fixed source
    assert MATH_B in preview.fixed_source and FW_B in preview.fixed_source
    # rescanned report loses the 'a' collision but keeps 'b'
    normalised = [c.normalised for c in preview.report.collisions]
    assert normalised == ["b"]


def test_strings_comments_and_identifiers_outside_group_are_preserved():
    src = (
        f"# keep {FI_LIGATURE} in comments and {MATH_X}\n"
        f"s = '{FI_LIGATURE} string {FW_X}'\n"
        f"{FI_LIGATURE} = 1\nfi = 2\n"
        f"other = {MATH_X}\n"
    )
    preview = preview_collision_fix(src, "fi")
    # exactly one replacement: the bare ligature identifier on line 3
    assert len(preview.edits) == 1
    edit = preview.edits[0]
    assert src[edit.offset:edit.end_offset] == FI_LIGATURE
    assert edit.offset == len(
        f"# keep {FI_LIGATURE} in comments and {MATH_X}\n"
        f"s = '{FI_LIGATURE} string {FW_X}'\n"
    )
    fixed = preview.fixed_source
    # comment and string contents keep their raw codepoints
    assert f"# keep {FI_LIGATURE} in comments" in fixed
    assert f"'{FI_LIGATURE} string {FW_X}'" in fixed
    # an identifier outside the targeted group is untouched
    assert f"other = {MATH_X}" in fixed
    # nothing but the one identifier span changed
    assert fixed == _apply(src, [e.to_dict() for e in preview.edits])


def test_codepoint_offsets_hold_for_astral_replacements():
    # The replacement ('X') is shorter in codepoints than the math bold X
    # (single astral codepoint) / fullwidth X; offsets index the ORIGINAL.
    src = f"abc = 1\n{MATH_X} = 2\n{FW_X} = 3\n"
    preview = preview_collision_fix(src, "X")
    for edit in preview.edits:
        assert src[edit.offset:edit.end_offset] == edit.original
    assert preview.edits[0].offset == len("abc = 1\n")
    assert preview.fixed_source == "abc = 1\nX = 2\nX = 3\n"


def test_combining_mark_spelling_is_replaced_and_rescan_is_clean():
    src = f"{ACUTE} = 1\nx = {E_COMBINING}\n"
    preview = preview_collision_fix(src, ACUTE)
    assert len(preview.edits) == 1
    assert preview.edits[0].original == E_COMBINING
    assert preview.edits[0].replacement == ACUTE
    assert preview.fixed_source == f"{ACUTE} = 1\nx = {ACUTE}\n"
    # the decomposed identifier is gone and rescanning reports no collision
    assert E_COMBINING not in preview.fixed_source
    assert preview.report.collision_count == 0


def test_bom_and_crlf_are_preserved():
    src = f"{BOM}{FI_LIGATURE} = 1\r\nfi = 2\r\n"
    preview = preview_collision_fix(src, "fi")
    assert len(preview.edits) == 1
    # offset accounts for the leading BOM
    assert preview.edits[0].offset == 1
    fixed = preview.fixed_source
    assert fixed.startswith(BOM)
    assert fixed == f"{BOM}fi = 1\r\nfi = 2\r\n"
    assert "\r\n" in fixed and "\n" not in fixed.replace("\r\n", "")


def test_lf_newlines_and_no_trailing_newline_are_preserved():
    src = f"{FI_LIGATURE} = {FW_FI}"  # no newline at end
    preview = preview_collision_fix(src, "fi")
    assert preview.fixed_source == "fi = fi"


def test_edits_are_in_source_order_and_to_dict_shape():
    src = f"{MATH_X} {FW_X} {MATH_X}\n"
    preview = preview_collision_fix(src, "X")
    offsets = [edit.offset for edit in preview.edits]
    assert offsets == sorted(offsets)
    d = preview.to_dict()
    assert d["normalised"] == "X"
    assert d["edit_count"] == 3
    assert set(d["edits"][0]) == {
        "offset", "end_offset", "original", "replacement"
    }
    assert "fixed_source" in d and "report" in d


def test_rescanned_report_has_one_fewer_collision():
    src = f"{MATH_A} {FW_A}\n{MATH_B} {FW_B}\n"
    before = scan_identifiers(src)
    assert before.collision_count == 2
    preview = preview_collision_fix(src, "b")
    assert preview.report.collision_count == 1
    assert preview.report.collisions[0].normalised == "a"
    # the fixed group is now one canonical spelling, still counted once
    names = {
        s.spelling
        for c in preview.report.collisions
        for s in c.spellings
    }
    assert MATH_B not in names and FW_B not in names


# --------------------------------------------------------- unit: bad input --

def test_missing_group_raises():
    with pytest.raises(CollisionGroupNotFound):
        preview_collision_fix(f"{FI_LIGATURE} = 1\nfi = 2\n", "nope")


def test_single_spelling_name_is_not_a_group():
    # 'foo' occurs but never collides
    with pytest.raises(CollisionGroupNotFound):
        preview_collision_fix("foo = 1\nfoo = foo\n", "foo")


def test_collision_with_a_different_normalised_name_is_not_targeted():
    src = f"{FI_LIGATURE} = 1\nfi = 2\n"
    with pytest.raises(CollisionGroupNotFound):
        preview_collision_fix(src, "x")  # the 'X' group does not exist here


def test_lex_failure_propagates():
    with pytest.raises(IdentifierLexError) as exc:
        preview_collision_fix(f"x = 1 $\n{FI_LIGATURE} = 2\n", "fi")
    assert exc.value.error == "unexpected_character"


def test_collision_in_otherwise_unlexable_source_is_not_repaired():
    # The '$' makes scanning fail before any group can be located.
    with pytest.raises(IdentifierLexError):
        preview_collision_fix(f"{FI_LIGATURE} = 1 $\nfi = 2\n", "fi")


# ----------------------------------------------------------------- API: 200 -

def test_api_preview_basic_shape(client):
    src = f"{FI_LIGATURE} = 1\nfi = 2\n"
    resp = client.post(
        "/api/v1/identifiers/normalization/fix-preview",
        json={"source": src, "normalised": "fi"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["stored"] is False
    assert body["normalised"] == "fi"
    assert body["edit_count"] == 1
    edit = body["edits"][0]
    assert edit == {
        "offset": 0,
        "end_offset": 1,
        "original": FI_LIGATURE,
        "replacement": "fi",
    }
    assert body["fixed_source"] == "fi = 1\nfi = 2\n"
    report = body["report"]
    assert report["collision_count"] == 0


def test_api_preview_preserves_strings_and_comments(client):
    src = (
        f"# {FI_LIGATURE} comment\n"
        f"s = '{FI_LIGATURE}'\n"
        f"{FI_LIGATURE} = 1\nfi = 2\n"
    )
    body = client.post(
        "/api/v1/identifiers/normalization/fix-preview",
        json={"source": src, "normalised": "fi"},
    ).json()
    assert body["edit_count"] == 1
    assert body["edits"][0]["offset"] == len(f"# {FI_LIGATURE} comment\ns = '{FI_LIGATURE}'\n")
    fixed = body["fixed_source"]
    assert f"# {FI_LIGATURE} comment" in fixed
    assert f"s = '{FI_LIGATURE}'" in fixed


def test_api_preview_bom_and_crlf_round_trip(client):
    src = f"{BOM}{MATH_X} = 1\r\n{FW_X} = 2\r\n"
    body = client.post(
        "/api/v1/identifiers/normalization/fix-preview",
        json={"source": src, "normalised": "X"},
    ).json()
    assert body["fixed_source"] == f"{BOM}X = 1\r\nX = 2\r\n"
    # rescanning the fixed source finds no X collision
    assert body["report"]["collision_count"] == 0


def test_api_preview_other_group_remains(client):
    src = f"{MATH_A} {FW_A}\n{MATH_B} {FW_B}\n"
    body = client.post(
        "/api/v1/identifiers/normalization/fix-preview",
        json={"source": src, "normalised": "a"},
    ).json()
    assert [c["normalised"] for c in body["report"]["collisions"]] == ["b"]


# --------------------------------------------------------------- API: 422 ---

def test_api_missing_group_422(client):
    resp = client.post(
        "/api/v1/identifiers/normalization/fix-preview",
        json={"source": f"{FI_LIGATURE} = 1\nfi = 2\n", "normalised": "zzz"},
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "collision_group_not_found"
    assert detail["normalised"] == "zzz"
    assert "message" in detail


def test_api_non_collision_name_422(client):
    resp = client.post(
        "/api/v1/identifiers/normalization/fix-preview",
        json={"source": "foo = 1\n", "normalised": "foo"},
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "collision_group_not_found"


def test_api_lex_failure_422_with_position(client):
    resp = client.post(
        "/api/v1/identifiers/normalization/fix-preview",
        json={"source": "x = 1 $\n", "normalised": "fi"},
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "unexpected_character"
    assert detail["offset"] == 6
    assert detail["line"] == 1 and detail["column"] == 7


def test_api_missing_fields_422(client):
    assert client.post(
        "/api/v1/identifiers/normalization/fix-preview",
        json={"source": "x = 1\n"},
    ).status_code == 422
    assert client.post(
        "/api/v1/identifiers/normalization/fix-preview",
        json={"normalised": "fi"},
    ).status_code == 422
    # empty normalised name is also a validation error
    assert client.post(
        "/api/v1/identifiers/normalization/fix-preview",
        json={"source": "x = 1\n", "normalised": ""},
    ).status_code == 422


# --------------------------------------------------------- API: persistence -

def test_api_preview_is_never_persisted(client):
    client.post(
        "/api/v1/identifiers/normalization/fix-preview",
        json={"source": f"{FI_LIGATURE} = 1\nfi = 2\n", "normalised": "fi"},
    )
    # no identifier report row and no audit row were created
    assert client.get(
        "/api/v1/identifiers/normalization/1"
    ).status_code == 404
    assert client.get("/api/v1/audits").json()["total"] == 0

    # a later stored report still gets id 1
    stored = client.post(
        "/api/v1/identifiers/normalization", json={"source": "y = 1\n"}
    )
    assert stored.status_code == 201
    assert stored.json()["id"] == 1


def test_api_failed_preview_writes_nothing(client):
    resp = client.post(
        "/api/v1/identifiers/normalization/fix-preview",
        json={"source": "x = 1 $\n", "normalised": "fi"},
    )
    assert resp.status_code == 422
    assert client.get("/api/v1/audits").json()["total"] == 0
    ok = client.post(
        "/api/v1/identifiers/normalization", json={"source": "y = 1\n"}
    )
    assert ok.json()["id"] == 1


def test_root_lists_fix_preview_endpoint(client):
    body = client.get("/").json()
    assert (
        "POST /api/v1/identifiers/normalization/fix-preview"
        in body["endpoints"]
    )
