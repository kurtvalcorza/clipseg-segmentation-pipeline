"""DAT20: a refused BYOD input in Section 4 shows exactly one line, ``Refused: <file>: <reason>``, with no traceback,
and the cell raises so Run all stops there; the router relays the refusal as the same line. A genuine failure keeps
its traceback. The Section 4 cell is executed from the committed notebook, as in the CST review-fix tests."""
# ruff: noqa: E501

from __future__ import annotations

import pytest

from test_clipseg_cst_review_fixes import _cell, _section4, _zip, carried, notebook  # noqa: F401  (fixtures)

REFUSAL_CELL = "clipseg_segmentation-17"


def _line(exc: BaseException) -> str:
    assert type(exc).__name__ == "BYODRefusal" and exc.dimer_refusal is True
    rendered = exc._render_traceback_()
    assert len(rendered) == 1 and "\n" not in rendered[0] and "Traceback" not in rendered[0]
    assert exc.__suppress_context__ or exc.__cause__ is None
    return rendered[0]


def test_the_scripted_refusal_input_is_one_line_naming_the_file(notebook, carried, tmp_path):  # noqa: F811
    hostname = tmp_path / "hostname"
    hostname.write_text("runtime-host\n", encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        _section4(notebook, carried, str(hostname))
    line = _line(caught.value)
    assert line.startswith("Refused: hostname: ") and "Run after" in line


def test_a_directory_without_masks_csv_is_refused_by_name(notebook, carried, tmp_path):  # noqa: F811
    folder = tmp_path / "etc"
    folder.mkdir()
    (folder / "hostname").write_text("runtime-host\n", encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        _section4(notebook, carried, str(folder))
    assert _line(caught.value).startswith("Refused: etc: ")


def test_a_refusal_drops_the_results_of_an_earlier_input(notebook, carried, tmp_path):  # noqa: F811
    ok = _section4(notebook, carried, str(_zip(tmp_path / "ok.zip", 12)))
    assert ok["train_records"] and ok["splits"]
    # Re-run the same cell in the namespace that holds the accepted results, as a learner re-running Section 4 would.
    source = _cell(notebook, "USE_BYOD = False").replace("USE_BYOD = False  # @param", "USE_BYOD = True  # @param", 1)
    source = source.replace("BYOD_PATH = ''  # @param", f"BYOD_PATH = {str(_zip(tmp_path / 'eleven.zip', 11))!r}  # @param", 1)
    with pytest.raises(ValueError) as caught:
        exec(compile(source, "<section 4>", "exec"), ok)
    assert not {"records", "splits", "train_records", "val_records", "test_records", "data_source"} & set(ok)
    line = _line(caught.value)
    assert line.startswith("Refused: eleven.zip: Your upload holds 11 records") and "files under outputs/ are from an earlier run" in line


def test_a_genuine_loader_failure_keeps_its_type(notebook, carried, tmp_path):  # noqa: F811
    def split_dataset(records, seed):
        raise RuntimeError("internal bug")

    with pytest.raises(RuntimeError, match="internal bug"):
        _section4(notebook, {**carried, "split_dataset": split_dataset}, str(_zip(tmp_path / "ok.zip", 12)))


def test_the_refusal_cell_id_and_the_router_line(notebook):  # noqa: F811
    cell = next(c for c in notebook["cells"] if c.get("id") == REFUSAL_CELL)
    source = "".join(cell["source"])
    assert cell["cell_type"] == "code" and "BYOD_PATH = ''" in source and "raise BYODRefusal(f'{byod_label}: {exc}" in source
    assert "BYOD upload refused" not in source
    router = [
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code" and "class IsolatedCellRefusal(Exception):" in "".join(c["source"])
    ]
    assert len(router) == 1 and 'return [f"Refused: {self}"]' in router[0] and 'elif kind == "refusal":' in router[0]
