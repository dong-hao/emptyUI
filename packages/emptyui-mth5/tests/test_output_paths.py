"""Shared output-path default (docs/issues.md M29)."""

from __future__ import annotations

from pathlib import Path

import pytest
from emptyui_mth5._paths import DEFAULT_OUTPUT_DIRNAME, resolve_output_dir


def test_empty_save_path_uses_the_dedicated_subfolder(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    out = resolve_output_dir("", data)
    assert out == data / DEFAULT_OUTPUT_DIRNAME
    assert out.is_dir()
    # the data directory itself gained only the subfolder — no archive beside
    # the raw files
    assert list(data.iterdir()) == [out]


def test_explicit_save_path_wins_and_is_created(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    explicit = tmp_path / "chosen"
    out = resolve_output_dir(str(explicit), data)
    assert out == explicit
    assert out.is_dir()


def test_save_path_that_is_a_file_raises(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    f = tmp_path / "not_a_dir"
    f.write_text("x")
    with pytest.raises(FileExistsError):
        resolve_output_dir(str(f), data)
