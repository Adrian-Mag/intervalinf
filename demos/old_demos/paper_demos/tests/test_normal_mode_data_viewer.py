"""Tests for the normal-mode data viewer utility."""

import os
import sys
from pathlib import Path

import pytest


_TEST_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_TEST_DIR, "../utils"))
sys.path.insert(0, os.path.join(_TEST_DIR, "../visualization"))

from normal_mode_data_viewer import (  # noqa: E402
    NormalModeDataRepository,
    NormalModeViewer,
    build_parser,
)


class _DummyPlot:
    def subplots(
        self, *args, **kwargs,
    ):  # pragma: no cover - should not be reached
        raise AssertionError("viewer should fail before creating a figure")


class _DummyWidget:
    pass


def test_default_data_root_points_to_paper_demo_data():
    parser = build_parser()
    args = parser.parse_args([])

    expected = Path(__file__).resolve().parents[1] / "data"

    assert args.data_root == expected
    assert (args.data_root / "normal-mode-synthetics").is_dir()


def test_viewer_reports_empty_synthetic_catalog(tmp_path):
    repository = NormalModeDataRepository(tmp_path)
    args = build_parser().parse_args(["--data-root", str(tmp_path)])

    with pytest.raises(SystemExit, match="No synthetic model files found"):
        NormalModeViewer(
            repository,
            args,
            _DummyPlot(),
            _DummyWidget,
            _DummyWidget,
            _DummyWidget,
        )
