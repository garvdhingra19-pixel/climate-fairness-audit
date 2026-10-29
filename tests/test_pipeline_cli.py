import json

import pytest

from cfa.cli import build_parser, main
from cfa.pipeline import run_demo


def test_demo_pipeline_writes_artifacts_and_finds_known_gap(tmp_path):
    summary = run_demo(tmp_path, n_districts=150, seed=3, n_boot=100)
    for name in (
        "districts_dvi.csv",
        "stations.csv",
        "audit_by_dvi_quintile.csv",
        "audit_by_settlement.csv",
        "audit_by_wealth_quintile.csv",
        "error_by_dvi_decile.csv",
        "audit_overview.png",
        "summary.json",
    ):
        assert (tmp_path / name).exists(), name
    gap = summary["gap_least_vs_most_visible"]
    assert gap["gap"] > 0  # synthetic data is built so low visibility => larger error
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert "SYNTHETIC" in saved["disclaimer"]


def test_cli_runs(tmp_path, capsys):
    assert main(["demo", "--out", str(tmp_path), "--n-districts", "60", "--n-boot", "60"]) == 0
    assert "SYNTHETIC" in capsys.readouterr().out


def test_cli_rejects_bad_args():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["demo", "--n-districts", "0"])
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
