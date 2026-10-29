"""End-to-end demo pipeline on synthetic data: DVI -> sensitivity -> audit -> figure."""

from __future__ import annotations

import json
from pathlib import Path

from cfa.config import QUINTILE_LABELS, SYNTHETIC_DISCLAIMER
from cfa.data.synthetic import add_synthetic_predictions, generate_synthetic_data
from cfa.fairness.audit import error_by_bin, gap_with_ci, group_metrics
from cfa.visibility.dvi import (
    assign_quintiles,
    compute_dvi,
    compute_visibility_inputs,
    sensitivity_analysis,
)
from cfa.viz import plot_audit_overview


def run_demo(
    out_dir: str | Path, n_districts: int = 300, seed: int = 42, n_boot: int = 500
) -> dict:
    """Run the full synthetic pipeline, write artefacts to out_dir, and return a summary dict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    districts, stations = generate_synthetic_data(n_districts, seed)
    inputs = compute_visibility_inputs(districts, stations)
    scores = compute_dvi(inputs)
    sens = sensitivity_analysis(scores, seed=seed)

    table = districts.merge(inputs, on="district_id").merge(scores, on="district_id")
    table["dvi_quintile"] = assign_quintiles(table["dvi"]).to_numpy()
    table["quintile_stability"] = sens.district_stability.to_numpy()
    table = add_synthetic_predictions(table, seed=seed + 1)

    by_quintile = group_metrics(
        table, "dvi_quintile", n_boot=n_boot, seed=seed, order=list(QUINTILE_LABELS)
    )
    by_settlement = group_metrics(
        table, "settlement", n_boot=n_boot, seed=seed, order=["rural", "mixed", "urban"]
    )
    by_wealth = group_metrics(
        table,
        "wealth_quintile",
        n_boot=n_boot,
        seed=seed,
        order=["W1_poorest", "W2", "W3", "W4", "W5_richest"],
    )
    gap = gap_with_ci(
        table, "dvi_quintile", QUINTILE_LABELS[0], QUINTILE_LABELS[-1], n_boot=n_boot, seed=seed
    )
    binned = error_by_bin(table, "dvi")

    table.to_csv(out / "districts_dvi.csv", index=False)
    stations.to_csv(out / "stations.csv", index=False)
    by_quintile.to_csv(out / "audit_by_dvi_quintile.csv")
    by_settlement.to_csv(out / "audit_by_settlement.csv")
    by_wealth.to_csv(out / "audit_by_wealth_quintile.csv")
    binned.to_csv(out / "error_by_dvi_decile.csv", index=False)
    fig_path = plot_audit_overview(
        table,
        stations,
        by_quintile,
        binned,
        out / "audit_overview.png",
        title_suffix=SYNTHETIC_DISCLAIMER,
    )

    summary = {
        "disclaimer": SYNTHETIC_DISCLAIMER,
        "n_districts": int(len(table)),
        "n_stations": int(len(stations)),
        "districts_without_station": int((inputs["stations_per_1000km2"] == 0).sum()),
        "gap_least_vs_most_visible": gap,
        "sensitivity": sens.summary,
        "figure": str(fig_path),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


def print_summary(summary: dict) -> None:
    gap = summary["gap_least_vs_most_visible"]
    sens = summary["sensitivity"]
    print(summary["disclaimer"])
    print(
        f"Districts: {summary['n_districts']}  Stations: {summary['n_stations']}  "
        f"Districts with no station: {summary['districts_without_station']}"
    )
    print(
        f"RMSE least visible (Q1): {gap['value_a']:.2f}   most visible (Q5): {gap['value_b']:.2f}"
    )
    print(
        f"Gap: {gap['gap']:.2f}  95% CI [{gap['gap_lo']:.2f}, {gap['gap_hi']:.2f}]  "
        f"significant={gap['significant']}"
    )
    print(
        f"DVI weight-sensitivity: Spearman mean={sens['spearman_mean']:.3f}, "
        f"min={sens['spearman_min']:.3f}"
    )
    print(f"Figure: {summary['figure']}")
