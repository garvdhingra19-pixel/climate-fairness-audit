"""Plots for the audit overview figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def plot_audit_overview(
    districts: pd.DataFrame,
    stations: pd.DataFrame,
    quintile_metrics: pd.DataFrame,
    binned: pd.DataFrame,
    path: str | Path,
    title_suffix: str = "",
) -> Path:
    """Three panels: DVI map with stations, RMSE by visibility quintile, error vs DVI."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    sc = ax.scatter(
        districts["lon"],
        districts["lat"],
        c=districts["dvi"],
        cmap="viridis",
        s=26,
        vmin=0,
        vmax=100,
    )
    ax.scatter(
        stations["lon"],
        stations["lat"],
        marker="x",
        s=10,
        c="black",
        linewidths=0.6,
        label="monitoring station",
    )
    ax.set_title("Data Visibility Index by district")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="lower right", fontsize=8)
    fig.colorbar(sc, ax=ax, label="DVI (0 = invisible, 100 = well seen)")

    ax = axes[1]
    labels = [str(i).replace("_", "\n") for i in quintile_metrics.index]
    x = np.arange(len(labels))
    yerr = np.vstack(
        [
            quintile_metrics["rmse"] - quintile_metrics["rmse_lo"],
            quintile_metrics["rmse_hi"] - quintile_metrics["rmse"],
        ]
    )
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(labels)))
    ax.bar(x, quintile_metrics["rmse"], yerr=yerr, capsize=4, color=colors)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("RMSE (ug/m3)")
    ax.set_title("Model error by visibility quintile (95% CI)")

    ax = axes[2]
    ax.scatter(
        districts["dvi"],
        (districts["pm25_pred"] - districts["pm25_true"]).abs(),
        s=12,
        alpha=0.35,
        label="district",
    )
    ax.plot(
        binned["x_mean"], binned["mae"], color="crimson", marker="o", label="decile mean abs. error"
    )
    ax.set_xlabel("DVI")
    ax.set_ylabel("Absolute error (ug/m3)")
    ax.set_title("Error vs visibility (dose-response)")
    ax.legend(fontsize=8)

    if title_suffix:
        fig.suptitle(title_suffix, fontsize=10, color="crimson")
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out
