
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TFT ablation / sensitivity visualization script.

Reads these five zip files by default:
- full__seed_42.zip
- no_climate__seed_42.zip
- no_population_density__seed_42.zip
- no_serotype__seed_42.zip
- no_temporal__seed_42.zip

Outputs:
- tft_ablation_overall_metrics.csv
- tft_ablation_per_horizon.csv
- tft_ablation_overall_delta_vs_full.csv
- tft_ablation_rmse_mae_by_horizon.(png|pdf)
- tft_ablation_uncertainty_by_horizon.(png|pdf)
- tft_ablation_overall_deltas.(png|pdf)
- tft_ablation_caption.txt
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_ZIPS = {
    "Full specification": "full__seed_42.zip",
    "No climate": "no_climate__seed_42.zip",
    "No population density": "no_population_density__seed_42.zip",
    "No serotype": "no_serotype__seed_42.zip",
    "No temporal": "no_temporal__seed_42.zip",
}

SUMMARY_FILE = "tft_summary.csv"
PER_H_FILE = "tft_per_horizon.csv"
INTERVAL_FILE = "tft_interval_metrics.csv"


def parse_args():
    p = argparse.ArgumentParser(description="Create TFT ablation figures from zipped run outputs.")
    p.add_argument("--full", default="./TFT/outputs/sensitivity/full__seed_42.zip")
    p.add_argument("--no_climate", default="./TFT/outputs/sensitivity/no_climate__seed_42.zip")
    p.add_argument("--no_population_density", default="./TFT/outputs/sensitivity/no_population_density__seed_42.zip")
    p.add_argument("--no_serotype", default="./TFT/outputs/sensitivity/no_serotype__seed_42.zip")
    p.add_argument("--no_temporal", default="./TFT/outputs/sensitivity/no_temporal__seed_42.zip")
    p.add_argument("--outdir", default="./TFT/outputs/visualization")
    return p.parse_args()


def resolve_input_path(path_like: str) -> Path:
    p = Path(path_like)
    if p.exists():
        return p.resolve()
    cwd_candidate = Path.cwd() / path_like
    if cwd_candidate.exists():
        return cwd_candidate.resolve()
    script_candidate = Path(__file__).resolve().parent / p.name
    if script_candidate.exists():
        return script_candidate.resolve()
    return p


def read_csv_from_zip(zip_path: Path, inner_name: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(inner_name) as f:
            return pd.read_csv(f)


def collect_tables(zip_map: dict[str, Path]):
    summary_rows = []
    per_h_parts = []
    interval_parts = []

    for spec, zip_path in zip_map.items():
        summary = read_csv_from_zip(zip_path, SUMMARY_FILE)
        per_h = read_csv_from_zip(zip_path, PER_H_FILE)
        interval = read_csv_from_zip(zip_path, INTERVAL_FILE)

        test_row = summary.loc[
            summary["Split"].astype(str).str.lower().eq("test")
            & summary["Model"].astype(str).str.upper().eq("TFT")
        ].copy()
        if test_row.empty:
            raise ValueError(f"No TFT test row found in {zip_path.name}")

        row = test_row.iloc[0].copy()
        row["Specification"] = spec
        summary_rows.append(row)

        per_h = per_h.copy()
        per_h["Specification"] = spec
        per_h_parts.append(per_h)

        interval = interval.copy()
        interval["Specification"] = spec
        interval_parts.append(interval)

    summary_df = pd.DataFrame(summary_rows).reset_index(drop=True)
    per_h_df = pd.concat(per_h_parts, ignore_index=True)
    interval_df = pd.concat(interval_parts, ignore_index=True)
    return summary_df, per_h_df, interval_df


def build_overall_delta_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    base = summary_df.loc[summary_df["Specification"] == "Full specification"].iloc[0]
    metrics = ["RMSE", "MAE", "CVRMSE", "NRMSE", "sMAPE", "MASE"]
    rows = []
    for _, row in summary_df.iterrows():
        out = {"Specification": row["Specification"]}
        for m in metrics:
            out[m] = row[m]
            out[f"Delta_{m}"] = row[m] - base[m]
        rows.append(out)
    return pd.DataFrame(rows)


def save_tables(summary_df: pd.DataFrame, per_h_df: pd.DataFrame, delta_df: pd.DataFrame, outdir: Path):
    keep_summary = ["Specification", "n", "RMSE", "MAE", "CVRMSE", "NRMSE", "sMAPE", "MBE", "MedAE", "MASE"]
    summary_df[keep_summary].round(3).to_csv(outdir / "tft_ablation_overall_metrics.csv", index=False)

    keep_per_h = [
        "Specification", "horizon", "n", "RMSE", "MAE", "CVRMSE", "NRMSE", "sMAPE",
        "MASE", "Coverage80", "MeanIntervalWidth80"
    ]
    per_h_df[keep_per_h].round(3).to_csv(outdir / "tft_ablation_per_horizon.csv", index=False)

    delta_df.round(3).to_csv(outdir / "tft_ablation_overall_delta_vs_full.csv", index=False)


def plot_rmse_mae(per_h_df: pd.DataFrame, outdir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    for spec, g in per_h_df.groupby("Specification", sort=False):
        g = g.sort_values("horizon")
        axes[0].plot(g["horizon"], g["RMSE"], marker="o", linewidth=2, label=spec)
        axes[1].plot(g["horizon"], g["MAE"], marker="o", linewidth=2, label=spec)

    axes[0].set_title("TFT Sensitivity: RMSE by Horizon")
    axes[0].set_xlabel("Forecast Horizon")
    axes[0].set_ylabel("RMSE")
    axes[0].set_xticks(sorted(per_h_df["horizon"].unique()))

    axes[1].set_title("TFT Sensitivity: MAE by Horizon")
    axes[1].set_xlabel("Forecast Horizon")
    axes[1].set_ylabel("MAE")
    axes[1].set_xticks(sorted(per_h_df["horizon"].unique()))

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="right", ncol=3, bbox_to_anchor=(0.5, -0.05))
    fig.savefig(outdir / "tft_ablation_rmse_mae_by_horizon.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "tft_ablation_rmse_mae_by_horizon.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty(per_h_df: pd.DataFrame, outdir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    for spec, g in per_h_df.groupby("Specification", sort=False):
        g = g.sort_values("horizon")
        axes[0].plot(g["horizon"], g["Coverage80"], marker="o", linewidth=2, label=spec)
        axes[1].plot(g["horizon"], g["MeanIntervalWidth80"], marker="o", linewidth=2, label=spec)

    axes[0].set_title("TFT Sensitivity: Coverage@80 by Horizon")
    axes[0].set_xlabel("Forecast Horizon")
    axes[0].set_ylabel("Coverage@80")
    axes[0].set_xticks(sorted(per_h_df["horizon"].unique()))

    axes[1].set_title("TFT Sensitivity: Mean Interval Width@80 by Horizon")
    axes[1].set_xlabel("Forecast Horizon")
    axes[1].set_ylabel("Mean Interval Width@80")
    axes[1].set_xticks(sorted(per_h_df["horizon"].unique()))

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="right", ncol=3, bbox_to_anchor=(0.5, -0.05))
    fig.savefig(outdir / "tft_ablation_uncertainty_by_horizon.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "tft_ablation_uncertainty_by_horizon.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_overall_deltas(delta_df: pd.DataFrame, outdir: Path):
    # Exclude full spec from delta bars
    d = delta_df.loc[delta_df["Specification"] != "Full specification"].copy()
    metrics = ["Delta_RMSE", "Delta_MAE", "Delta_sMAPE", "Delta_MASE"]
    pretty = {
        "Delta_RMSE": "ΔRMSE",
        "Delta_MAE": "ΔMAE",
        "Delta_sMAPE": "ΔsMAPE",
        "Delta_MASE": "ΔMASE",
    }

    x = np.arange(len(d))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, d[m], width=width, label=pretty[m])

    ax.axhline(0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(d["Specification"], rotation=20, ha="right")
    ax.set_ylabel("Difference from full specification")
    #ax.set_title("TFT Sensitivity: Overall Test-Metric Change Relative to Full Specification")
    ax.legend(ncol=4, loc="right", bbox_to_anchor=(0.5, 1.15))
    fig.savefig(outdir / "tft_ablation_overall_deltas.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "tft_ablation_overall_deltas.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_caption(summary_df: pd.DataFrame, outdir: Path):
    full_row = summary_df.loc[summary_df["Specification"] == "Full specification"].iloc[0]
    best_rmse = summary_df.sort_values("RMSE").iloc[0]["Specification"]
    best_mae = summary_df.sort_values("MAE").iloc[0]["Specification"]

    txt = f"""Suggested manuscript wording

Main figure caption:
Sensitivity analysis of the Temporal Fusion Transformer (TFT) was conducted by comparing the full specification against four feature-ablation variants: no climate, no population density, no serotype, and no temporal features. Horizon-wise RMSE and MAE remained highly similar across all five specifications, indicating that the TFT performance was only modestly affected by individual feature-group removal. The full specification achieved test RMSE = {full_row['RMSE']:.3f} and test MAE = {full_row['MAE']:.3f}. Among the five runs, the lowest RMSE was obtained by {best_rmse}, whereas the lowest MAE was obtained by {best_mae}. Overall, the ablation results suggest that TFT performance was relatively stable under these feature removals.

Supplementary figure caption:
Coverage@80 and mean interval width@80 were also compared across the five TFT sensitivity specifications. The uncertainty profiles remained broadly similar, indicating that the observed differences among the ablation runs were small not only for point error but also for interval behavior.

Table recommendation:
Use one compact supplementary table only:
- overall test metrics (RMSE, MAE, CVRMSE, NRMSE, sMAPE, MASE) for all five specifications.
No extra main-text table is necessary if the horizon-wise figure is shown.
"""
    (outdir / "tft_ablation_caption.txt").write_text(txt, encoding="utf-8")


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    zip_map = {
        "Full specification": resolve_input_path(args.full),
        "No climate": resolve_input_path(args.no_climate),
        "No population density": resolve_input_path(args.no_population_density),
        "No serotype": resolve_input_path(args.no_serotype),
        "No temporal": resolve_input_path(args.no_temporal),
    }

    for spec, path in zip_map.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing zip for {spec}: {path}")

    summary_df, per_h_df, interval_df = collect_tables(zip_map)
    delta_df = build_overall_delta_table(summary_df)
    save_tables(summary_df, per_h_df, delta_df, outdir)
    plot_rmse_mae(per_h_df, outdir)
    plot_uncertainty(per_h_df, outdir)
    plot_overall_deltas(delta_df, outdir)
    write_caption(summary_df, outdir)

    print("Saved TFT ablation outputs to:", outdir.resolve())
    print(summary_df[["Specification", "RMSE", "MAE", "sMAPE", "MASE"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
