
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prophet sensitivity visualization script

Purpose
-------
Compare two Prophet sensitivity runs:
1. Full specification
2. No month dummies

Expected inputs
---------------
Two ZIP files, each containing at minimum:
- prophet_summary.csv
- prophet_per_horizon.csv

Outputs
-------
- prophet_sensitivity_overall_test_comparison.csv
- prophet_sensitivity_horizon_comparison.csv
- prophet_sensitivity_rmse_mae_comparison.png
- prophet_sensitivity_rmse_mae_comparison.pdf
- prophet_sensitivity_uncertainty_comparison.png
- prophet_sensitivity_uncertainty_comparison.pdf
- prophet_sensitivity_caption.txt

Recommended manuscript use
--------------------------
Main manuscript:
- prophet_sensitivity_rmse_mae_comparison.(png/pdf)

Supplementary:
- prophet_sensitivity_uncertainty_comparison.(png/pdf)
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
import shutil

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Visualize Prophet sensitivity comparison from two ZIP outputs.")
    p.add_argument("--full_zip", default="./Prophet/outputs/sensitivity/full.zip", help="ZIP for the full Prophet specification")
    p.add_argument("--no_month_zip", default="./Prophet/outputs/sensitivity/no_month_dummies.zip", help="ZIP for the no-month-dummies Prophet specification")
    p.add_argument("--outdir", default="./Prophet/outputs/sensitivity_analysis", help="Output directory")
    return p.parse_args()


def resolve_path(path_like: str) -> Path:
    p = Path(path_like)
    if p.exists():
        return p.resolve()
    cwd_p = Path.cwd() / path_like
    if cwd_p.exists():
        return cwd_p.resolve()
    script_p = Path(__file__).resolve().parent / Path(path_like).name
    if script_p.exists():
        return script_p.resolve()
    return p


def extract_zip(zip_path: Path, extract_dir: Path):
    shutil.rmtree(extract_dir, ignore_errors=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)


def load_prophet_outputs(extract_dir: Path, label: str):
    summary_path = extract_dir / "prophet_summary.csv"
    per_h_path = extract_dir / "prophet_per_horizon.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing prophet_summary.csv in {extract_dir}")
    if not per_h_path.exists():
        raise FileNotFoundError(f"Missing prophet_per_horizon.csv in {extract_dir}")

    summary = pd.read_csv(summary_path)
    per_h = pd.read_csv(per_h_path)

    # Keep only Prophet test row in summary
    summary_test = summary.copy()
    if "Model" in summary_test.columns:
        summary_test = summary_test[summary_test["Model"].astype(str).str.lower().eq("prophet")]
    if "Split" in summary_test.columns:
        summary_test = summary_test[summary_test["Split"].astype(str).str.lower().eq("test")]
    if summary_test.empty:
        raise ValueError(f"No Prophet test row found in {summary_path}")

    summary_test = summary_test.iloc[[0]].copy()
    summary_test.insert(0, "Spec", label)

    per_h = per_h.copy()
    per_h.insert(0, "Spec", label)

    return summary_test, per_h


def build_overall_table(full_summary: pd.DataFrame, nomonth_summary: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "Spec", "n", "RMSE", "MAE", "CVRMSE", "NRMSE", "sMAPE",
        "MBE", "MedAE", "MASE", "Coverage", "MeanIntervalWidth"
    ]
    rows = []
    for df in [full_summary, nomonth_summary]:
        row = {}
        for c in keep:
            row[c] = df.iloc[0][c] if c in df.columns else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)

    # Add delta row: NoMonth - Full
    delta = {"Spec": "Delta (NoMonth - Full)"}
    for c in keep[1:]:
        try:
            delta[c] = out.loc[out["Spec"] == "No month dummies", c].iloc[0] - out.loc[out["Spec"] == "Full", c].iloc[0]
        except Exception:
            delta[c] = np.nan
    out = pd.concat([out, pd.DataFrame([delta])], ignore_index=True)
    return out


def build_horizon_table(full_per_h: pd.DataFrame, nomonth_per_h: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "Spec", "horizon", "RMSE", "MAE", "CVRMSE", "NRMSE", "sMAPE",
        "MASE", "Coverage", "MeanIntervalWidth", "precision", "recall", "f1"
    ]
    frames = []
    for df in [full_per_h, nomonth_per_h]:
        cols = [c for c in keep if c in df.columns]
        frames.append(df[cols].copy())
    out = pd.concat(frames, ignore_index=True)

    # Add horizon-wise delta table merged internally
    common_cols = [c for c in ["RMSE", "MAE", "CVRMSE", "NRMSE", "sMAPE", "MASE", "Coverage", "MeanIntervalWidth", "precision", "recall", "f1"] if c in out.columns]
    full_h = out[out["Spec"] == "Full"].copy()
    nom_h = out[out["Spec"] == "No month dummies"].copy()
    merged = full_h.merge(nom_h, on="horizon", suffixes=("_full", "_nom"))
    delta_rows = []
    for _, r in merged.iterrows():
        d = {"Spec": "Delta (NoMonth - Full)", "horizon": int(r["horizon"])}
        for c in common_cols:
            d[c] = r.get(f"{c}_nom", np.nan) - r.get(f"{c}_full", np.nan)
        delta_rows.append(d)
    out = pd.concat([out, pd.DataFrame(delta_rows)], ignore_index=True, sort=False)
    return out.sort_values(["Spec", "horizon"]).reset_index(drop=True)


def write_caption(outdir: Path):
    caption = (
        "Figure X. Sensitivity analysis of the Prophet model comparing the full specification "
        "with a reduced specification excluding month-dummy regressors. The full specification "
        "showed lower RMSE and MAE across most forecast horizons, indicating better core "
        "count-scale predictive accuracy, whereas the no-month-dummies variant showed only "
        "minor improvements in selected secondary metrics such as sMAPE and interval width. "
        "These results support retaining the full Prophet specification as the primary model."
    )
    (outdir / "prophet_sensitivity_caption.txt").write_text(caption, encoding="utf-8")


def plot_main_figure(full_per_h: pd.DataFrame, nomonth_per_h: pd.DataFrame, outdir: Path):
    full = full_per_h.sort_values("horizon")
    nom = nomonth_per_h.sort_values("horizon")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(full["horizon"], full["RMSE"], marker="o", linewidth=2, label="Full")
    axes[0].plot(nom["horizon"], nom["RMSE"], marker="s", linewidth=2, label="No month dummies")
    axes[0].set_title("RMSE by Forecast Horizon")
    axes[0].set_xlabel("Horizon (months)")
    axes[0].set_ylabel("RMSE")
    axes[0].set_xticks(sorted(full["horizon"].unique()))
    axes[0].legend()

    axes[1].plot(full["horizon"], full["MAE"], marker="o", linewidth=2, label="Full")
    axes[1].plot(nom["horizon"], nom["MAE"], marker="s", linewidth=2, label="No month dummies")
    axes[1].set_title("MAE by Forecast Horizon")
    axes[1].set_xlabel("Horizon (months)")
    axes[1].set_ylabel("MAE")
    axes[1].set_xticks(sorted(full["horizon"].unique()))
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(outdir / "prophet_sensitivity_rmse_mae_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "prophet_sensitivity_rmse_mae_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty_figure(full_per_h: pd.DataFrame, nomonth_per_h: pd.DataFrame, outdir: Path):
    # Create only if both fields exist
    needed = ["Coverage", "MeanIntervalWidth"]
    if not all(c in full_per_h.columns and c in nomonth_per_h.columns for c in needed):
        return

    full = full_per_h.sort_values("horizon")
    nom = nomonth_per_h.sort_values("horizon")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(full["horizon"], full["Coverage"], marker="o", linewidth=2, label="Full")
    axes[0].plot(nom["horizon"], nom["Coverage"], marker="s", linewidth=2, label="No month dummies")
    axes[0].set_title("Coverage by Forecast Horizon")
    axes[0].set_xlabel("Horizon (months)")
    axes[0].set_ylabel("Coverage")
    axes[0].set_xticks(sorted(full["horizon"].unique()))
    axes[0].legend()

    axes[1].plot(full["horizon"], full["MeanIntervalWidth"], marker="o", linewidth=2, label="Full")
    axes[1].plot(nom["horizon"], nom["MeanIntervalWidth"], marker="s", linewidth=2, label="No month dummies")
    axes[1].set_title("Mean Interval Width by Forecast Horizon")
    axes[1].set_xlabel("Horizon (months)")
    axes[1].set_ylabel("Interval width")
    axes[1].set_xticks(sorted(full["horizon"].unique()))
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(outdir / "prophet_sensitivity_uncertainty_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "prophet_sensitivity_uncertainty_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    full_zip = resolve_path(args.full_zip)
    no_month_zip = resolve_path(args.no_month_zip)
    outdir = resolve_path(args.outdir)

    if not full_zip.exists():
        raise FileNotFoundError(f"Full ZIP not found: {full_zip}")
    if not no_month_zip.exists():
        raise FileNotFoundError(f"No-month-dummies ZIP not found: {no_month_zip}")

    outdir.mkdir(parents=True, exist_ok=True)
    tmp_root = outdir / "_tmp_extract"
    full_extract = tmp_root / "full"
    nom_extract = tmp_root / "no_month_dummies"

    extract_zip(full_zip, full_extract)
    extract_zip(no_month_zip, nom_extract)

    full_summary, full_per_h = load_prophet_outputs(full_extract, "Full")
    nom_summary, nom_per_h = load_prophet_outputs(nom_extract, "No month dummies")

    overall_table = build_overall_table(full_summary, nom_summary)
    horizon_table = build_horizon_table(full_per_h, nom_per_h)

    overall_table.to_csv(outdir / "prophet_sensitivity_overall_test_comparison.csv", index=False)
    horizon_table.to_csv(outdir / "prophet_sensitivity_horizon_comparison.csv", index=False)

    plot_main_figure(full_per_h, nom_per_h, outdir)
    plot_uncertainty_figure(full_per_h, nom_per_h, outdir)
    write_caption(outdir)

    print(f"Saved outputs to: {outdir.resolve()}")
    print("\nOverall test comparison:")
    print(overall_table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
