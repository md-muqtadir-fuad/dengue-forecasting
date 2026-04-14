#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCMCI (nonlinear) causal discovery script for dengue time-series analysis.

This script is designed to be practical for manuscript workflows:
- user-config section at the top
- national or district-wise mode
- GPDC-based nonlinear conditional independence testing via Tigramite PCMCI
- manuscript-style summary table for chosen predictors -> target hypotheses
- figure exports (graph, p-value heatmaps, effect-value heatmaps, summary bars)

Install requirements if needed:
    pip install tigramite pandas numpy scipy matplotlib seaborn

Notes
-----
1. For exact manuscript replication, use a preprocessing file that contains the exact
   variables of interest (for example current Temperature, Rainfall, DENV-2, DENV-3,
   and Dengue target).
2. In national mode, if your file still contains multiple district rows per date, the script
   can auto-aggregate numeric columns by date. For log1p-transformed targets, that may not
   match a true national-count analysis, so a genuinely national input file is preferred.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import combine_pvalues

# =========================================================
# USER CONFIG: 
# =========================================================
INPUT_PATH = "./data/raw/prime_dataset_model_input_with_purge.csv"
OUTPUT_DIR = "./Causality/outputs/PCMCI"

MODE = "district"                # "national" or "district"
DATE_COL = "Date"
DISTRICT_COL = "District"
SPLIT_COL = "split"
SPLITS_TO_USE = None              # e.g. ["train", "val", "purge", "test"] or None

TARGET_COL = "Log_NoOfDenguePatients"
PREDICTORS = ["AvgTemp_lag_3", "Rainfall_lag_2","denv4","denv1_lag_1"]
CONTROLS = ["denv4","denv1_lag_1"]                     # e.g. ["denv2", "denv3"]
EXTRA_VARS = []                   # any additional variables to include in PCMCI graph

TAU_MIN = 1
TAU_MAX = 6
PC_ALPHA = 0.05                   # PC stage alpha; can also be None
ALPHA_LEVEL = 0.05                # final significance threshold
FDR_METHOD = "none"              # e.g. "none", "fdr_bh"
VERBOSITY = 1

USE_GPDC_TORCH = False            # if installed and desired
AUTO_AGGREGATE_NATIONAL = True    # if national mode sees repeated dates, aggregate numeric cols by date
NATIONAL_AGG_METHOD = "mean"     # "mean", "median", or "sum"
SAVE_PDF = True
SAVE_PNG = True
FIG_DPI = 300


sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


# ---------------------------------------------------------------------
# Pretty names for tables / plots
# ---------------------------------------------------------------------
BASE_NAME_MAP = {
    # target / id
    "District": "District",
    "Date": "Date",
    "Month-year": "Month-Year",
    "Year": "Year",
    "Month": "Month",
    "Month_sin": "Month (Sin)",
    "Month_cos": "Month (Cos)",

    # dengue target
    "NoOfDenguePatients": "Dengue Cases",
    "Log_NoOfDenguePatients": "Log(Dengue Cases)",

    # climate
    "Rainfall": "Rainfall",
    "MinTemp": "Min. Temp",
    "AvgTemp": "Avg. Temp",
    "MaxTemp": "Max. Temp",
    "Humidity": "Humidity",
    "MonthlyAvgVisibility": "Visibility",
    "MonthlyAvgSeaLevelPressure": "Sea Level Pressure",
    "MonthlyAvgSunshineHours": "Sunshine Hrs",
    "MonthlyPrevailingWindSpeed": "Wind Speed",
    "MonthlyPrevailingWindDir": "Wind Direction",

    # static / categorical
    "PopulationDensity": "Pop. Density",
    "dominant": "Dominant Serotype",

    # serotypes
    "denv1": "DENV-1",
    "denv2": "DENV-2",
    "denv3": "DENV-3",
    "denv4": "DENV-4",
}

# optional explicit overrides for special dummy / engineered columns
column_renaming = {
    "MonthlyPrevailingWindDir_ENE": "Wind Dir. (ENE)",
}

LAGGED_NAME_PATTERN = re.compile(r"^(?P<base>.+?)_lag_(?P<lag>\d+)$")


def parse_input_offset(var_name: str) -> Tuple[str, int]:
    name = str(var_name).strip()
    m = LAGGED_NAME_PATTERN.match(name)
    if not m:
        return name, 0
    return m.group("base"), int(m.group("lag"))


def _format_base_name(col: str) -> str:
    """Return a pretty base name without lag annotation."""
    c = str(col).strip()

    if c in column_renaming:
        return column_renaming[c]

    if c in BASE_NAME_MAP:
        return BASE_NAME_MAP[c]

    # generic serotype pattern like denv1, denv2, ...
    m = re.fullmatch(r"denv(\d+)", c.lower())
    if m:
        return f"DENV-{m.group(1)}"

    # wind direction dummy
    if c.startswith("MonthlyPrevailingWindDir_"):
        suffix = c.replace("MonthlyPrevailingWindDir_", "").strip()
        return f"Wind Dir. ({suffix})"

    # fallback formatter
    c = c.replace("_", " ")
    c = re.sub(r"\s+", " ", c).strip()
    c = re.sub(r"([a-z])([A-Z])", r"\1 \2", c)
    c = c.replace("Avg ", "Avg. ")
    c = c.replace("Min ", "Min. ")
    c = c.replace("Max ", "Max. ")
    c = c.replace("Pop Density", "Pop. Density")
    return c


def pretty_column_name(col: str) -> str:
    """
    Convert raw column name to manuscript-friendly label.
    Handles already-lagged variables like AvgTemp_lag_3 or denv2_lag_1.
    """
    c = str(col).strip()

    # explicit override first
    if c in column_renaming:
        return column_renaming[c]

    # lagged variable pattern
    lag_match = re.match(r"^(.*)_lag_(\d+)$", c)
    if lag_match:
        base, lag_num = lag_match.groups()
        return f"{_format_base_name(base)} (Lag {lag_num})"

    return _format_base_name(c)


def make_var_display(var_name: str) -> str:
    # do not duplicate lag text
    return pretty_column_name(var_name)


def build_variable_lag_metadata(vars_: List[str], target_col: str) -> pd.DataFrame:
    rows = []
    target_base, target_offset = parse_input_offset(target_col)
    for var in vars_:
        base, offset = parse_input_offset(var)
        rows.append(
            {
                "variable": var,
                "base_variable": base,
                "input_offset_months": offset,
                "is_prelagged": bool(offset > 0),
                "target_variable": target_col,
                "target_base_variable": target_base,
                "target_input_offset_months": target_offset,
            }
        )
    return pd.DataFrame(rows)


def warn_if_prelagged(vars_: List[str], target_col: str, out_dir: Path) -> pd.DataFrame:
    meta = build_variable_lag_metadata(vars_, target_col)
    prelagged = meta[meta["is_prelagged"]].copy()
    meta.to_csv(out_dir / "variable_lag_metadata.csv", index=False)

    if prelagged.empty:
        note = (
            "No pre-lagged analysis variables detected. PCMCI lag output can be interpreted directly "
            "as the tested lag to the target series."
        )
    else:
        note_lines = [
            "Pre-lagged analysis variables detected.",
            "PCMCI will treat the provided columns exactly as supplied.",
            "Therefore, if a source variable is already shifted (for example AvgTemp_lag_3),",
            "the reported PCMCI lag is NOT the full temporal offset back to the original raw series.",
            "",
            "Effective total lag to target is computed as:",
            "    source_input_offset_months + pcmci_lag_months - target_input_offset_months",
            "",
            "Detected pre-lagged variables:",
        ]
        for _, row in prelagged.iterrows():
            note_lines.append(f" - {row['variable']} (base={row['base_variable']}, input_offset={row['input_offset_months']} months)")
        note = "\n".join(note_lines)
        warnings.warn(note)

    (out_dir / "lag_interpretation_note.txt").write_text(note, encoding="utf-8")
    return meta


def parse_args():
    parser = argparse.ArgumentParser(description="PCMCI nonlinear causal discovery for dengue time series.")
    parser.add_argument("--input", type=str, default=INPUT_PATH)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--mode", type=str, default=MODE, choices=["national", "district"])
    parser.add_argument("--date_col", type=str, default=DATE_COL)
    parser.add_argument("--district_col", type=str, default=DISTRICT_COL)
    parser.add_argument("--split_col", type=str, default=SPLIT_COL)
    parser.add_argument("--target_col", type=str, default=TARGET_COL)
    parser.add_argument("--predictors", nargs="+", default=PREDICTORS)
    parser.add_argument("--controls", nargs="*", default=CONTROLS)
    parser.add_argument("--extra_vars", nargs="*", default=EXTRA_VARS)
    parser.add_argument("--tau_min", type=int, default=TAU_MIN)
    parser.add_argument("--tau_max", type=int, default=TAU_MAX)
    parser.add_argument("--pc_alpha", type=float, default=PC_ALPHA)
    parser.add_argument("--alpha_level", type=float, default=ALPHA_LEVEL)
    parser.add_argument("--fdr_method", type=str, default=FDR_METHOD)
    parser.add_argument("--verbosity", type=int, default=VERBOSITY)
    parser.add_argument("--use_gpdc_torch", action="store_true", default=USE_GPDC_TORCH)
    parser.add_argument("--auto_aggregate_national", action="store_true", default=AUTO_AGGREGATE_NATIONAL)
    parser.add_argument("--national_agg_method", type=str, default=NATIONAL_AGG_METHOD, choices=["mean", "median", "sum"])
    parser.add_argument("--save_pdf", action="store_true", default=SAVE_PDF)
    parser.add_argument("--save_png", action="store_true", default=SAVE_PNG)
    return parser.parse_args()


def import_tigramite(use_gpdc_torch: bool = False):
    try:
        from tigramite import data_processing as pp
        from tigramite.pcmci import PCMCI
        from tigramite import plotting as tp
        if use_gpdc_torch:
            from tigramite.independence_tests.gpdc_torch import GPDCtorch
            return pp, PCMCI, tp, GPDCtorch
        else:
            from tigramite.independence_tests.gpdc import GPDC
            return pp, PCMCI, tp, GPDC
    except ImportError as e:
        missing = getattr(e, "name", None)
        if missing == "dcor":
            raise ImportError(
                "Missing dependency 'dcor', which is required by Tigramite GPDC. "
                "Install it with: pip install dcor"
            ) from e
        if missing in {"gpytorch", "torch"}:
            raise ImportError(
                "Missing dependency for GPDCtorch. Install it with: pip install gpytorch torch"
            ) from e
        raise ImportError(
            "Tigramite is required for this script. Install it first, for example: "
            "pip install tigramite"
        ) from e


def load_dataset(args) -> pd.DataFrame:
    df = pd.read_csv(args.input)
    df.columns = [c.strip() for c in df.columns]

    if args.date_col not in df.columns:
        raise ValueError(f"Date column '{args.date_col}' not found. Available columns: {df.columns.tolist()}")
    df[args.date_col] = pd.to_datetime(df[args.date_col], errors="raise")

    if args.split_col in df.columns and SPLITS_TO_USE:
        split_set = {str(x).strip().lower() for x in SPLITS_TO_USE}
        df[args.split_col] = df[args.split_col].astype(str).str.strip().str.lower()
        before = len(df)
        df = df[df[args.split_col].isin(split_set)].copy()
        print(f"Filtered by splits {sorted(split_set)}: {before} -> {len(df)} rows")

    if args.district_col in df.columns:
        df[args.district_col] = df[args.district_col].astype(str).str.strip()

    df = df.sort_values([c for c in [args.district_col if args.district_col in df.columns else None, args.date_col] if c]).reset_index(drop=True)
    return df


def required_variables(args) -> List[str]:
    vars_ = [args.target_col] + list(args.predictors) + list(args.controls) + list(args.extra_vars)
    seen = set()
    ordered = []
    for v in vars_:
        if v and v not in seen:
            ordered.append(v)
            seen.add(v)
    return ordered


def validate_columns(df: pd.DataFrame, vars_: List[str], args) -> None:
    missing = [v for v in vars_ if v not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required variable columns: {missing}.\nAvailable columns: {df.columns.tolist()}"
        )
    if args.mode == "district" and args.district_col not in df.columns:
        raise ValueError(f"District mode requires district column '{args.district_col}'.")


def aggregate_national(df: pd.DataFrame, vars_: List[str], args) -> pd.DataFrame:
    # If already one row per date, leave it alone.
    counts = df.groupby(args.date_col).size()
    if counts.max() == 1:
        return df[[args.date_col] + vars_].copy().sort_values(args.date_col).reset_index(drop=True)

    if not args.auto_aggregate_national:
        raise ValueError(
            "National mode detected multiple rows per date. Provide a pre-aggregated national file or enable auto aggregation."
        )

    numeric_vars = [v for v in vars_ if pd.api.types.is_numeric_dtype(df[v])]
    if len(numeric_vars) != len(vars_):
        non_num = [v for v in vars_ if v not in numeric_vars]
        raise ValueError(f"All analysis variables must be numeric for auto national aggregation. Non-numeric: {non_num}")

    warnings.warn(
        "Auto-aggregating multiple rows per date for national mode. "
        "For log1p-transformed targets, this may not equal a true national-count series."
    )

    agg_map = {v: args.national_agg_method for v in numeric_vars}
    out = df.groupby(args.date_col, as_index=False).agg(agg_map)
    return out.sort_values(args.date_col).reset_index(drop=True)


def make_tigramite_dataframe(df: pd.DataFrame, vars_: List[str], pp):
    array = df[vars_].astype(float).to_numpy()
    dataframe = pp.DataFrame(array, var_names=vars_)
    return dataframe


def run_pcmci_one(df_unit: pd.DataFrame, vars_: List[str], args, pp, PCMCI, tp, GPDCClass):
    dataframe = make_tigramite_dataframe(df_unit, vars_, pp)

    cond_test = GPDCClass()
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_test, verbosity=args.verbosity)
    results = pcmci.run_pcmci(
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        pc_alpha=args.pc_alpha,
        alpha_level=args.alpha_level,
        fdr_method=args.fdr_method,
    )
    return dataframe, pcmci, results


def long_results_from_pcmci(results: Dict, vars_: List[str], target_col: str, tau_min: int, tau_max: int, alpha_level: float) -> pd.DataFrame:
    p_matrix = results["p_matrix"]
    val_matrix = results["val_matrix"]
    graph = results.get("graph", None)

    target_base_global, target_input_offset_global = parse_input_offset(target_col)

    rows = []
    n = len(vars_)
    for i in range(n):
        source_base, source_input_offset = parse_input_offset(vars_[i])
        for j in range(n):
            target_base, target_input_offset = parse_input_offset(vars_[j])
            for tau in range(tau_min, tau_max + 1):
                pval = float(p_matrix[i, j, tau])
                val = float(val_matrix[i, j, tau])
                graph_entry = graph[i, j, tau] if graph is not None else ""
                effective_total_lag = source_input_offset + tau - target_input_offset
                rows.append(
                    {
                        "source": vars_[i],
                        "source_display": make_var_display(vars_[i]),
                        "source_base": source_base,
                        "source_input_offset_months": int(source_input_offset),
                        "target": vars_[j],
                        "target_display": make_var_display(vars_[j]),
                        "target_base": target_base,
                        "target_input_offset_months": int(target_input_offset),
                        "lag_months": int(tau),
                        "effective_total_lag_months": int(effective_total_lag),
                        "p_value": pval,
                        "val": val,
                        "significant": pval < alpha_level,
                        "graph": str(graph_entry),
                        "analysis_target_col": target_col,
                        "analysis_target_base": target_base_global,
                        "analysis_target_input_offset_months": int(target_input_offset_global),
                    }
                )
    return pd.DataFrame(rows)


def save_matrix_csvs(results: Dict, vars_: List[str], out_dir: Path, prefix: str):
    p_matrix = results["p_matrix"]
    val_matrix = results["val_matrix"]
    n = len(vars_)
    max_tau = p_matrix.shape[2] - 1

    for tau in range(1, max_tau + 1):
        p_df = pd.DataFrame(p_matrix[:, :, tau], index=vars_, columns=vars_)
        v_df = pd.DataFrame(val_matrix[:, :, tau], index=vars_, columns=vars_)
        p_df.to_csv(out_dir / f"{prefix}_p_matrix_lag{tau}.csv")
        v_df.to_csv(out_dir / f"{prefix}_val_matrix_lag{tau}.csv")


def strongest_links_to_target(long_df: pd.DataFrame, target_col: str, predictors: List[str], alpha_level: float, method_label: str = "PCMCI") -> pd.DataFrame:
    rows = []
    for pred in predictors:
        sub = long_df[(long_df["source"] == pred) & (long_df["target"] == target_col)].copy()
        pred_base, pred_input_offset = parse_input_offset(pred)
        target_base, target_input_offset = parse_input_offset(target_col)
        if sub.empty:
            rows.append(
                {
                    "Causal Hypothesis": f"{pretty_column_name(pred)} → {pretty_column_name(target_col)}",
                    "Method": method_label,
                    "Source Variable": pretty_column_name(pred),
                    "Source Base Variable": pretty_column_name(pred_base),
                    "Source Input Offset (Months)": pred_input_offset,
                    "Target Variable": pretty_column_name(target_col),
                    "Target Base Variable": pretty_column_name(target_base),
                    "Target Input Offset (Months)": target_input_offset,
                    "Strongest PCMCI Lag (Months)": np.nan,
                    "Effective Total Lag (Months)": np.nan,
                    "p-value": np.nan,
                    "Conclusion": "No result",
                    "Effect Value": np.nan,
                }
            )
            continue
        best = sub.sort_values(["p_value", "effective_total_lag_months", "lag_months"]).iloc[0]
        rows.append(
            {
                "Causal Hypothesis": f"{pretty_column_name(pred)} → {pretty_column_name(target_col)}",
                "Method": method_label,
                "Source Variable": pretty_column_name(pred),
                "Source Base Variable": pretty_column_name(pred_base),
                "Source Input Offset (Months)": pred_input_offset,
                "Target Variable": pretty_column_name(target_col),
                "Target Base Variable": pretty_column_name(target_base),
                "Target Input Offset (Months)": target_input_offset,
                "Strongest PCMCI Lag (Months)": int(best["lag_months"]),
                "Effective Total Lag (Months)": int(best["effective_total_lag_months"]),
                "p-value": float(best["p_value"]),
                "Conclusion": "Sig. Non-Linear" if float(best["p_value"]) < alpha_level else "Not Sig.",
                "Effect Value": float(best["val"]),
            }
        )
    return pd.DataFrame(rows)


def save_graph_figure(results: Dict, vars_: List[str], tp, out_base: Path, title: str):
    pretty_vars = [pretty_column_name(v) for v in vars_]
    try:
        tp.plot_graph(
            graph=results["graph"],
            val_matrix=results["val_matrix"],
            var_names=pretty_vars,
            save_name=str(out_base.with_suffix(".png")) if SAVE_PNG else None,
            figsize=(8, 6),
        )
        if SAVE_PDF:
            tp.plot_graph(
                graph=results["graph"],
                val_matrix=results["val_matrix"],
                var_names=pretty_vars,
                save_name=str(out_base.with_suffix(".pdf")),
                figsize=(8, 6),
            )
    except Exception as e:
        warnings.warn(f"Could not save Tigramite graph figure: {e}")


def _save_current_fig(path_base: Path):
    if SAVE_PNG:
        plt.savefig(path_base.with_suffix(".png"), dpi=FIG_DPI, bbox_inches="tight")
    if SAVE_PDF:
        plt.savefig(path_base.with_suffix(".pdf"), dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


def save_target_heatmaps(long_df: pd.DataFrame, target_col: str, predictors: List[str], out_dir: Path, prefix: str):
    sub = long_df[(long_df["target"] == target_col) & (long_df["source"].isin(predictors))].copy()
    if sub.empty:
        return

    sub["source_plot_label"] = sub["source_display"]

    p_pivot = sub.pivot(index="source_plot_label", columns="lag_months", values="p_value")
    v_pivot = sub.pivot(index="source_plot_label", columns="lag_months", values="val")
    nlogp = -np.log10(p_pivot.clip(lower=1e-300))

    plt.figure(figsize=(1.6 * max(4, p_pivot.shape[1]), 0.8 * max(3, p_pivot.shape[0]) + 2))
    sns.heatmap(nlogp, annot=p_pivot.round(4), fmt="", cmap="YlOrRd")
    plt.title(f"PCMCI p-values to {pretty_column_name(target_col)} (annotated, color = -log10 p)")
    plt.xlabel("PCMCI Lag (months)")
    plt.ylabel("Source Variable")
    _save_current_fig(out_dir / f"{prefix}_target_pvalue_heatmap")

    plt.figure(figsize=(1.6 * max(4, v_pivot.shape[1]), 0.8 * max(3, v_pivot.shape[0]) + 2))
    sns.heatmap(v_pivot, annot=True, fmt=".3f", cmap="RdBu_r", center=0)
    plt.title(f"PCMCI effect values to {pretty_column_name(target_col)}")
    plt.xlabel("PCMCI Lag (months)")
    plt.ylabel("Source Variable")
    _save_current_fig(out_dir / f"{prefix}_target_val_heatmap")


def save_summary_bar(summary_df: pd.DataFrame, out_dir: Path, prefix: str):
    use = summary_df.dropna(subset=["p-value"]).copy()
    if use.empty:
        return
    use["minus_log10_p"] = -np.log10(use["p-value"].clip(lower=1e-300))
    if "Effective Total Lag (Months)" in use.columns:
        use["Hypothesis Label"] = (
            use["Causal Hypothesis"].astype(str)
            + " | eff lag="
            + use["Effective Total Lag (Months)"].astype("Int64").astype(str)
        )
        y_col = "Hypothesis Label"
    else:
        y_col = "Causal Hypothesis"
    plt.figure(figsize=(10, max(4, 0.7 * len(use))))
    sns.barplot(data=use.sort_values("minus_log10_p", ascending=False), x="minus_log10_p", y=y_col)
    plt.axvline(-np.log10(ALPHA_LEVEL), linestyle="--", color="red", label=f"alpha={ALPHA_LEVEL}")
    plt.xlabel("-log10(p-value)")
    plt.ylabel("Hypothesis")
    plt.title("Strongest PCMCI evidence by hypothesis")
    plt.legend(loc="best")
    _save_current_fig(out_dir / f"{prefix}_summary_bar")


def fisher_combine_by_predictor(district_link_tables: Dict[str, pd.DataFrame], target_col: str, predictors: List[str], tau_min: int, tau_max: int, alpha_level: float) -> pd.DataFrame:
    rows = []
    target_base, target_input_offset = parse_input_offset(target_col)
    for pred in predictors:
        pred_base, pred_input_offset = parse_input_offset(pred)
        best_row = None
        for tau in range(tau_min, tau_max + 1):
            pvals = []
            vals = []
            used_districts = []
            for district, long_df in district_link_tables.items():
                sub = long_df[
                    (long_df["source"] == pred)
                    & (long_df["target"] == target_col)
                    & (long_df["lag_months"] == tau)
                ]
                if not sub.empty and pd.notna(sub.iloc[0]["p_value"]):
                    pvals.append(float(sub.iloc[0]["p_value"]))
                    vals.append(float(sub.iloc[0]["val"]))
                    used_districts.append(district)
            if not pvals:
                continue
            fisher_stat, fisher_p = combine_pvalues(pvals, method="fisher")
            eff_lag = pred_input_offset + tau - target_input_offset
            row = {
                "Causal Hypothesis": f"{pretty_column_name(pred)} → {pretty_column_name(target_col)}",
                "Method": "PCMCI",
                "Source Variable": pretty_column_name(pred),
                "Source Base Variable": pretty_column_name(pred_base),
                "Source Input Offset (Months)": pred_input_offset,
                "Target Variable": pretty_column_name(target_col),
                "Target Base Variable": pretty_column_name(target_base),
                "Target Input Offset (Months)": target_input_offset,
                "Strongest PCMCI Lag (Months)": tau,
                "Effective Total Lag (Months)": eff_lag,
                "p-value": float(fisher_p),
                "Fisher Statistic": float(fisher_stat),
                "Mean Effect Value": float(np.mean(vals)) if vals else np.nan,
                "Districts Used": len(used_districts),
                "Conclusion": "Sig. Non-Linear" if float(fisher_p) < alpha_level else "Not Sig.",
            }
            if best_row is None or row["p-value"] < best_row["p-value"]:
                best_row = row
        if best_row is None:
            best_row = {
                "Causal Hypothesis": f"{pretty_column_name(pred)} → {pretty_column_name(target_col)}",
                "Method": "PCMCI",
                "Source Variable": pretty_column_name(pred),
                "Source Base Variable": pretty_column_name(pred_base),
                "Source Input Offset (Months)": pred_input_offset,
                "Target Variable": pretty_column_name(target_col),
                "Target Base Variable": pretty_column_name(target_base),
                "Target Input Offset (Months)": target_input_offset,
                "Strongest PCMCI Lag (Months)": np.nan,
                "Effective Total Lag (Months)": np.nan,
                "p-value": np.nan,
                "Fisher Statistic": np.nan,
                "Mean Effect Value": np.nan,
                "Districts Used": 0,
                "Conclusion": "No result",
            }
        rows.append(best_row)
    return pd.DataFrame(rows)


def save_district_strength_heatmap(district_best_df: pd.DataFrame, out_dir: Path, target_col: str):
    if district_best_df.empty:
        return

    plot_df = district_best_df.copy()
    pretty_target = pretty_column_name(target_col)

    plot_df["Hypothesis"] = plot_df["source"].astype(str) + " → " + plot_df["target"].astype(str)
    plot_df = plot_df[plot_df["target"] == pretty_target].copy()
    if plot_df.empty:
        return

    plot_df["minus_log10_p"] = -np.log10(plot_df["p_value"].clip(lower=1e-300))
    piv = plot_df.pivot(index="district", columns="Hypothesis", values="minus_log10_p")
    annot = plot_df.pivot(index="district", columns="Hypothesis", values="Strongest PCMCI Lag (Months)")

    plt.figure(figsize=(1.1 * max(4, piv.shape[1]) + 2, 0.5 * max(6, piv.shape[0]) + 2))
    sns.heatmap(piv, annot=annot, fmt=".0f", cmap="YlOrRd")
    plt.title("District-wise strongest PCMCI evidence (color = -log10 p, annotation = lag)")
    plt.xlabel("Hypothesis")
    plt.ylabel("District")
    _save_current_fig(out_dir / "district_strongest_heatmap")


def run_national(df: pd.DataFrame, vars_: List[str], args, pp, PCMCI, tp, GPDCClass, out_dir: Path):
    df_nat = aggregate_national(df, vars_, args)
    dataframe, pcmci, results = run_pcmci_one(df_nat, vars_, args, pp, PCMCI, tp, GPDCClass)

    long_df = long_results_from_pcmci(results, vars_, args.target_col, args.tau_min, args.tau_max, args.alpha_level)
    long_df.to_csv(out_dir / "pcmci_links_long.csv", index=False)
    save_matrix_csvs(results, vars_, out_dir, prefix="national")

    summary_df = strongest_links_to_target(long_df, args.target_col, args.predictors, args.alpha_level)
    summary_df.to_csv(out_dir / "pcmci_manuscript_summary.csv", index=False)

    save_graph_figure(results, vars_, tp, out_dir / "pcmci_graph", title="PCMCI causal graph")
    save_target_heatmaps(long_df, args.target_col, args.predictors, out_dir, prefix="national")
    save_summary_bar(summary_df, out_dir, prefix="national")

    meta = {
        "mode": "national",
        "n_rows": len(df_nat),
        "date_min": str(df_nat[args.date_col].min()),
        "date_max": str(df_nat[args.date_col].max()),
        "variables": vars_,
        "predictors": args.predictors,
        "controls": args.controls,
        "prelagged_predictors": [p for p in args.predictors if parse_input_offset(p)[1] > 0],
        "target_col": args.target_col,
        "tau_min": args.tau_min,
        "tau_max": args.tau_max,
        "pc_alpha": args.pc_alpha,
        "alpha_level": args.alpha_level,
        "fdr_method": args.fdr_method,
    }
    save_json(meta, out_dir / "pcmci_run_metadata.json")
    return summary_df


def run_district(df: pd.DataFrame, vars_: List[str], args, pp, PCMCI, tp, GPDCClass, out_dir: Path):
    district_tables = {}
    district_best_rows = []

    for district, g in df.groupby(args.district_col):
        g = g.sort_values(args.date_col).reset_index(drop=True)
        if len(g) <= args.tau_max + 5:
            warnings.warn(f"Skipping district {district}: not enough rows for tau_max={args.tau_max}.")
            continue

        district_dir = out_dir / "districts" / str(district)
        ensure_dir(district_dir)

        dataframe, pcmci, results = run_pcmci_one(g[[args.date_col] + vars_].copy(), vars_, args, pp, PCMCI, tp, GPDCClass)
        long_df = long_results_from_pcmci(results, vars_, args.target_col, args.tau_min, args.tau_max, args.alpha_level)
        long_df.insert(0, "district", district)
        long_df.to_csv(district_dir / f"{district}_pcmci_links_long.csv", index=False)
        save_matrix_csvs(results, vars_, district_dir, prefix=str(district))

        summary_df = strongest_links_to_target(long_df, args.target_col, args.predictors, args.alpha_level)
        summary_df.insert(0, "district", district)
        summary_df.to_csv(district_dir / f"{district}_pcmci_manuscript_summary.csv", index=False)

        save_graph_figure(results, vars_, tp, district_dir / f"{district}_pcmci_graph", title=f"PCMCI graph - {district}")
        save_target_heatmaps(long_df, args.target_col, args.predictors, district_dir, prefix=f"{district}")
        save_summary_bar(summary_df, district_dir, prefix=f"{district}")

        district_tables[district] = long_df.copy()
        for pred in args.predictors:
            sub = long_df[(long_df["source"] == pred) & (long_df["target"] == args.target_col)].copy()
            if sub.empty:
                continue
            best = sub.sort_values(["p_value", "lag_months"]).iloc[0]
            district_best_rows.append(
                {
                    "district": district,
                    "source": pretty_column_name(pred),
                    "source_base": pretty_column_name(parse_input_offset(pred)[0]),
                    "source_input_offset_months": parse_input_offset(pred)[1],
                    "target": pretty_column_name(args.target_col),
                    "target_base": pretty_column_name(parse_input_offset(args.target_col)[0]),
                    "target_input_offset_months": parse_input_offset(args.target_col)[1],
                    "Strongest PCMCI Lag (Months)": int(best["lag_months"]),
                    "Effective Total Lag (Months)": int(best["effective_total_lag_months"]),
                    "p_value": float(best["p_value"]),
                    "val": float(best["val"]),
                    "Conclusion": "Sig. Non-Linear" if float(best["p_value"]) < args.alpha_level else "Not Sig.",
                }
            )

    if not district_tables:
        raise ValueError("No district-level PCMCI results were generated.")

    all_long = pd.concat(district_tables.values(), ignore_index=True)
    all_long.to_csv(out_dir / "pcmci_links_long_all_districts.csv", index=False)

    district_best_df = pd.DataFrame(district_best_rows)
    district_best_df.to_csv(out_dir / "pcmci_strongest_links_by_district.csv", index=False)
    save_district_strength_heatmap(district_best_df, out_dir, args.target_col)

    fisher_df = fisher_combine_by_predictor(district_tables, args.target_col, args.predictors, args.tau_min, args.tau_max, args.alpha_level)
    fisher_df.to_csv(out_dir / "pcmci_manuscript_summary_fisher_combined.csv", index=False)
    save_summary_bar(fisher_df, out_dir, prefix="district_fisher_combined")

    meta = {
        "mode": "district",
        "n_districts": len(district_tables),
        "variables": vars_,
        "predictors": args.predictors,
        "controls": args.controls,
        "prelagged_predictors": [p for p in args.predictors if parse_input_offset(p)[1] > 0],
        "target_col": args.target_col,
        "tau_min": args.tau_min,
        "tau_max": args.tau_max,
        "pc_alpha": args.pc_alpha,
        "alpha_level": args.alpha_level,
        "fdr_method": args.fdr_method,
        "districts": sorted(district_tables.keys()),
    }
    save_json(meta, out_dir / "pcmci_run_metadata.json")
    return fisher_df


def write_readme_stub(out_dir: Path, args, vars_: List[str]):
    txt = [
        "PCMCI run outputs",
        "=" * 80,
        f"Input file: {args.input}",
        f"Mode: {args.mode}",
        f"Target: {args.target_col}",
        f"Predictors: {args.predictors}",
        f"Controls: {args.controls}",
        f"Variables included in PCMCI graph: {vars_}",
        f"tau_min={args.tau_min}, tau_max={args.tau_max}",
        f"pc_alpha={args.pc_alpha}, alpha_level={args.alpha_level}, fdr_method={args.fdr_method}",
        "",
        "Main outputs:",
        "- pcmci_manuscript_summary.csv or pcmci_manuscript_summary_fisher_combined.csv",
        "- pcmci_links_long.csv or pcmci_links_long_all_districts.csv",
        "- variable_lag_metadata.csv",
        "- lag_interpretation_note.txt",
        "- graph figures",
        "- target p-value and effect-value heatmaps",
        "",
        "Interpretation note:",
        "- If a predictor is already lagged in the input file (for example AvgTemp_lag_3),",
        "  then PCMCI lag is not the full temporal offset back to the original raw series.",
        "- See 'Effective Total Lag (Months)' in the manuscript summary outputs.",
    ]
    (out_dir / "run_summary.txt").write_text("\n".join(txt), encoding="utf-8")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    print(f"Input file : {args.input}")
    print(f"Output dir : {out_dir}")

    pp, PCMCI, tp, GPDCClass = import_tigramite(args.use_gpdc_torch)
    df = load_dataset(args)
    vars_ = required_variables(args)
    validate_columns(df, vars_, args)
    write_readme_stub(out_dir, args, vars_)
    lag_meta = warn_if_prelagged(vars_, args.target_col, out_dir)

    if args.mode == "national":
        summary_df = run_national(df, vars_, args, pp, PCMCI, tp, GPDCClass, out_dir)
    else:
        summary_df = run_district(df, vars_, args, pp, PCMCI, tp, GPDCClass, out_dir)

    print("\nPCMCI run complete.")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()