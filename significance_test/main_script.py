#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final significance testing for dengue forecasting models.

Protocol finalized
------------------
Reference model:
    Prophet

Competitor models:
    Attention-LSTM, Stacked LSTM, XGBoost, TFT

Loss families:
    1) squared error  -> RMSE-oriented significance analysis
    2) absolute error -> MAE-oriented significance analysis

Testing unit:
    - align common District-TargetDate-horizon rows pairwise
    - average losses across districts within each TargetDate
    - run the forecast comparison test on the TargetDate-level mean loss series

Statistical test:
    Diebold-Mariano test with Harvey-Leybourne-Newbold (HLN) small-sample correction
    using Newey-West lag = horizon - 1

Multiplicity control:
    Holm correction within each horizon and loss family across
    the four Prophet-vs-competitor comparisons.

Input requirements
------------------
Each file must contain:
    District, TargetDate, horizon, y_true_count
and either:
    y_pred_count
or:
    y_pred_count_p50   (used for TFT)

Optional columns are ignored.

Outputs
-------
coverage_audit_by_model_horizon.csv
pairwise_overlap_vs_prophet.csv
all_models_common_intersection_audit.csv
pairwise_targetdate_mean_losses.csv
dm_hln_squared_error_results.csv
dm_hln_absolute_error_results.csv
table_dm_squared_error_summary.csv
table_dm_absolute_error_summary.csv
significance_testing_results.xlsx
"""

from __future__ import annotations

import argparse
import math
import zipfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


DEFAULT_FILES = {
    "Prophet": "./Prophet/outputs/prophet_test_predictions_long.csv",
    "Attention-LSTM": "./attn-LSTM/outputs/attention_lstm_test_predictions_long.csv",
    "Stacked LSTM": "./stacked-LSTM/outputs/stacked_lstm_test_predictions_long.csv",
    "XGBoost": "./XGBoost/outputs/xgb_test_predictions_long.csv",
    "TFT": "./TFT/outputs/tft_test_predictions_long.csv",
}
REFERENCE_MODEL = "Prophet"
COMPARATORS = ["Attention-LSTM", "Stacked LSTM", "XGBoost", "TFT"]


def parse_args():
    p = argparse.ArgumentParser(description="Final DM+HLN significance testing on count-scale forecast errors.")
    p.add_argument("--prophet", default=DEFAULT_FILES["Prophet"])
    p.add_argument("--attention", default=DEFAULT_FILES["Attention-LSTM"])
    p.add_argument("--stacked_lstm", default=DEFAULT_FILES["Stacked LSTM"])
    p.add_argument("--xgb", default=DEFAULT_FILES["XGBoost"])
    p.add_argument("--tft", default=DEFAULT_FILES["TFT"])
    p.add_argument("--outdir", default="./significance_test/outputs")
    return p.parse_args()


def resolve_input_path(path_like: str) -> Path:
    p = Path(path_like)
    if p.exists():
        return p.resolve()
    cwd_p = (Path.cwd() / p.name)
    if cwd_p.exists():
        return cwd_p.resolve()
    script_p = (Path(__file__).resolve().parent / p.name)
    if script_p.exists():
        return script_p.resolve()
    return p


def holm_adjust(pvals: pd.Series) -> pd.Series:
    """Holm step-down adjusted p-values, returned in original order."""
    pvals = pvals.astype(float)
    out = pd.Series(np.nan, index=pvals.index, dtype=float)
    valid = pvals.notna()
    if valid.sum() == 0:
        return out
    vals = pvals[valid].values
    idx = pvals[valid].index.to_list()
    order = np.argsort(vals)
    sorted_vals = vals[order]
    m = len(sorted_vals)
    adj_sorted = np.empty(m, dtype=float)
    running = 0.0
    for i, p in enumerate(sorted_vals):
        adj_i = (m - i) * p
        running = max(running, adj_i)
        adj_sorted[i] = min(running, 1.0)
    for pos, ord_idx in enumerate(order):
        out.loc[idx[ord_idx]] = adj_sorted[pos]
    return out


def dm_hln_test(d: np.ndarray, h: int) -> Dict[str, float]:
    """
    Diebold-Mariano with HLN small-sample correction.

    d_t = loss_competitor - loss_reference
    Positive mean(d) => reference model has lower loss (better).
    """
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    T = len(d)

    out = {
        "n_timepoints": int(T),
        "mean_loss_diff": np.nan,
        "dm_stat": np.nan,
        "p_value": np.nan,
        "long_run_variance": np.nan,
    }
    if T < 3:
        return out

    mean_d = float(np.mean(d))
    out["mean_loss_diff"] = mean_d

    lag = max(int(h) - 1, 0)
    centered = d - mean_d

    gamma0 = float(np.dot(centered, centered) / T)
    lrv = gamma0
    for k in range(1, lag + 1):
        if k >= T:
            break
        gamma_k = float(np.dot(centered[k:], centered[:-k]) / T)
        lrv += 2.0 * gamma_k

    out["long_run_variance"] = float(lrv)
    if not np.isfinite(lrv) or lrv <= 0:
        return out

    dm = mean_d / math.sqrt(lrv / T)

    h_eff = min(max(int(h), 1), T - 1)
    correction_term = (T + 1 - 2 * h_eff + (h_eff * (h_eff - 1)) / T) / T
    if correction_term <= 0:
        return out

    dm_hln = dm * math.sqrt(correction_term)
    p_value = float(2.0 * student_t.sf(abs(dm_hln), df=T - 1))
    out["dm_stat"] = float(dm_hln)
    out["p_value"] = p_value
    return out


def load_and_normalize(path: Path, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = ["District", "TargetDate", "horizon", "y_true_count"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"{model_name}: missing required column '{col}'")

    if "y_pred_count" not in df.columns:
        if "y_pred_count_p50" in df.columns:
            df["y_pred_count"] = df["y_pred_count_p50"]
        else:
            raise ValueError(f"{model_name}: missing 'y_pred_count' and 'y_pred_count_p50'")

    df["District"] = df["District"].astype(str).str.strip()
    df["TargetDate"] = pd.to_datetime(df["TargetDate"], errors="raise")
    df["horizon"] = pd.to_numeric(df["horizon"], errors="raise").astype(int)
    df["y_true_count"] = pd.to_numeric(df["y_true_count"], errors="coerce")
    df["y_pred_count"] = pd.to_numeric(df["y_pred_count"], errors="coerce")
    df = df.dropna(subset=["District", "TargetDate", "horizon", "y_true_count", "y_pred_count"]).copy()

    if "split" in df.columns:
        test_mask = df["split"].astype(str).str.lower().eq("test")
        if test_mask.any():
            df = df.loc[test_mask].copy()

    # keep only needed columns
    df["Model"] = model_name
    key_cols = ["District", "TargetDate", "horizon"]
    if df.duplicated(subset=key_cols).any():
        dup_n = int(df.duplicated(subset=key_cols).sum())
        raise ValueError(f"{model_name}: duplicate District-TargetDate-horizon rows found ({dup_n})")

    return df[key_cols + ["y_true_count", "y_pred_count", "Model"]].copy()


def build_coverage_audits(dfs: Dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_model_rows = []
    for model, df in dfs.items():
        for h, part in df.groupby("horizon"):
            by_model_rows.append({
                "Model": model,
                "horizon": int(h),
                "rows": int(len(part)),
                "target_dates": int(part["TargetDate"].nunique()),
                "districts": int(part["District"].nunique()),
                "target_date_start": part["TargetDate"].min(),
                "target_date_end": part["TargetDate"].max(),
            })
    coverage_df = pd.DataFrame(by_model_rows).sort_values(["Model", "horizon"]).reset_index(drop=True)

    pairwise_rows = []
    ref = dfs[REFERENCE_MODEL]
    for comp in COMPARATORS:
        cmp_df = dfs[comp]
        for h in sorted(set(ref["horizon"]).intersection(set(cmp_df["horizon"]))):
            ref_h = ref.loc[ref["horizon"] == h, ["District", "TargetDate", "horizon"]]
            cmp_h = cmp_df.loc[cmp_df["horizon"] == h, ["District", "TargetDate", "horizon"]]
            common = ref_h.merge(cmp_h, on=["District", "TargetDate", "horizon"], how="inner")
            pairwise_rows.append({
                "ReferenceModel": REFERENCE_MODEL,
                "ComparatorModel": comp,
                "horizon": int(h),
                "ref_rows": int(len(ref_h)),
                "comp_rows": int(len(cmp_h)),
                "common_rows": int(len(common)),
                "ref_target_dates": int(ref_h["TargetDate"].nunique()),
                "comp_target_dates": int(cmp_h["TargetDate"].nunique()),
                "common_target_dates": int(common["TargetDate"].nunique()),
            })
    pairwise_df = pd.DataFrame(pairwise_rows).sort_values(["ComparatorModel", "horizon"]).reset_index(drop=True)

    # all-model common intersection audit
    all_sets = {
        model: set(map(tuple, df[["District", "TargetDate", "horizon"]].itertuples(index=False, name=None)))
        for model, df in dfs.items()
    }
    common_all = set.intersection(*all_sets.values())
    all_rows = []
    for model, s in all_sets.items():
        extra = s - common_all
        all_rows.append({
            "Model": model,
            "rows_total": len(s),
            "rows_in_all_model_intersection": len(common_all),
            "rows_outside_all_model_intersection": len(extra),
        })
    all_common_df = pd.DataFrame(all_rows).sort_values("Model").reset_index(drop=True)
    return coverage_df, pairwise_df, all_common_df


def pairwise_targetdate_losses(ref_df: pd.DataFrame, cmp_df: pd.DataFrame, h: int) -> pd.DataFrame:
    ref = ref_df.loc[ref_df["horizon"] == h].copy()
    cmp_ = cmp_df.loc[cmp_df["horizon"] == h].copy()

    merged = ref.merge(
        cmp_,
        on=["District", "TargetDate", "horizon"],
        how="inner",
        suffixes=("_ref", "_cmp"),
        validate="one_to_one",
    )
    if merged.empty:
        return pd.DataFrame()

    truth_diff = np.abs(merged["y_true_count_ref"] - merged["y_true_count_cmp"])
    if truth_diff.max() > 1e-6:
        raise ValueError(
            f"Truth mismatch for {REFERENCE_MODEL} vs comparator at horizon {h}. "
            f"max |y_true_ref - y_true_cmp| = {truth_diff.max()}"
        )

    merged["se_ref"] = (merged["y_pred_count_ref"] - merged["y_true_count_ref"]) ** 2
    merged["se_cmp"] = (merged["y_pred_count_cmp"] - merged["y_true_count_ref"]) ** 2
    merged["ae_ref"] = np.abs(merged["y_pred_count_ref"] - merged["y_true_count_ref"])
    merged["ae_cmp"] = np.abs(merged["y_pred_count_cmp"] - merged["y_true_count_ref"])

    td = (
        merged.groupby("TargetDate", as_index=False)
        .agg(
            n_districts=("District", "nunique"),
            se_ref=("se_ref", "mean"),
            se_cmp=("se_cmp", "mean"),
            ae_ref=("ae_ref", "mean"),
            ae_cmp=("ae_cmp", "mean"),
        )
        .sort_values("TargetDate")
        .reset_index(drop=True)
    )
    td["sq_diff"] = td["se_cmp"] - td["se_ref"]
    td["abs_diff"] = td["ae_cmp"] - td["ae_ref"]
    return td


def compute_results(dfs: Dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mean_loss_rows = []
    sq_rows = []
    abs_rows = []

    for comp in COMPARATORS:
        for h in range(1, 7):
            td = pairwise_targetdate_losses(dfs[REFERENCE_MODEL], dfs[comp], h)
            if td.empty:
                sq = {"n_timepoints": 0, "mean_loss_diff": np.nan, "dm_stat": np.nan, "p_value": np.nan, "long_run_variance": np.nan}
                ab = {"n_timepoints": 0, "mean_loss_diff": np.nan, "dm_stat": np.nan, "p_value": np.nan, "long_run_variance": np.nan}
            else:
                tmp = td.copy()
                tmp.insert(0, "ComparatorModel", comp)
                tmp.insert(0, "ReferenceModel", REFERENCE_MODEL)
                tmp.insert(0, "horizon", int(h))
                mean_loss_rows.append(tmp)
                sq = dm_hln_test(td["sq_diff"].to_numpy(), h=h)
                ab = dm_hln_test(td["abs_diff"].to_numpy(), h=h)

            sq_rows.append({
                "loss_family": "squared_error",
                "ReferenceModel": REFERENCE_MODEL,
                "ComparatorModel": comp,
                "horizon": int(h),
                **sq,
            })
            abs_rows.append({
                "loss_family": "absolute_error",
                "ReferenceModel": REFERENCE_MODEL,
                "ComparatorModel": comp,
                "horizon": int(h),
                **ab,
            })

    mean_loss_df = pd.concat(mean_loss_rows, ignore_index=True) if mean_loss_rows else pd.DataFrame()
    sq_df = pd.DataFrame(sq_rows)
    abs_df = pd.DataFrame(abs_rows)

    # Holm adjust within each horizon and loss family
    def add_holm(df: pd.DataFrame) -> pd.DataFrame:
        out = []
        for h, grp in df.groupby("horizon", sort=True):
            grp = grp.copy()
            grp["holm_p_value"] = holm_adjust(grp["p_value"]).values
            grp["significant_0_05"] = grp["holm_p_value"] < 0.05
            out.append(grp)
        return pd.concat(out, ignore_index=True).sort_values(["horizon", "ComparatorModel"]).reset_index(drop=True)

    sq_df = add_holm(sq_df)
    abs_df = add_holm(abs_df)
    return mean_loss_df, sq_df, abs_df


def build_summary(df: pd.DataFrame, loss_family: str) -> pd.DataFrame:
    out = df[[
        "horizon",
        "ComparatorModel",
        "n_timepoints",
        "mean_loss_diff",
        "dm_stat",
        "p_value",
        "holm_p_value",
        "significant_0_05",
    ]].copy()
    out.insert(0, "loss_family", loss_family)
    return out


def write_readme(outdir: Path):
    text = """Final significance testing outputs

Protocol used
-------------
Reference model: Prophet
Competitors: Attention-LSTM, Stacked LSTM, XGBoost, TFT

Two significance families:
1. squared error  -> RMSE-oriented
2. absolute error -> MAE-oriented

Testing unit:
- common District-TargetDate-horizon rows, aligned pairwise
- losses averaged across districts within each TargetDate
- DM test with HLN small-sample correction
- Newey-West lag = horizon - 1
- Holm correction within each horizon and loss family

Important note
--------------
If any model has fewer available rows than the others, this script keeps the
comparison valid by using only the pairwise common overlap for that comparison.
See:
- coverage_audit_by_model_horizon.csv
- pairwise_overlap_vs_prophet.csv
- all_models_common_intersection_audit.csv
"""
    (outdir / "README.txt").write_text(text, encoding="utf-8")


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    input_map = {
        "Prophet": resolve_input_path(args.prophet),
        "Attention-LSTM": resolve_input_path(args.attention),
        "Stacked LSTM": resolve_input_path(args.stacked_lstm),
        "XGBoost": resolve_input_path(args.xgb),
        "TFT": resolve_input_path(args.tft),
    }

    dfs = {}
    for model, path in input_map.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing file for {model}: {path}")
        dfs[model] = load_and_normalize(path, model)

    coverage_df, pairwise_df, all_common_df = build_coverage_audits(dfs)
    mean_loss_df, sq_df, abs_df = compute_results(dfs)
    sq_summary = build_summary(sq_df, "squared_error")
    abs_summary = build_summary(abs_df, "absolute_error")

    coverage_df.to_csv(outdir / "coverage_audit_by_model_horizon.csv", index=False)
    pairwise_df.to_csv(outdir / "pairwise_overlap_vs_prophet.csv", index=False)
    all_common_df.to_csv(outdir / "all_models_common_intersection_audit.csv", index=False)
    mean_loss_df.to_csv(outdir / "pairwise_targetdate_mean_losses.csv", index=False)
    sq_df.to_csv(outdir / "dm_hln_squared_error_results.csv", index=False)
    abs_df.to_csv(outdir / "dm_hln_absolute_error_results.csv", index=False)
    sq_summary.to_csv(outdir / "table_dm_squared_error_summary.csv", index=False)
    abs_summary.to_csv(outdir / "table_dm_absolute_error_summary.csv", index=False)
    write_readme(outdir)

    xlsx_path = outdir / "significance_testing_results.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        coverage_df.to_excel(writer, sheet_name="coverage_audit", index=False)
        pairwise_df.to_excel(writer, sheet_name="overlap_vs_prophet", index=False)
        all_common_df.to_excel(writer, sheet_name="all_common_audit", index=False)
        sq_summary.to_excel(writer, sheet_name="dm_squared_summary", index=False)
        abs_summary.to_excel(writer, sheet_name="dm_absolute_summary", index=False)
        sq_df.to_excel(writer, sheet_name="dm_squared_full", index=False)
        abs_df.to_excel(writer, sheet_name="dm_absolute_full", index=False)
        if not mean_loss_df.empty:
            mean_loss_df.to_excel(writer, sheet_name="targetdate_mean_losses", index=False)

    # zip outputs
    zip_path = outdir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(outdir.rglob("*")):
            if fp.is_file():
                zf.write(fp, arcname=fp.relative_to(outdir))

    print(f"Saved significance testing outputs to: {outdir.resolve()}")
    print(f"Saved zip archive to: {zip_path.resolve()}")
    print("\nAll-model common intersection audit:")
    print(all_common_df.to_string(index=False))
    print("\nSquared-error DM summary:")
    print(sq_summary.to_string(index=False))
    print("\nAbsolute-error DM summary:")
    print(abs_summary.to_string(index=False))


if __name__ == "__main__":
    main()