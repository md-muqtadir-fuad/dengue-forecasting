#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SARIMAX sensitivity runner.

Purpose:
- Keep SARIMAX inside the same paper-level sensitivity framework
- Use SARIMAX-appropriate specification sensitivity instead of fake feature-group ablations
- Reuse the working main SARIMAX script as the execution engine

Sensitivity experiments:
- full               : baseline config from JSON
- fixed_order_only   : same baseline orders, but disables order search
- no_seasonal        : removes seasonal component and disables order search
- simpler_nonseasonal: simpler non-seasonal ARIMA, no seasonal component, no order search

This runner launches the main SARIMAX script once per experiment, then aggregates
its outputs into sensitivity summary tables.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


@dataclass
class RunnerConfig:
    base_script: str
    fixed_config_json: str
    output_dir: str
    input_path: str | None = None
    target_col: str | None = None
    horizon: int | None = None
    random_state: int | None = None
    min_train_points: int | None = None
    maxiter: int | None = None
    diagnostics_lags: int | None = None
    adf_alpha: float | None = None
    ablation_names: str = "full,fixed_order_only,no_seasonal,simpler_nonseasonal"
    python_exe: str = sys.executable


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def resolve_baseline(cfg: RunnerConfig) -> Dict:
    base = load_json(cfg.fixed_config_json)
    resolved = dict(base)
    # CLI overrides if provided
    for key, value in {
        "input_path": cfg.input_path,
        "target_col": cfg.target_col,
        "horizon": cfg.horizon,
        "random_state": cfg.random_state,
        "min_train_points": cfg.min_train_points,
        "maxiter": cfg.maxiter,
        "diagnostics_lags": cfg.diagnostics_lags,
        "adf_alpha": cfg.adf_alpha,
    }.items():
        if value is not None:
            resolved[key] = value

    required = [
        "input_path", "target_col", "horizon", "random_state", "order",
        "seasonal_order", "min_train_points", "maxiter", "diagnostics_lags",
        "use_order_search", "adf_alpha"
    ]
    missing = [k for k in required if k not in resolved]
    if missing:
        raise ValueError(f"Missing baseline config keys: {missing}")
    return resolved


def build_experiments(base: Dict, names: List[str]) -> List[Dict]:
    p, d, q = base["order"]
    P, D, Q, s = base["seasonal_order"]

    # Use base.d and base.D so we preserve differencing assumptions from the main run.
    experiments = {
        "full": {
            "name": "full",
            "description": "Baseline SARIMAX specification from the main run config.",
            "order": [p, d, q],
            "seasonal_order": [P, D, Q, s],
            "use_order_search": bool(base["use_order_search"]),
        },
        "fixed_order_only": {
            "name": "fixed_order_only",
            "description": "Same baseline order and seasonal order, but no order search.",
            "order": [p, d, q],
            "seasonal_order": [P, D, Q, s],
            "use_order_search": False,
        },
        "no_seasonal": {
            "name": "no_seasonal",
            "description": "Removes the seasonal SARIMA component and disables order search.",
            "order": [p, d, q],
            "seasonal_order": [0, 0, 0, 0],
            "use_order_search": False,
        },
        "simpler_nonseasonal": {
            "name": "simpler_nonseasonal",
            "description": "Uses a simpler non-seasonal ARIMA specification with no seasonal component.",
            "order": [1, d, 0],
            "seasonal_order": [0, 0, 0, 0],
            "use_order_search": False,
        },
    }

    out = []
    for n in names:
        if n not in experiments:
            raise ValueError(
                f"Unknown SARIMAX sensitivity experiment: {n}. "
                f"Allowed: {list(experiments.keys())}"
            )
        exp = dict(base)
        exp.update(experiments[n])
        out.append(exp)
    return out


def build_subprocess_command(cfg: RunnerConfig, exp: Dict, exp_out_dir: Path) -> List[str]:
    p, d, q = exp["order"]
    P, D, Q, s = exp["seasonal_order"]
    cmd = [
        cfg.python_exe,
        cfg.base_script,
        "--input", str(exp["input_path"]),
        "--output_dir", str(exp_out_dir),
        "--target_col", str(exp["target_col"]),
        "--horizon", str(int(exp["horizon"])),
        "--random_state", str(int(exp["random_state"])),
        "--p", str(int(p)),
        "--d", str(int(d)),
        "--q", str(int(q)),
        "--P", str(int(P)),
        "--D", str(int(D)),
        "--Q", str(int(Q)),
        "--seasonal_period", str(int(s)),
        "--min_train_points", str(int(exp["min_train_points"])),
        "--maxiter", str(int(exp["maxiter"])),
        "--diagnostics_lags", str(int(exp["diagnostics_lags"])),
        "--adf_alpha", str(float(exp["adf_alpha"])),
    ]
    if not bool(exp["use_order_search"]):
        cmd.append("--no_order_search")
    return cmd


def run_one_experiment(cfg: RunnerConfig, exp: Dict, root_out_dir: Path) -> Dict:
    exp_name = exp["name"]
    exp_dir = root_out_dir / exp_name
    ensure_dir(exp_dir)

    exp_cfg_path = exp_dir / "sensitivity_experiment_config.json"
    save_json(exp, exp_cfg_path)

    cmd = build_subprocess_command(cfg, exp, exp_dir)
    result = subprocess.run(cmd, capture_output=True, text=True)

    (exp_dir / "runner_stdout.log").write_text(result.stdout or "", encoding="utf-8")
    (exp_dir / "runner_stderr.log").write_text(result.stderr or "", encoding="utf-8")

    status = "ok" if result.returncode == 0 else "failed"
    row = {
        "experiment": exp_name,
        "status": status,
        "returncode": result.returncode,
        "description": exp["description"],
        "order": str(tuple(exp["order"])),
        "seasonal_order": str(tuple(exp["seasonal_order"])),
        "use_order_search": bool(exp["use_order_search"]),
        "output_dir": str(exp_dir),
    }

    if status != "ok":
        return row

    # Load key outputs if available
    per_h_path = exp_dir / "sarimax_per_horizon.csv"
    summary_path = exp_dir / "sarimax_summary.csv"
    nat_summary_path = exp_dir / "sarimax_summary_national_from_district_sum.csv"

    if per_h_path.exists():
        per_h = pd.read_csv(per_h_path)
        per_h.insert(0, "experiment", exp_name)
        per_h.to_csv(exp_dir / "sarimax_per_horizon_with_experiment.csv", index=False)
        row["test_rmse_mean_across_h"] = per_h["RMSE"].mean() if "RMSE" in per_h.columns else pd.NA
        row["test_mae_mean_across_h"] = per_h["MAE"].mean() if "MAE" in per_h.columns else pd.NA
        row["test_f1_mean_across_h"] = per_h["f1"].mean() if "f1" in per_h.columns else pd.NA
    else:
        row["test_rmse_mean_across_h"] = pd.NA
        row["test_mae_mean_across_h"] = pd.NA
        row["test_f1_mean_across_h"] = pd.NA

    if summary_path.exists():
        s = pd.read_csv(summary_path)
        s_test = s[(s["Model"] == "SARIMAX") & (s["Split"] == "test")]
        if not s_test.empty:
            row["overall_test_rmse"] = s_test["RMSE"].iloc[0]
            row["overall_test_mae"] = s_test["MAE"].iloc[0]
            row["overall_test_mase"] = s_test["MASE"].iloc[0]
    if nat_summary_path.exists():
        ns = pd.read_csv(nat_summary_path)
        ns_test = ns[(ns["Model"] == "SARIMAX_from_district_sum") & (ns["Split"] == "test")]
        if not ns_test.empty:
            row["national_test_rmse"] = ns_test["RMSE"].iloc[0]
            row["national_test_mae"] = ns_test["MAE"].iloc[0]

    return row


def aggregate_outputs(root_out_dir: Path, experiment_rows: List[Dict]) -> None:
    exp_df = pd.DataFrame(experiment_rows)
    exp_df.to_csv(root_out_dir / "sarimax_sensitivity_experiments.csv", index=False)

    long_frames = []
    nat_long_frames = []
    for row in experiment_rows:
        if row.get("status") != "ok":
            continue
        exp_name = row["experiment"]
        exp_dir = Path(row["output_dir"])
        per_h_path = exp_dir / "sarimax_per_horizon.csv"
        nat_summary_path = exp_dir / "sarimax_summary_national_from_district_sum.csv"
        summary_path = exp_dir / "sarimax_summary.csv"
        if per_h_path.exists():
            df = pd.read_csv(per_h_path)
            df.insert(0, "experiment", exp_name)
            long_frames.append(df)
        if nat_summary_path.exists():
            df = pd.read_csv(nat_summary_path)
            df.insert(0, "experiment", exp_name)
            nat_long_frames.append(df)
        elif summary_path.exists():
            pass

    if long_frames:
        long_df = pd.concat(long_frames, ignore_index=True)
        long_df.to_csv(root_out_dir / "sarimax_sensitivity_per_horizon_long.csv", index=False)
        numeric_cols = [c for c in ["RMSE", "MAE", "R2", "CVRMSE", "NRMSE", "sMAPE", "MBE", "MedAE", "MASE", "precision", "recall", "f1", "specificity", "roc_auc", "pr_auc"] if c in long_df.columns]
        if numeric_cols:
            summary = long_df.groupby("experiment", as_index=False)[numeric_cols].mean()
            summary.to_csv(root_out_dir / "sarimax_sensitivity_per_horizon_summary.csv", index=False)

    if nat_long_frames:
        nat_df = pd.concat(nat_long_frames, ignore_index=True)
        nat_df.to_csv(root_out_dir / "sarimax_sensitivity_national_summary_long.csv", index=False)


def write_root_summary(root_out_dir: Path, cfg: RunnerConfig, experiments: List[Dict]) -> None:
    lines = [
        "SARIMAX sensitivity run summary",
        "=" * 80,
        f"Base script: {cfg.base_script}",
        f"Fixed baseline config JSON: {cfg.fixed_config_json}",
        f"Output dir: {root_out_dir}",
        "",
        "Sensitivity principle",
        "-" * 80,
        "SARIMAX does not use the exogenous feature-group pipeline used by the ML/DL models.",
        "So sensitivity is done through SARIMAX specification changes rather than fake feature ablations.",
        "",
        "Experiments",
        "-" * 80,
    ]
    for exp in experiments:
        lines.append(
            f"{exp['name']}: order={tuple(exp['order'])}, seasonal_order={tuple(exp['seasonal_order'])}, "
            f"use_order_search={exp['use_order_search']} | {exp['description']}"
        )
    (root_out_dir / "run_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> RunnerConfig:
    p = argparse.ArgumentParser(description="SARIMAX sensitivity runner using the working main SARIMAX script.")
    p.add_argument("--base_script", type=str, default="sarimax_main.py")
    p.add_argument("--fixed_config_json", type=str, default="SARIMAX/outputs/run_config.json")
    p.add_argument("--output_dir", type=str, default="SARIMAX/outputs/sensitivity_final")
    p.add_argument("--input", dest="input_path", type=str, default=None)
    p.add_argument("--target_col", type=str, default=None)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--random_state", type=int, default=None)
    p.add_argument("--min_train_points", type=int, default=None)
    p.add_argument("--maxiter", type=int, default=None)
    p.add_argument("--diagnostics_lags", type=int, default=None)
    p.add_argument("--adf_alpha", type=float, default=None)
    p.add_argument("--ablation_names", type=str, default="full,fixed_order_only,no_seasonal,simpler_nonseasonal")
    a = p.parse_args()
    return RunnerConfig(
        base_script=a.base_script,
        fixed_config_json=a.fixed_config_json,
        output_dir=a.output_dir,
        input_path=a.input_path,
        target_col=a.target_col,
        horizon=a.horizon,
        random_state=a.random_state,
        min_train_points=a.min_train_points,
        maxiter=a.maxiter,
        diagnostics_lags=a.diagnostics_lags,
        adf_alpha=a.adf_alpha,
        ablation_names=a.ablation_names,
    )


def main() -> None:
    cfg = parse_args()
    root_out_dir = Path(cfg.output_dir)
    ensure_dir(root_out_dir)

    base = resolve_baseline(cfg)
    save_json(base, root_out_dir / "baseline_config_resolved.json")

    names = [x.strip() for x in cfg.ablation_names.split(",") if x.strip()]
    experiments = build_experiments(base, names)
    write_root_summary(root_out_dir, cfg, experiments)

    rows = []
    for exp in experiments:
        rows.append(run_one_experiment(cfg, exp, root_out_dir))

    aggregate_outputs(root_out_dir, rows)
    print(f"Saved SARIMAX sensitivity outputs to: {root_out_dir}")


if __name__ == "__main__":
    main()
