import argparse
import re
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRETTY_NAMES = {
    "denv4": "DENV-4",
    "Year": "Year",
    "AvgTemp_lag_3": "Avg. Temp (Lag 3)",
    "PopulationDensity": "Pop. Density",
    "MonthlyPrevailingWindDir_ENE": "Wind Dir. (ENE)",
    "Rainfall_lag_2": "Rainfall (Lag 2)",
    "Rainfall_lag_3": "Rainfall (Lag 3)",
    "MonthlyAvgSunshineHours_lag_1": "Sunshine Hrs (Lag 1)",
    "denv1_lag_1": "DENV-1 (Lag 1)",
    "Humidity_lag_1": "Humidity (Lag 1)",
    "Log_NoOfDenguePatients": "Log(Dengue Cases)",
    "Month_sin": "Month (Sin)",
    "Month_cos": "Month (Cos)",
    "TargetMonth_sin": "Target Month (Sin)",
    "TargetMonth_cos": "Target Month (Cos)",
    "AR_LogCases": "AR Log(Dengue Cases)",
    "District_DHAKA": "District DHAKA",
    "District_BARISHAL": "District BARISHAL",
    "District_BHOLA": "District BHOLA",
    "District_CHATTOGRAM": "District CHATTOGRAM",
    "District_COXS BAZAR": "District COXS BAZAR",
    "District_FARIDPUR": "District FARIDPUR",
    "District_JESSORE": "District JESSORE",
    "District_KHULNA": "District KHULNA",
    "District_MYMENSINGH": "District MYMENSINGH",
    "District_RAJSHAHI": "District RAJSHAHI",
    "District_SYLHET": "District SYLHET",
}


def pretty_name(col: str) -> str:
    if col in PRETTY_NAMES:
        return PRETTY_NAMES[col]

    c = str(col).strip()

    lag_match = re.match(r"^(.*)_lag_(\d+)$", c)
    if lag_match:
        base, lag_num = lag_match.groups()
        return f"{pretty_name(base)} (Lag {lag_num})"

    if re.fullmatch(r"denv\d+", c.lower()):
        num = re.findall(r"\d+", c)[0]
        return f"DENV-{num}"

    if c.startswith("District_"):
        return c.replace("_", " ")

    c = c.replace("_", " ")
    c = re.sub(r"\s+", " ", c).strip()
    c = re.sub(r"([a-z])([A-Z])", r"\1 \2", c)
    c = c.replace("Avg ", "Avg. ")
    c = c.replace("Min ", "Min. ")
    c = c.replace("Max ", "Max. ")
    c = c.replace("Pop Density", "Pop. Density")
    return c


def load_permutation_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    first_col = df.columns[0]
    if first_col.startswith("Unnamed"):
        df = df.rename(columns={first_col: "Feature"})
    elif first_col != "Feature":
        df = df.rename(columns={first_col: "Feature"})

    horizon_cols = [c for c in df.columns if re.fullmatch(r"h\d+", str(c))]
    if not horizon_cols:
        raise ValueError("No horizon columns found. Expected columns like h1, h2, ..., h6.")

    # Sort horizons numerically
    horizon_cols = sorted(horizon_cols, key=lambda x: int(x[1:]))

    if "mean_delta_mae" not in df.columns:
        df["mean_delta_mae"] = df[horizon_cols].mean(axis=1)

    df["FeaturePretty"] = df["Feature"].map(pretty_name)
    df = df.sort_values("mean_delta_mae", ascending=False).reset_index(drop=True)

    return df, horizon_cols


def find_uncertainty_columns(df: pd.DataFrame, horizon_cols: list[str]) -> dict[str, str]:
    """
    Looks for uncertainty columns matching:
    h1_std, h1_se, h1_sem, h1_err
    """
    mapping = {}
    candidates = ["std", "se", "sem", "err"]

    for h in horizon_cols:
        for suffix in candidates:
            col = f"{h}_{suffix}"
            if col in df.columns:
                mapping[h] = col
                break

    return mapping


def save_summary_csv(df: pd.DataFrame, horizon_cols: list[str], out_dir: Path) -> None:
    cols = ["Feature", "FeaturePretty", "mean_delta_mae"] + horizon_cols
    existing = [c for c in cols if c in df.columns]
    df.loc[:, existing].to_csv(out_dir / "attention_lstm_permutation_importance_summary.csv", index=False)


def plot_aggregate_bar(df: pd.DataFrame, out_dir: Path, top_n: int = 12) -> None:
    plot_df = df.head(top_n).copy().iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, max(5, 0.45 * len(plot_df))))
    ax.barh(plot_df["FeaturePretty"], plot_df["mean_delta_mae"])
    ax.set_xlabel("Mean ΔMAE after permutation")
    ax.set_ylabel("Predictor")
    ax.set_title("Attention-LSTM Aggregate Permutation Importance")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "attention_lstm_permutation_importance_bar_aggregate.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "attention_lstm_permutation_importance_bar_aggregate.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame, horizon_cols: list[str], out_dir: Path, top_n: int = 12) -> None:
    plot_df = df.head(top_n).copy()
    heat = plot_df.set_index("FeaturePretty")[horizon_cols]

    fig, ax = plt.subplots(figsize=(10, max(5, 0.45 * len(plot_df))))
    im = ax.imshow(heat.values, aspect="auto")

    ax.set_xticks(range(len(horizon_cols)))
    ax.set_xticklabels([h.upper() for h in horizon_cols])
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Predictor")
    ax.set_title("Attention-LSTM Permutation Importance by Horizon")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("ΔMAE after permutation")

    fig.tight_layout()
    fig.savefig(out_dir / "attention_lstm_permutation_importance_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "attention_lstm_permutation_importance_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_line_chart(
    df: pd.DataFrame,
    horizon_cols: list[str],
    out_dir: Path,
    top_n: int = 8,
) -> None:
    plot_df = df.head(top_n).copy()
    uncertainty_map = find_uncertainty_columns(plot_df, horizon_cols)

    markers = cycle(["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"])
    x = np.arange(1, len(horizon_cols) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    for _, row in plot_df.iterrows():
        y = row[horizon_cols].to_numpy(dtype=float)
        marker = next(markers)

        if len(uncertainty_map) == len(horizon_cols):
            yerr = np.array([row[uncertainty_map[h]] for h in horizon_cols], dtype=float)
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker=marker,
                linewidth=1.8,
                capsize=3,
                label=row["FeaturePretty"],
            )
        else:
            ax.plot(
                x,
                y,
                marker=marker,
                linewidth=1.8,
                label=row["FeaturePretty"],
            )

    ax.set_xticks(x)
    ax.set_xticklabels([h.upper() for h in horizon_cols])
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("ΔMAE after permutation")
    ax.set_title("Attention-LSTM Permutation Importance Across Horizons")
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)

    fig.tight_layout()
    fig.savefig(out_dir / "attention_lstm_permutation_importance_line.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "attention_lstm_permutation_importance_line.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Attention-LSTM aggregate permutation importance.")
    parser.add_argument(
        "--input_path",
        type=str,
        default=r"attn-LSTM\outputs\attention_lstm_permutation_importance.csv",
        help="Path to attention_lstm_permutation_importance.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"data\output",
        help="Directory to save plots",
    )
    parser.add_argument(
        "--top_n_bar",
        type=int,
        default=12,
        help="Top N predictors for aggregate bar chart",
    )
    parser.add_argument(
        "--top_n_heatmap",
        type=int,
        default=12,
        help="Top N predictors for heatmap",
    )
    parser.add_argument(
        "--top_n_line",
        type=int,
        default=8,
        help="Top N predictors for line chart",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df, horizon_cols = load_permutation_csv(input_path)

    save_summary_csv(df, horizon_cols, output_dir)
    plot_aggregate_bar(df, output_dir, top_n=args.top_n_bar)
    plot_heatmap(df, horizon_cols, output_dir, top_n=args.top_n_heatmap)
    plot_line_chart(df, horizon_cols, output_dir, top_n=args.top_n_line)

    print("Saved files:")
    print(output_dir / "attention_lstm_permutation_importance_summary.csv")
    print(output_dir / "attention_lstm_permutation_importance_bar_aggregate.png")
    print(output_dir / "attention_lstm_permutation_importance_bar_aggregate.pdf")
    print(output_dir / "attention_lstm_permutation_importance_heatmap.png")
    print(output_dir / "attention_lstm_permutation_importance_heatmap.pdf")
    print(output_dir / "attention_lstm_permutation_importance_line.png")
    print(output_dir / "attention_lstm_permutation_importance_line.pdf")


if __name__ == "__main__":
    main()