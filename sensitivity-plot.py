#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Selected manuscript comparisons
SELECTED_SPECS: Dict[str, Tuple[str, str]] = {
    'Prophet': ('full', 'no_month_dummies'),
    'MLR': ('full', 'no_serotype'),
    'Attention-LSTM': ('full', 'no_serotype'),
    'SARIMAX': ('full', 'simpler_nonseasonal'),
}

# Stable plotting order
MODEL_ORDER = ['Prophet', 'MLR', 'Attention-LSTM', 'SARIMAX']

# Per-model subplot titles
PANEL_TITLES = {
    'Prophet': 'Prophet',
    'MLR': 'MLR',
    'Attention-LSTM': 'Attention-LSTM',
    'SARIMAX': 'SARIMAX',
}

# Consistent line styles
SPEC_STYLES = {
    'full': {'linestyle': '-', 'marker': 'o', 'linewidth': 2.2, 'markersize': 6},
    'comparator': {'linestyle': '--', 'marker': 's', 'linewidth': 2.0, 'markersize': 5},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Create the manuscript sensitivity RMSE figure from the master sensitivity workbook.'
    )
    parser.add_argument(
        '--input_path',
        default=r'data\raw\sensitivity_master_table_4_models.xlsx',
        help='Path to the sensitivity master workbook.',
    )
    parser.add_argument(
        '--output_dir',
        default=r'data\output',
        help='Directory where the figure files will be saved.',
    )
    parser.add_argument(
        '--sheet_name',
        default='Horizon_Master',
        help='Workbook sheet containing horizon-level results.',
    )
    parser.add_argument(
        '--header_row',
        type=int,
        default=2,
        help='Zero-based header row index in the workbook sheet. Default matches the current master workbook.',
    )
    return parser.parse_args()


def load_horizon_table(workbook_path: Path, sheet_name: str, header_row: int) -> pd.DataFrame:
    if not workbook_path.exists():
        raise FileNotFoundError(f'Workbook not found: {workbook_path}')

    df = pd.read_excel(workbook_path, sheet_name=sheet_name, header=header_row)
    required_columns = {'Model', 'Specification', 'Horizon', 'RMSE'}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns in {sheet_name}: {sorted(missing)}')

    df = df.copy()
    df['Model'] = df['Model'].astype(str).str.strip()
    df['Specification'] = df['Specification'].astype(str).str.strip()
    df['Horizon'] = pd.to_numeric(df['Horizon'], errors='coerce')
    df['RMSE'] = pd.to_numeric(df['RMSE'], errors='coerce')
    df = df.dropna(subset=['Horizon', 'RMSE'])
    df['Horizon'] = df['Horizon'].astype(int)
    return df


def validate_selected_specs(df: pd.DataFrame) -> None:
    for model, (full_spec, comparator_spec) in SELECTED_SPECS.items():
        model_specs = set(df.loc[df['Model'] == model, 'Specification'].unique())
        expected = {full_spec, comparator_spec}
        missing = expected - model_specs
        if missing:
            raise ValueError(
                f'Model {model} is missing required specification(s): {sorted(missing)}. '
                f'Available: {sorted(model_specs)}'
            )

        for spec in expected:
            horizons = sorted(df.loc[(df['Model'] == model) & (df['Specification'] == spec), 'Horizon'].unique())
            if horizons != [1, 2, 3, 4, 5, 6]:
                raise ValueError(
                    f'Model {model}, specification {spec} does not contain the expected horizons 1-6. '
                    f'Found: {horizons}'
                )


def make_panel_plot(df: pd.DataFrame, output_dir: Path) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=False)
    axes = axes.flatten()

    for ax, model in zip(axes, MODEL_ORDER):
        full_spec, comparator_spec = SELECTED_SPECS[model]
        for label, spec, style_key in [
            ('Full', full_spec, 'full'),
            (comparator_spec.replace('_', ' ').title(), comparator_spec, 'comparator'),
        ]:
            sub = (
                df.loc[(df['Model'] == model) & (df['Specification'] == spec), ['Horizon', 'RMSE']]
                .sort_values('Horizon')
            )
            style = SPEC_STYLES[style_key]
            ax.plot(
                sub['Horizon'],
                sub['RMSE'],
                label=label,
                **style,
            )

        ax.set_title(PANEL_TITLES[model])
        ax.set_xticks([1, 2, 3, 4, 5, 6])
        ax.set_xlabel('Horizon (months)')
        ax.set_ylabel('RMSE')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    #fig.suptitle('Sensitivity RMSE by Forecast Horizon', fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    png_path = output_dir / 'sensitivity_rmse_by_horizon.png'
    pdf_path = output_dir / 'sensitivity_rmse_by_horizon.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    args = parse_args()
    workbook_path = Path(args.input_path)
    output_dir = Path(args.output_dir)

    df = load_horizon_table(workbook_path, args.sheet_name, args.header_row)
    validate_selected_specs(df)
    png_path, pdf_path = make_panel_plot(df, output_dir)

    print(f'Saved PNG: {png_path.resolve()}')
    print(f'Saved PDF: {pdf_path.resolve()}')


if __name__ == '__main__':
    main()