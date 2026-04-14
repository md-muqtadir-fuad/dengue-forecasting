# SARIMAX

Grouped district-level time series autoregressive model for monthly dengue case prediction.

## Overview

**Model Type:** SARIMAX (Seasonal ARIMA with Exogenous Variables)  
**Architecture:** One SARIMAX model per district  
**Forecasting:** Rolling-origin monthly forecasting  
**Aggregation:** District forecasts summed to national totals  
**Order Search:** Limited candidate comparison (ENABLED by default)

## Configuration

### Run Mode
- **Final Run:** ENABLED
- **Hyperparameter Tuning:** NOT APPLICABLE (classical time series)
- **Order Search:** ENABLED (limited internal candidate comparison)
- **Forecast Mode:** Rolling-origin monthly forecasting

### Input Data
- **Source:** `./data/raw/prime_dataset_model_input_with_purge.csv`
- **Split:** Use existing split column from preprocessing
- **Required Columns:**
  - District
  - Date
  - split (train/val/test)
  - Log_NoOfDenguePatients (target)

### Features & Model Design
- **Model Type:** Univariate per district
- **Exogenous Variables:** None (avoids future-exog leakage in multi-step forecasting)
- **Default Non-Seasonal Order:** (1,1,1)
- **Default Seasonal Order:** (0,1,1,12)
- **Seasonal Period:** 12 months
- **Target Scale:** Log-transformed (forecasts converted back to count scale)
- **Forecast Clipping:** ENABLED (max_pred_log_clip = 12.0)
- **Minimum History:** 24 months required before forecasting

### Order Search
- **Enabled:** YES (by default)
- **Search Type:** Limited candidate specification comparison
- **Selection Basis:** Fit success / convergence / AIC / BIC
- **Disable:** Use `--no_order_search` to force a fixed order

### Training Settings
- **Training Split:** Initial fitting history
- **Validation Split:** Included in rolling-origin evaluation
- **Purge Split:** Excluded from evaluation (but kept for chronology)
- **Test Split:** Final holdout only
- **Horizon Loop:** 1-6 months (default)
- **Optimizer Iterations:** 200 (default)
- **Residual Diagnostics:** ENABLED
- **Forecast Intervals:** ENABLED

### Reproducibility
- **Random Seed:** 42
- **Config Saving:** YES
- **Split Manifest:** YES
- **District-Level Audit:** YES
- **Keep Purge Rows:** YES (mandatory)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Default Run
```bash
python sarimax_district.py
```

### Custom Output Directory
```bash
python sarimax_district.py --output_dir SARIMAX/outputs
```

### Custom Input File
```bash
python sarimax_district.py --input ./data/raw/prime_dataset_model_input_with_purge.csv
```

### Optional Parameters
```bash
# Change forecast horizon (1-6)
python sarimax_district.py --horizon 6

# Specify target column
python sarimax_district.py --target_col Log_NoOfDenguePatients

# Set random seed
python sarimax_district.py --random_state 42

# Change non-seasonal order (p, d, q)
python sarimax_district.py --p 1 --d 1 --q 1

# Change seasonal order (P, D, Q, seasonal_period)
python sarimax_district.py --P 0 --D 1 --Q 1 --seasonal_period 12

# Change minimum training history (in months)
python sarimax_district.py --min_train_points 24

# Change optimizer iterations
python sarimax_district.py --maxiter 200

# Change diagnostic lags
python sarimax_district.py --diagnostics_lags 24

# Disable order search and force specific order
python sarimax_district.py --no_order_search
```

### Recommended Final Run
```bash
python sarimax_district.py \
  --input ./data/raw/prime_dataset_model_input_with_purge.csv \
  --output_dir SARIMAX/outputs \
  --horizon 6
```

### Fixed-Order Run
```bash
python sarimax_district.py \
  --input ./data/raw/prime_dataset_model_input_with_purge.csv \
  --output_dir SARIMAX/outputs \
  --horizon 6 \
  --p 1 --d 1 --q 1 \
  --P 0 --D 1 --Q 1 \
  --seasonal_period 12 \
  --no_order_search
```

## Sensitivity Analysis

```bash
python ./sar-sen.py \
  --base_script ./sarimax_district.py \
  --fixed_config_json ./outputs/run_config.json \
  --output_dir ./outputs/sensitivity
```

## Outputs

Saved to `--output_dir` (default: `outputs/`):

- `*_train_predictions_long.csv` - Training predictions
- `*_val_predictions_long.csv` - Validation predictions
- `*_test_predictions_long.csv` - Test predictions
- `*_metrics_by_district.csv` - Performance by district
- `*_metrics_by_district_horizon.csv` - Performance by district × horizon
- `*_per_horizon.csv` - Metrics by forecast horizon
- `*_regimewise_metrics.csv` - Metrics by regime (high/low cases)
- `*_national_summary.csv` - Aggregated national metrics from summed forecasts
- `*_district_rolling_forecast_audit.csv` - Rolling forecast audit tables
- `*_model_parameters_by_district.csv` - Model parameters per district
- `*_aic_bic_candidates_comparison.csv` - AIC/BIC candidate comparison
- `*_residual_diagnostics.json` - Residual diagnostic statistics
- ACF/PACF diagnostic plots
- Forecast interval outputs
- `split_manifest.csv` - Split information
- `row_audit.json` - Row-level audit output
- `run_config.json` - Exact configuration used
- `run_summary.txt` - Run summary and statistics
- Figures and visualizations
- Zipped output archive

## Important Notes

- **No feature-based tuning** like XGBoost/RF/deep models
- **Only search behavior:** Limited internal SARIMAX order comparison (ON by default)
- **Do not remove purge rows** from the modeling input file
- **District-level model:** One separate SARIMAX per district, not a single pooled global model
- **Univariate approach:** No exogenous regressors to avoid multi-step forecasting leakage
- **Log-scale forecasting:** Targets are log-transformed; forecasts are converted back to count scale

## Environment

Python 3.8.5

Key packages:
- statsmodels 0.12.2
- pmdarima 1.8.0
- pandas 1.1.5
- numpy 1.18.5
- scipy 1.5.4
- scikit-learn 0.23.2