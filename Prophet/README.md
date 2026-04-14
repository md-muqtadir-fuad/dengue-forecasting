# Prophet

Grouped district-level time series forecasting model for monthly dengue case prediction.

## Overview

**Model Type:** Facebook Prophet (Time Series)  
**Architecture:** One Prophet model per district  
**Forecasting:** Rolling-origin monthly forecasting  
**Aggregation:** District forecasts summed to national totals  
**Tuning:** Light validation-based parameter search (ENABLED)

## Configuration

### Run Mode
- **Final Run:** ENABLED
- **Hyperparameter Tuning:** ENABLED (validation-only)
- **Tuning Type:** Light defensible Prophet parameter search
- **Forecast Frequency:** Monthly start (MS)

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
- **External Predictors:** None (univariate only)
- **Month Dummies:** ENABLED (default) to represent monthly seasonality
- **Yearly Seasonality:** DISABLED
- **Scaling:** None (Prophet handles internally)

### Training Settings
- **Training Split:** Initial history for each district
- **Validation Split:** Tuning and rolling-origin validation
- **Purge Split:** Excluded from evaluation (but kept for chronology)
- **Test Split:** Final holdout only
- **Horizon Loop:** 1-6 months (default)
- **Minimum History:** 12 months (default)
- **Forecast Intervals:** ENABLED

### Tuning Parameters
- **Search Scope:**
  - `changepoint_prior_scale`: [0.01, 0.05, 0.10, 0.30]
  - `seasonality_prior_scale`: [1.0, 5.0, 10.0, 20.0]
  - `seasonality_mode`: [additive, multiplicative]
- **Search Basis:** Validation MAE on district-level rolling forecasts
- **Search Granularity:** District-specific parameter selection
- **Test Split Usage:** NO
- **Purge Split Usage:** NO

### Reproducibility
- **Random Seed:** 42
- **Config Saving:** YES
- **Split Manifest:** YES
- **District-Level Tuning:** YES
- **Keep Purge Rows:** YES (mandatory)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Default Run
```bash
python prophet_district.py
```

### Custom Output Directory
```bash
python prophet_district.py --output_dir Prophet/outputs
```

### Custom Input File
```bash
python prophet_district.py --input ./data/raw/prime_dataset_model_input_with_purge.csv
```

### Optional Parameters
```bash
# Enable tuning
python prophet_district.py --use_tuning

# Change forecast horizon (1-6)
python prophet_district.py --horizon 6

# Change minimum history (in months)
python prophet_district.py --min_history_months 12

# Change forecast interval width
python prophet_district.py --interval_width 0.95

# Set changepoint prior scale
python prophet_district.py --changepoint_prior_scale 0.05

# Set seasonality prior scale
python prophet_district.py --seasonality_prior_scale 10.0

# Set seasonality mode
python prophet_district.py --seasonality_mode additive
python prophet_district.py --seasonality_mode multiplicative

# Disable month dummies
python prophet_district.py --no_month_dummies

# Specify target column
python prophet_district.py --target_col Log_NoOfDenguePatients
```

### Recommended Final Run
```bash
python prophet_district.py \
  --input ./data/raw/prime_dataset_model_input_with_purge.csv \
  --output_dir Prophet/outputs \
  --horizon 6 \
  --min_history_months 12 \
  --interval_width 0.95 \
  --use_tuning
```

## Sensitivity Analysis

```bash
python ./prop-sensitivity.py \
  --input ./data/raw/prime_dataset_model_input_with_purge.csv \
  --output_dir Prophet/outputs/sensitivity \
  --fixed_tuning_csv ./Prophet/outputs/prophet_tuning_results.csv \
  --ablation_names full,no_month_dummies
```

## Outputs

Saved to `--output_dir` (default: `outputs/`):

- `*_train_predictions_long.csv` - Training predictions
- `*_val_predictions_long.csv` - Validation predictions
- `*_test_predictions_long.csv` - Test predictions
- `*_metrics_by_district.csv` - Performance by district
- `*_metrics_by_district_horizon.csv` - Performance by district × horizon
- `*_per_horizon.csv` - Metrics by forecast horizon
- `*_national_summary.csv` - Aggregated national metrics from summed forecasts
- `prophet_tuning_results.csv` - Full tuning results by district
- `*_selected_parameters_by_district.csv` - Selected Prophet parameters per district
- `*_district_rolling_forecast_audit.csv` - Rolling forecast audit tables
- `*_regressor_coefficients.csv` - Regressor coefficient tables
- `*_final_train_diagnostics_summary.txt` - Diagnostics from final training
- `*_train_fitted_residuals.csv` - Training residuals
- `*_acf_pacf_diag_results.json` - ACF/PACF/Ljung-Box diagnostics
- Component plots (trend, seasonality, etc.)
- Forecast visualizations by district
- `run_config.json` - Exact configuration used
- `run_summary.txt` - Run summary and statistics
- `split_manifest.csv` - Split information
- Zipped output archive

## Important Notes

- **Tuning is ENABLED** by default in this version; it performs real validation-based Prophet parameter search
- **Tuning is light**, not a large optimization pipeline
- **Do not remove purge rows** from the modeling input file
- **Month dummies are ON by default**; only disable with `--no_month_dummies` if intentionally needed
- **District-level model:** One separate Prophet model per district, not a single pooled global model
- **Univariate approach:** No external regressors used; relies on built-in Prophet seasonality and trend
- **Forecast intervals:** Automatically included in all predictions (95% by default)