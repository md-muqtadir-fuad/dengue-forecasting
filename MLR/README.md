# Multiple Linear Regression (MLR)

Classical linear regression baseline for direct multi-horizon district-level dengue forecasting.

## Overview

**Model Type:** Multiple Linear Regression (Classical)  
**Architecture:** Pooled district-level monthly panel  
**Fitting Engine:** statsmodels OLS with HC3 robust covariance  
**Forecasting:** Direct multi-horizon (1-6 months ahead)  
**Tuning:** Not applicable (classical regression)

## Configuration

### Run Mode
- **Final Run:** ENABLED
- **Hyperparameter Tuning:** NOT APPLICABLE
- **Forecasting:** Direct multi-horizon
- **Model Form:** One separate regression per forecast horizon

### Input Data
- **Source:** `prime_dataset_final_selected_features_with_splits.csv`
- **Split:** Use existing split column from preprocessing
- **Required Columns:**
  - District
  - Date
  - Month-year
  - split (train/val/test)
  - Log_NoOfDenguePatients (target)

### Features
- Numeric predictors from preprocessing pipeline
- Temporal encoding: Month_sin + Month_cos
- **District One-Hot Encoding:** DISABLED by default
  - **Reason:** Prevents perfect multicollinearity with PopulationDensity
  - **If Enabled:** PopulationDensity automatically dropped
- **Scaling:** None (standardized coefficients computed post-fit)

### Training Settings
- **Training Split:** Used for model fitting
- **Validation Split:** Used for validation reporting
- **Purge Split:** Not mandatory but strongly preferred for better horizon coverage
- **Test Split:** Final holdout only
- **Horizon Loop:** 1-6 months (default)
- **Robust Covariance:** HC3 (default)

### Statistical Options
- **Fitting Engine:** statsmodels OLS
- **HC Robust Covariance Types:** HC1, HC3 (configurable)
- **Outbreak Quantile:** 0.90 (default)

### Reproducibility
- **Random Seed:** 42
- **Config Saving:** YES
- **Split Manifest:** YES
- **Target Coverage Report:** YES
- **Diagnostic Statistics:** YES

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Default Run
```bash
python mlr_district.py
```

### Custom Output Directory
```bash
python mlr_district.py --output_dir mlr_run_outputs
```

### Custom Input File
```bash
python mlr_district.py --input prime_dataset_final_selected_features_with_splits.csv
```

### Optional Parameters
```bash
# Change forecast horizon (1-6)
python mlr_district.py --horizon 6

# Enable district one-hot encoding
python mlr_district.py --use_district_ohe

# Specify target column
python mlr_district.py --target_col Log_NoOfDenguePatients

# Set random seed
python mlr_district.py --random_state 42

# Change robust covariance type
python mlr_district.py --robust_cov_type HC3
python mlr_district.py --robust_cov_type HC1

# Change outbreak quantile threshold
python mlr_district.py --outbreak_quantile 0.90
```

### Recommended Final Run
```bash
python mlr_district.py \
  --input prime_dataset_final_selected_features_with_splits.csv \
  --output_dir mlr_run_outputs \
  --horizon 6 \
  --robust_cov_type HC3
```

## Sensitivity Analysis

```bash
python ./mlr-sen.py \
  --robust_cov_type HC3 \
  --outbreak_quantile 0.9 \
  --ablation_names full,no_climate,no_serotype,no_temporal,no_population_density
```

## Outputs

Saved to `--output_dir` (default: `outputs/`):

- `*_train_predictions_long.csv` - Training predictions
- `*_val_predictions_long.csv` - Validation predictions
- `*_test_predictions_long.csv` - Test predictions
- `*_test_residuals_long.csv` - Test residuals
- `*_metrics_by_district.csv` - Performance by district
- `*_metrics_by_district_horizon.csv` - Performance by district × horizon
- `*_per_horizon.csv` - Metrics by forecast horizon
- `*_regimewise_metrics.csv` - Metrics by regime (high/low cases)
- `*_outbreak_classification_metrics.csv` - Outbreak detection metrics
- `*_coefficients_h*.csv` - Raw coefficients per horizon
- `*_standardized_coefficients_h*.csv` - Standardized coefficients per horizon
- `*_aggregate_standardized_coefficients.csv` - Aggregated standardized coefficients
- `*_vif_h*.csv` - Variance Inflation Factor tables per horizon
- `*_diagnostics_h*.txt` - Diagnostics per horizon
- `*_model_summary_h*.txt` - Detailed model summaries per horizon
- `mlr_data_audit.json` - Data audit report
- `split_manifest.csv` - Split information
- `target_coverage_report.json` - Target coverage by horizon
- `run_config.json` - Exact configuration used
- `run_summary.txt` - Run summary and statistics
- Figures and visualizations

## Important Notes

- **No tuning mode** available for this classical regression model
- **Purge rows** are not strictly mandatory but strongly preferred for better target coverage at larger horizons
- **District one-hot encoding is OFF by default** for good reason:
  - Prevents multicollinearity with PopulationDensity
  - Do not enable unless you explicitly want district fixed effects
  - If enabled, PopulationDensity will be automatically dropped