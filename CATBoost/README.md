# CATBoost

Direct multi-horizon district-level gradient boosting model for dengue case prediction.

## Overview

**Model Type:** CATBoost (Categorical Boosting)  
**Architecture:** Pooled district-level monthly panel  
**Features:** District as categorical feature, outbreak-aware sample weighting  
**Forecasting:** Direct multi-horizon (1-6 months ahead)

## Configuration

### Run Mode
- **Final Run:** ENABLED
- **Hyperparameter Tuning:** DISABLED (best settings fixed from prior tuning)
- **Forecasting:** Direct multi-horizon
- **Optimization:** Best parameters already determined

### Input Data
- **Source:** `prime_dataset_model_input_with_purge.csv`
- **Split:** Use existing split column from preprocessing
- **Required Columns:**
  - District (categorical feature)
  - Date
  - split (train/val/test)
  - Log_NoOfDenguePatients (target)

### Features
- Predictors from preprocessing pipeline
- **District:** Treated as categorical feature (default)
- **Categorical Features:** Object-type predictors kept as categorical
- **Blocked Features:** Year, Month_sin, Month_cos
- **Temporal Encoding:** TargetMonth_sin + TargetMonth_cos (added by model)
- **Scaling:** None (CatBoost handles internally)

### Training Settings
- **Training Split:** Used for model fitting
- **Validation Split:** Early stopping / model selection
- **Purge Split:** Excluded from fitting and evaluation (but kept in input)
- **Test Split:** Final holdout only
- **Horizon Loop:** 1-6 months (default)
- **Early Stopping:** ENABLED
- **Sample Weighting:** Outbreak-aware (ENABLED)

### Output Aggregation
- **National Predictions:** Computed from summed district forecasts
- **Regime-Wise Metrics:** High/low case regimes analyzed separately

### Reproducibility
- **Random Seed:** 42
- **Config Saving:** YES
- **Split Manifest:** YES
- **Best Iteration Record:** YES
- **Keep Purge Rows:** YES (mandatory)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Default Run
```bash
python catboost_district.py
```

### Custom Output Directory
```bash
python catboost_district.py --output_dir catboost_run_outputs
```

### Custom Input File
```bash
python catboost_district.py --input ./data/raw/prime_dataset_model_input_with_purge.csv
```

### Optional Parameters
```bash
# Enable hyperparameter tuning
python catboost_district.py --use_tuning

# Change forecast horizon (1-6)
python catboost_district.py --horizon 6

# Specify target column
python catboost_district.py --target_col Log_NoOfDenguePatients

# Set random seed
python catboost_district.py --random_state 42

# Disable district categorical feature
python catboost_district.py --no_district_cat

# Block additional features
python catboost_district.py --block_feature Rainfall_lag_3
python catboost_district.py --block_feature denv4

# Adjust outbreak weighting
python catboost_district.py \
  --outbreak_q 0.90 \
  --extreme_outbreak_q 0.97 \
  --outbreak_weight 5.0 \
  --extreme_outbreak_weight 12.0
```

### Recommended Final Run
```bash
python catboost_district.py \
  --input ./data/raw/prime_dataset_model_input_with_purge.csv \
  --output_dir catboost_run_outputs \
  --horizon 6
```

## Sensitivity Analysis

```bash
python ./cat-sen.py \
  --output_dir ./outputs/sensitivity \
  --fixed_config_json ./outputs/catboost_best_params.json \
  --seed_list 42 \
  --ablation_names full,no_climate,no_serotype,no_temporal,no_population_density
```

## Outputs

Saved to `--output_dir` (default: `outputs/`):

- `catboost_model_h*.joblib` - Fitted models (one per horizon)
- `catboost_best_params.json` - Best parameters configuration
- `*_metrics_by_district.csv` - Performance by district
- `*_metrics_by_district_horizon.csv` - Performance by district × horizon
- `*_per_horizon.csv` - Metrics by forecast horizon
- `*_regimewise_metrics.csv` - Metrics by regime (high/low cases)
- `*_outbreak_classification_metrics.csv` - Outbreak detection metrics
- `*_national_summary_metrics.csv` - Aggregated national metrics
- `*_train_predictions_long.csv` - Training predictions
- `*_val_predictions_long.csv` - Validation predictions
- `*_test_predictions_long.csv` - Test predictions
- `*_test_residuals_long.csv` - Test residuals
- `run_config.json` - Exact configuration used
- `run_summary.txt` - Run summary and statistics
- SHAP feature importance outputs
- CatBoost prediction-importance outputs
- Figures and visualizations

## Important Notes

- **Do not remove purge rows** from the modeling input file
- **Do not use old split arguments:**
  - `--test_frac`
  - `--val_months`
  - `--purge_months`
  
  This version takes split definitions directly from the preprocessed file