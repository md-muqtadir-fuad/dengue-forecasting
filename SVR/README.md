# Support Vector Regression (SVR)

Direct multi-horizon district-level kernel-based regression model for dengue case prediction.

## Overview

**Model Type:** Support Vector Regression (Kernel-Based)  
**Architecture:** Pooled district-level monthly panel  
**Kernel:** RBF (default)  
**Features:** SHAP interpretation, permutation importance  
**Forecasting:** Direct multi-horizon (1-6 months ahead)

## Configuration

### Run Mode
- **Final Run:** ENABLED
- **Hyperparameter Tuning:** DISABLED (best settings fixed from prior tuning)
- **Forecasting:** Direct multi-horizon
- **Model Form:** One separate SVR per forecast horizon

### Input Data
- **Source:** `./data/raw/prime_dataset_model_input_with_purge.csv`
- **Split:** Use existing split column from preprocessing
- **Required Columns:**
  - District
  - Date
  - Month-year
  - split (train/val/test)
  - Log_NoOfDenguePatients (target)

### Features
- Numeric predictors from preprocessing pipeline
- Temporal encoding: Month_sin + Month_cos (from preprocessing)
- **Horizon-specific features:** TargetMonth_sin + TargetMonth_cos (created by model)
- **District One-Hot Encoding:** ENABLED (default)
- **Default Kernel:** RBF
- **Model-Specific Scaling:** ENABLED (StandardScaler fitted on training data only)

### Training Settings
- **Training Split:** Used for model fitting
- **Validation Split:** Model selection
- **Purge Split:** Excluded from fitting and evaluation (but kept in input)
- **Test Split:** Final holdout only
- **Horizon Loop:** 1-6 months (default)
- **Sample Weighting:** Not used
- **Scaling:** Fitted inside each horizon model on training data only

### Tuning
- **Enabled:** NO in final run
- **Search Method:** Parameter sampling (when explicitly enabled)
- **Best Params Source:** Fixed from prior tuning
- **Optional Mode:** Available only for separate tuning experiments

### Reproducibility
- **Random Seed:** 42
- **Config Saving:** YES
- **Split Manifest:** YES
- **Best Parameter Record:** YES
- **Keep Purge Rows:** YES (mandatory)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Default Run
```bash
python svr_district.py
```

### Custom Output Directory
```bash
python svr_district.py --output_dir SVR/outputs
```

### Custom Input File
```bash
python svr_district.py --input ./data/raw/prime_dataset_model_input_with_purge.csv
```

### Optional Parameters
```bash
# Enable tuning
python svr_district.py --use_tuning

# Change forecast horizon (1-6)
python svr_district.py --horizon 6

# Change tuning iterations
python svr_district.py --tuning_iter 24

# Specify target column
python svr_district.py --target_col Log_NoOfDenguePatients

# Set random seed
python svr_district.py --random_state 42

# Enable district one-hot encoding
python svr_district.py --use_district_ohe

# Disable district one-hot encoding
python svr_district.py --no_district_ohe

# Adjust SHAP settings
python svr_district.py --shap_max_background 80 --shap_max_samples 120 --shap_nsamples 100
```

### Recommended Final Run
```bash
python svr_district.py \
  --input ./data/raw/prime_dataset_model_input_with_purge.csv \
  --output_dir SVR/outputs \
  --horizon 6 \
  --use_district_ohe
```

## Sensitivity Analysis

```bash
python ./SVR-sen.py \
  --output_dir SVR/outputs/sensitivity \
  --fixed_config_json ./SVR/outputs/svr_best_params.json \
  --seed_list 42 \
  --ablation_names full,no_climate,no_serotype,no_temporal,no_population_density
```

## Outputs

Saved to `--output_dir` (default: `outputs/`):

- `svr_model_h*.joblib` - Fitted models (one per horizon)
- `svr_best_params.json` - Best parameters configuration
- `*_metrics_by_district.csv` - Performance by district
- `*_metrics_by_district_horizon.csv` - Performance by district × horizon
- `*_per_horizon.csv` - Metrics by forecast horizon
- `*_regimewise_metrics.csv` - Metrics by regime (high/low cases)
- `*_outbreak_classification_metrics.csv` - Outbreak detection metrics
- `*_train_predictions_long.csv` - Training predictions
- `*_val_predictions_long.csv` - Validation predictions
- `*_test_predictions_long.csv` - Test predictions
- `*_test_residuals_long.csv` - Test residuals
- `run_config.json` - Exact configuration used
- `run_summary.txt` - Run summary and statistics
- SHAP feature importance outputs
- Permutation importance outputs
- Row audit output
- Figures and visualizations
- Zipped output archive

## Important Notes

- **Do not remove purge rows** from the modeling input file
- **Do not use old split arguments:**
  - `--test_frac`
  - `--val_months`
  - `--purge_months`
  
  This version takes split definitions directly from the preprocessed file

- **District one-hot encoding is ON by default** in this script
- **Scaling:** Part of the model pipeline (appropriate for SVR with RBF kernel)
- **Kernel:** RBF is default; can be modified via configuration
- **SHAP Interpretation:** Allows understanding feature contributions to predictions
- **Permutation Importance:** Computed for global feature importance assessment