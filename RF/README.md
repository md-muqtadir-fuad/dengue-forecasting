# Random Forest

Direct multi-horizon district-level ensemble decision tree model for dengue case prediction.

## Overview

**Model Type:** Random Forest (Ensemble Tree-Based)  
**Architecture:** Pooled district-level monthly panel  
**Features:** Outbreak-aware sample weighting, SHAP interpretation  
**Forecasting:** Direct multi-horizon (1-6 months ahead)

## Configuration

### Run Mode
- **Final Run:** ENABLED
- **Hyperparameter Tuning:** DISABLED (best settings fixed from prior tuning)
- **Forecasting:** Direct multi-horizon
- **Model Form:** One separate Random Forest per forecast horizon

### Input Data
- **Source:** `prime_dataset_model_input_with_purge.csv`
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
- **Scaling:** None (Random Forest handles internally)

### Training Settings
- **Training Split:** Used for model fitting
- **Validation Split:** Model selection
- **Purge Split:** Excluded from fitting and evaluation (but kept in input)
- **Test Split:** Final holdout only
- **Horizon Loop:** 1-6 months (default)
- **Sample Weighting:** Outbreak-aware (ENABLED)

### Tuning
- **Enabled:** NO in final run
- **Search Method:** Parameter sampling (when explicitly enabled)
- **Best Params Source:** Fixed from prior tuning
- **Optional Mode:** Available only for separate tuning experiments

### Reproducibility
- **Random Seed:** 42
- **Config Saving:** YES
- **Split Manifest:** YES
- **Best Model-Size Record:** YES
- **Keep Purge Rows:** YES (mandatory)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Default Run
```bash
python random_forest_district.py
```

### Custom Output Directory
```bash
python random_forest_district.py --output_dir rf_run_outputs
```

### Custom Input File
```bash
python random_forest_district.py --input ./data/raw/prime_dataset_model_input_with_purge.csv
```

### Optional Parameters
```bash
# Enable tuning
python random_forest_district.py --use_tuning

# Change forecast horizon (1-6)
python random_forest_district.py --horizon 6

# Change tuning iterations
python random_forest_district.py --tuning_iter 20

# Specify target column
python random_forest_district.py --target_col Log_NoOfDenguePatients

# Set random seed
python random_forest_district.py --random_state 42

# Enable district one-hot encoding
python random_forest_district.py --use_district_ohe

# Disable district one-hot encoding
python random_forest_district.py --no_district_ohe
```

### Recommended Final Run
```bash
python random_forest_district.py \
  --input ./data/raw/prime_dataset_model_input_with_purge.csv \
  --output_dir rf_run_outputs \
  --horizon 6 \
  --use_district_ohe
```

## Sensitivity Analysis

```bash
python ./rf-sen.py \
  --fixed_config_json ./outputs/rf_best_params.json \
  --seed_list 42 \
  --ablation_names full,no_climate,no_serotype,no_temporal,no_population_density
```

## Outputs

Saved to `--output_dir` (default: `outputs/`):

- `rf_model_h*.joblib` - Fitted models (one per horizon)
- `rf_best_params.json` - Best parameters configuration
- `*_metrics_by_district.csv` - Performance by district
- `*_metrics_by_district_horizon.csv` - Performance by district × horizon
- `*_per_horizon.csv` - Metrics by forecast horizon
- `*_regimewise_metrics.csv` - Metrics by regime (high/low cases)
- `*_outbreak_classification_metrics.csv` - Outbreak detection metrics
- `*_train_predictions_long.csv` - Training predictions
- `*_val_predictions_long.csv` - Validation predictions
- `*_test_predictions_long.csv` - Test predictions
- `*_test_residuals_long.csv` - Test residuals
- `*_feature_importance_mdi.csv` - Mean Decrease in Impurity feature importance
- `run_config.json` - Exact configuration used
- `run_summary.txt` - Run summary and statistics
- SHAP feature importance outputs
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

- **TargetMonth_sin and TargetMonth_cos** are created inside the script from TargetDate for each horizon
- **Outbreak-aware weighting** helps balance the importance of rare high-incidence events