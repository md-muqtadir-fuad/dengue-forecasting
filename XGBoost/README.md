# XGBoost (Direct Multi-Horizon District-Panel)

## Overview

XGBoost model for dengue forecasting with direct multi-horizon predictions at the district level. Uses a panel structure with separate horizon-specific forecasting logic in a single model run. Best hyperparameters are fixed from prior tuning experiments.

## Configuration

### Run Mode
- **Final run:** ENABLED
- **Hyperparameter tuning:** DISABLED
- **Reason:** Best settings already fixed from prior tuning
- **Forecasting mode:** Direct multi-horizon (separate horizon-specific forecasting logic inside one run)

### Input Data
- **Preprocessed file:** `prime_dataset_model_input_with_purge.csv`
- **Split source:** Use existing split column from preprocessing
- **Required rows in input file:** Train + validation + purge + test must all be kept
- **Target column:** `Log_NoOfDenguePatients`
- **Required core columns:** `District`, `Date`, `Month-year`, `split`, `Log_NoOfDenguePatients`

### Features
- **Numeric predictors:** Use selected numeric predictors from the preprocessing pipeline
- **Month encoding:** `Month_sin` + `Month_cos` from preprocessing
- **Horizon-specific seasonal features:** `TargetMonth_sin` + `TargetMonth_cos` added inside model
- **District one-hot encoding:** ENABLED by default
- **Global scaling:** NOT applied inside this XGBoost pipeline

### Training Settings
- **Train split:** Used for fitting
- **Validation split:** Used for early stopping and model selection
- **Purge split:** Excluded from fitting and evaluation targets, but retained in input file for correct horizon construction
- **Test split:** Final holdout only
- **Horizon loop:** 1 to 6 months by default

### Tuning
- **Enabled:** NO in final run
- **Search method:** Parameter sampling only when explicitly enabled
- **Best params source:** Fixed final configuration from prior tuning
- **Note:** Optional tuning mode remains available only for separate tuning experiments

### Reproducibility
- **Random seed:** 42
- **Save exact config:** YES
- **Save split manifest:** YES
- **Save best-iteration / final parameter record:** YES
- **Keep purge rows:** YES (mandatory)

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Default Run

Run with default input/output:

```bash
python xgboost.py
```

### Custom Output Folder

Run with a custom output directory:

```bash
python xgboost.py --output_dir xgb_run_outputs
```

### Custom Input File

Run with a specific preprocessed file:

```bash
python xgboost.py --input ./data/raw/prime_dataset_model_input_with_purge.csv
```

### Optional Switches

Enable hyperparameter tuning:

```bash
python xgboost.py --use_tuning
```

Change maximum forecast horizon:

```bash
python xgboost.py --horizon 6
```

Change target column:

```bash
python xgboost.py --target_col Log_NoOfDenguePatients
```

Change random seed:

```bash
python xgboost.py --random_state 42
```

Enable district one-hot encoding:

```bash
python xgboost.py --use_district_ohe
```

Disable district one-hot encoding:

```bash
python xgboost.py --no_district_ohe
```

### Recommended Final Run

```bash
python xgboost.py --input ./data/raw/prime_dataset_model_input_with_purge.csv --output_dir xgb_run_outputs --horizon 6 --no_tuning
```

## Sensitivity Analysis

```bash
python ./XGBoost/xgb-sen.py --output_dir ./XGBoost/outputs/sensitivity --fixed_config_json ./XGBoost/outputs/xgb_best_params.json --seed_list 42 --ablation_names full,no_climate,no_serotype,no_temporal,no_population_density
```

## Outputs

The model saves the following outputs:

- Fitted model files for each horizon
- Train predictions
- Validation predictions
- Test predictions
- Residual / error tables
- Per-horizon metrics
- District-wise metrics
- District × horizon metrics
- Outbreak classification metrics
- Split manifest
- Row-audit output
- SHAP outputs
- Gain importance outputs
- Figures
- `xgb_best_params.json`
- `run_config.json`
- `run_summary.txt`
- Zipped output archive

## Important Notes

- **Do not remove purge rows** from the modeling input file.
- **Do not use old split arguments** such as `--test_frac`, `--val_months`, or `--purge_months`, because this version takes split definitions directly from the preprocessing file.