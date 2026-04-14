# Temporal Fusion Transformer (TFT)

Direct multi-horizon district-level transformer-based deep learning model for dengue case prediction.

## Overview

**Model Type:** Temporal Fusion Transformer (Deep Learning)  
**Architecture:** Pooled district-level monthly panel with one model per horizon  
**Framework:** PyTorch Lightning + pytorch-forecasting  
**Features:** Multi-head attention, interpretable variable importance  
**Forecasting:** Direct multi-horizon (1-6 months ahead)

## Configuration

### Run Mode
- **Final Run:** ENABLED
- **Hyperparameter Tuning:** DISABLED (best settings fixed from prior tuning)
- **Forecasting:** Direct multi-horizon
- **Model Form:** One separate TFT per forecast horizon (unless internally managing all horizons)

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
- Predictors from preprocessing pipeline
- Temporal encoding: Month_sin + Month_cos (from preprocessing)
- **Horizon-specific features:** TargetMonth_sin + TargetMonth_cos (when applicable)
- **District Encoding:** Kept as static series identifier / encoded feature
- **Lookback / Encoder Length:** Script default unless overridden
- **Model-Specific Scaling:** ENABLED (fitted on training data only)

### Training Settings
- **Training Split:** Used for model fitting
- **Validation Split:** Early stopping / checkpoint selection / model selection
- **Purge Split:** Excluded from fitting and evaluation (but kept in input)
- **Test Split:** Final holdout only
- **Horizon Loop:** 1-6 months (typical default)
- **Early Stopping:** ENABLED
- **Checkpoint Loading:** ENABLED (if supported)
- **Gradient Clipping/Regularization:** Script defaults

### Tuning
- **Enabled:** NO in final run
- **Search Method:** Parameter sampling (when explicitly enabled)
- **Best Params Source:** Fixed from prior tuning
- **Optional Mode:** Available only for separate tuning experiments

### Reproducibility
- **Random Seed:** 42
- **Config Saving:** YES
- **Split Manifest:** YES
- **Best Checkpoint Record:** YES
- **Keep Purge Rows:** YES (mandatory)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Default Run
```bash
python tft_district.py
```

### Custom Output Directory
```bash
python tft_district.py --output_dir TFT/outputs
```

### Custom Input File
```bash
python tft_district.py --input ./data/raw/prime_dataset_model_input_with_purge.csv
```

### Optional Parameters
```bash
# Enable tuning
python tft_district.py --use_tuning

# Change forecast horizon (1-6)
python tft_district.py --horizon 6

# Change lookback / encoder length
python tft_district.py --lookback 6
python tft_district.py --max_encoder_length 6

# Change training epochs
python tft_district.py --epochs 300

# Change batch size
python tft_district.py --batch_size 32

# Change hidden size
python tft_district.py --hidden_size 32

# Change attention heads
python tft_district.py --attention_head_size 4

# Change dropout
python tft_district.py --dropout 0.1

# Change learning rate
python tft_district.py --learning_rate 0.001

# Specify target column
python tft_district.py --target_col Log_NoOfDenguePatients

# Set random seed
python tft_district.py --random_state 42

# Control district encoding
python tft_district.py --use_district_ohe
python tft_district.py --no_district_ohe
```

### Recommended Final Run
```bash
python tft_district.py \
  --input ./data/raw/prime_dataset_model_input_with_purge.csv \
  --output_dir TFT/outputs \
  --horizon 6 \
  --epochs 300 \
  --batch_size 32
```

## Sensitivity Analysis

```bash
python ./tft-sen.py \
  --output_dir ./TFT/outputs/sensitivity \
  --fixed_config_json ./TFT/outputs/tft_best_config.json \
  --seed_list 42 \
  --ablation_names full,no_climate,no_serotype,no_temporal,no_population_density
```

## Outputs

Saved to `--output_dir` (default: `outputs/`):

- `tft_model_h*.ckpt` / model checkpoint files per horizon
- `tft_best_config.json` - Best parameters configuration
- `*_metrics_by_district.csv` - Performance by district
- `*_metrics_by_district_horizon.csv` - Performance by district × horizon
- `*_per_horizon.csv` - Metrics by forecast horizon
- `*_outbreak_classification_metrics.csv` - Outbreak detection metrics
- `*_train_predictions_long.csv` - Training predictions
- `*_val_predictions_long.csv` - Validation predictions
- `*_test_predictions_long.csv` - Test predictions
- `*_test_residuals_long.csv` - Test residuals
- `*_scaler_artifacts.pkl` - Scaling/normalization artifacts
- `*_variable_importance.csv` - TFT variable importance / attention summaries
- `split_manifest.csv` - Split information
- `row_audit.json` - Row-level audit output
- `run_config.json` - Exact configuration used
- `run_summary.txt` - Run summary and statistics
- Figures and visualizations
- Zipped output archive

## Important Notes

- **Do not remove purge rows** from the modeling input file
- **Do not use old split arguments:**
  - `--test_frac`
  - `--val_months`
  - `--purge_months`
  
  This version takes split definitions directly from the preprocessed file

- **Scaling/Normalization:** Part of the model pipeline (appropriate for TFT)
- **Config/CLI Parameters:** If script has separate defaults, always specify in final run commands
- **Early Stopping:** Automatically enabled; monitors validation performance
- **Checkpoint Loading:** Loads best checkpoint if supported in script
- **Attention Mechanisms:** Multi-head attention enables model interpretability

## Environment

**Python 3.8.5**

**Key Packages:**
- pytorch-lightning 2.4.0
- pytorch-forecasting 1.1.1
- torch 2.4.1
- pandas 2.0.3
- numpy 1.24.4
- scikit-learn 1.3.2
- scipy 1.10.1