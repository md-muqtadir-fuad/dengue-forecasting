# Attention-Based LSTM

Direct multi-horizon district-level forecasting model for dengue case prediction.

## Overview

**Model Type:** LSTM with Attention Mechanism  
**Architecture:** Pooled district-level panel with one model per horizon  
**Sequence Design:** Lookback-window sequence model with static horizon branch

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
  - District
  - Date
  - Month-year
  - split (train/val/test)
  - Log_NoOfDenguePatients (target)

### Features
- Selected numeric predictors from preprocessing pipeline
- Temporal encoding: Month sin/cos + TargetMonth sin/cos
- District one-hot encoding: ENABLED (default)
- Sequence lookback: 6 months (default)
- Model-specific scaling: ENABLED (fitted on training data only)

### Training Settings
- **Training Split:** Used for model fitting
- **Validation Split:** Early stopping / model selection
- **Purge Split:** Excluded from fitting and evaluation
- **Test Split:** Final holdout only
- **Horizon Loop:** 1-6 months (default)
- **Epochs:** 300 (default)
- **Batch Size:** 32 (default)

### Reproducibility
- **Random Seed:** 42
- **Config Saving:** YES
- **Split Manifest:** YES
- **Keep Purge Rows:** YES (mandatory)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Default Run
```bash
python attention_lstm_district.py
```

### Custom Output Directory
```bash
python attention_lstm_district.py --output_dir attn_lstm_run_outputs
```

### Custom Input File
```bash
python attention_lstm_district.py --input ./data/raw/prime_dataset_model_input_with_purge.csv
```

### Optional Parameters
```bash
# Enable hyperparameter tuning
python attention_lstm_district.py --use_tuning

# Change forecast horizon (1-6)
python attention_lstm_district.py --horizon 6

# Change lookback length
python attention_lstm_district.py --lookback 6

# Change training epochs
python attention_lstm_district.py --epochs 300

# Change batch size
python attention_lstm_district.py --batch_size 32

# Enable district one-hot encoding
python attention_lstm_district.py --use_district_ohe

# Disable district one-hot encoding
python attention_lstm_district.py --no_district_ohe

# Specify target column
python attention_lstm_district.py --target_col Log_NoOfDenguePatients

# Set random seed
python attention_lstm_district.py --random_state 42

# Choose loss function
python attention_lstm_district.py --loss_name mse
python attention_lstm_district.py --loss_name huber
```

### Recommended Final Run
```bash
python attention_lstm_district.py \
  --input ./data/raw/prime_dataset_model_input_with_purge.csv \
  --output_dir attn_lstm_run_outputs \
  --horizon 6 \
  --lookback 6 \
  --epochs 300 \
  --batch_size 32 \
  --loss_name mse
```

## Sensitivity Analysis

```bash
python ./alstm-sen.py \
  --output_dir ./outputs/sensitivity \
  --fixed_config_json ./outputs/attention_lstm_best_configs.json \
  --seed_list 42 \
  --ablation_names full,no_climate,no_serotype,no_temporal,no_population_density
```

## Outputs

Saved to `--output_dir` (default: `outputs/`):

- `attention_lstm_model_h*.keras` - Fitted models (one per horizon)
- `*_best_configs.json` - Best parameters configuration
- `*_metrics_by_district.csv` - Performance by district
- `*_metrics_by_district_horizon.csv` - Performance by district × horizon
- `*_per_horizon.csv` - Metrics by forecast horizon
- `*_outbreak_classification_metrics.csv` - Outbreak detection metrics
- `*_train_predictions_long.csv` - Training predictions
- `*_val_predictions_long.csv` - Validation predictions
- `*_test_predictions_long.csv` - Test predictions
- `*_test_residuals_long.csv` - Test residuals
- `run_config.json` - Exact configuration used
- `run_summary.txt` - Run summary and statistics
- Attention profiles and permutation importance outputs
- Figures and visualizations