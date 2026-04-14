# Stacked LSTM

Direct multi-horizon district-level deep learning RNN model for dengue case prediction.

## Overview

**Model Type:** Stacked LSTM (Recurrent Neural Network)  
**Architecture:** Pooled district-level panel with one model per horizon  
**Sequence Design:** Multi-layer LSTM with lookback window  
**Training:** Early stopping with ReduceLROnPlateau  
**Optimization:** Best parameters fixed from prior tuning

## Configuration

### Run Mode
- **Final Run:** ENABLED
- **Hyperparameter Tuning:** DISABLED (best settings fixed from prior tuning)
- **Forecasting:** Direct multi-horizon
- **Model Form:** One stacked LSTM per forecast horizon

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
- **Sequence Lookback:** 6 months (default)
- **Model-Specific Scaling:** ENABLED (fitted on training data only)

### Training Settings
- **Training Split:** Used for model fitting
- **Validation Split:** Early stopping / model selection
- **Purge Split:** Excluded from fitting and evaluation (but kept in input)
- **Test Split:** Final holdout only
- **Horizon Loop:** 1-6 months (default)
- **Epochs:** 300 (default)
- **Batch Size:** 32 (default)
- **Early Stopping:** ENABLED
- **ReduceLROnPlateau:** ENABLED (learning rate scheduling)

### Reproducibility
- **Random Seed:** 42
- **Config Saving:** YES
- **Split Manifest:** YES
- **Best Epoch Record:** YES
- **Keep Purge Rows:** YES (mandatory)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Default Run
```bash
python stacked_lstm_district.py
```

### Custom Output Directory
```bash
python stacked_lstm_district.py --output_dir stacked-LSTM/outputs
```

### Custom Input File
```bash
python stacked_lstm_district.py --input ./data/raw/prime_dataset_model_input_with_purge.csv
```

### Optional Parameters
```bash
# Enable tuning
python stacked_lstm_district.py --use_tuning

# Change forecast horizon (1-6)
python stacked_lstm_district.py --horizon 6

# Change lookback length
python stacked_lstm_district.py --lookback 6

# Change training epochs
python stacked_lstm_district.py --epochs 300

# Change batch size
python stacked_lstm_district.py --batch_size 32

# Enable district one-hot encoding
python stacked_lstm_district.py --use_district_ohe

# Disable district one-hot encoding
python stacked_lstm_district.py --no_district_ohe

# Specify target column
python stacked_lstm_district.py --target_col Log_NoOfDenguePatients

# Set random seed
python stacked_lstm_district.py --random_state 42

# Choose loss function
python stacked_lstm_district.py --loss_name mse
python stacked_lstm_district.py --loss_name huber
```

### Recommended Final Run
```bash
python stacked_lstm_district.py \
  --input ./data/raw/prime_dataset_model_input_with_purge.csv \
  --output_dir ./stacked-LSTM/outputs \
  --horizon 6 \
  --lookback 6 \
  --epochs 300 \
  --batch_size 32 \
  --use_district_ohe \
  --loss_name mse
```

## Sensitivity Analysis

```bash
python ./slstm-sen.py \
  --output_dir stacked-LSTM/outputs/sensitivity \
  --fixed_config_json ./stacked-LSTM/outputs/stacked_lstm_best_configs.json \
  --seed_list 42 \
  --ablation_names full,no_climate,no_serotype,no_temporal,no_population_density
```

## Outputs

Saved to `--output_dir` (default: `outputs/`):

- `stacked_lstm_model_h*.keras` - Fitted models (one per horizon)
- `*_best_configs.json` - Best parameters configuration
- `*_metrics_by_district.csv` - Performance by district
- `*_metrics_by_district_horizon.csv` - Performance by district × horizon
- `*_per_horizon.csv` - Metrics by forecast horizon
- `*_outbreak_classification_metrics.csv` - Outbreak detection metrics
- `*_train_predictions_long.csv` - Training predictions
- `*_val_predictions_long.csv` - Validation predictions
- `*_test_predictions_long.csv` - Test predictions
- `*_test_residuals_long.csv` - Test residuals
- `*_training_history_h*.csv` - Training history per horizon
- `*_scaler_artifacts_h*.pkl` - Scaling artifacts per horizon
- `run_config.json` - Exact configuration used
- `run_summary.txt` - Run summary and statistics
- Permutation importance outputs
- Figures and visualizations
- Zipped output archive

## Important Notes

- **Do not remove purge rows** from the modeling input file
- **Do not use old split arguments:**
  - `--test_frac`
  - `--val_months`
  - `--purge_months`
  
  This version takes split definitions directly from the preprocessed file

- **Scaling:** Part of the model pipeline (appropriate for LSTM)
- **Loss Function:** Always specify explicitly in final run commands if script has default inconsistencies
- **Early Stopping:** Automatically enabled; monitors validation performance
- **Learning Rate Scheduling:** ReduceLROnPlateau optimizes convergence