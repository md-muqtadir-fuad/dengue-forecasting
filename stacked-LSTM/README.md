MODEL: Stacked LSTM — Direct Multi-Horizon District-Panel

RUN MODE
- Final run: ENABLED
- Hyperparameter tuning: DISABLED in final run
- Reason: best settings already fixed from prior tuning
- Forecasting mode: direct multi-horizon
- Model form: pooled district-level monthly panel
- One separate stacked LSTM model is fit for each forecast horizon

INPUT
- Preprocessed file: ./data/raw/prime_dataset_model_input_with_purge.csv
- Split source: use existing split column from preprocessing
- Required rows in input file: train + val + purge + test must all be kept
- Required core columns: District, Date, Month-year, split, Log_NoOfDenguePatients
- Target: Log_NoOfDenguePatients

FEATURES
- Use numeric predictors from the preprocessing pipeline
- Month encoding from preprocessing: Month_sin + Month_cos
- Horizon-specific seasonal features added inside model: TargetMonth_sin + TargetMonth_cos
- District one-hot encoding: ENABLED by default
- Sequence lookback: 6 months by default
- Model-specific scaling: ENABLED inside the script, fitted on training data only
- No separate global preprocessing scaling file needed

TRAINING
- Train split: used for fitting
- Validation split: used for early stopping / model selection
- Purge split: excluded from fitting and excluded from evaluation targets, but retained in the input file for correct horizon construction
- Test split: final holdout only
- Horizon loop: 1 to 6 months by default
- Epochs: 300 by default
- Batch size: 32 by default
- Early stopping: ENABLED
- ReduceLROnPlateau: ENABLED

TUNING
- Enabled: NO in final run
- Search method: parameter sampling only when explicitly enabled
- Best params source: fixed final configuration from prior tuning
- Optional tuning mode remains available only for separate tuning experiments

OUTPUTS TO SAVE
- fitted model files for each horizon
- training history files
- scaler artifacts for each horizon
- train predictions
- validation predictions
- test predictions
- residual / error tables
- per-horizon metrics
- district-wise metrics
- district × horizon metrics
- outbreak classification metrics
- split manifest
- row-audit output
- permutation importance outputs
- figures
- best-params / best-config record
- run_config.json
- run_summary.txt
- zipped output archive

REPRODUCIBILITY
- Random seed: 42
- Save exact config used: YES
- Save split manifest: YES
- Save best-epoch / final parameter record: YES
- Keep purge rows in source file: YES, mandatory

DEPENDENCIES
- Install dependencies first:
  pip install -r requirements.txt

DEFAULT RUN
- Run with default input/output:
  python stacked_lstm.py

EXPLICIT OUTPUT FOLDER
- Run with a custom output folder:
  python stacked_lstm.py --output_dir stacked-LSTM/outputs

EXPLICIT INPUT FILE
- Run with a specific preprocessed file:
  python stacked_lstm.py --input ./data/raw/prime_dataset_model_input_with_purge.csv

OPTIONAL SWITCHES
- Enable tuning:
  python stacked_lstm.py --use_tuning

- Change maximum forecast horizon:
  python stacked_lstm.py --horizon 6

- Change lookback length:
  python stacked_lstm.py --lookback 6

- Change epochs:
  python stacked_lstm.py --epochs 300

- Change batch size:
  python stacked_lstm.py --batch_size 32

- Explicitly keep district one-hot encoding on:
  python stacked_lstm.py --use_district_ohe

- Explicitly turn district one-hot encoding off:
  python stacked_lstm.py --no_district_ohe

- Change target column if needed:
  python stacked_lstm.py --target_col Log_NoOfDenguePatients

- Change random seed:
  python stacked_lstm.py --random_state 42

- Set loss explicitly:
  python stacked_lstm.py --loss_name mse
  python stacked_lstm.py --loss_name huber

EXAMPLE FINAL RUN
- Recommended final run:
  python stacked_lstm.py --input ./data/raw/prime_dataset_model_input_with_purge.csv --output_dir ./stacked-LSTM/outputs --horizon 6 --lookback 6 --epochs 300 --batch_size 32 --use_district_ohe --loss_name mse

IMPORTANT NOTES
- Do not remove purge rows from the modeling input file.
- Do not use old split arguments like:
  --test_frac
  --val_months
  --purge_months
  because this version takes split definitions directly from the preprocessing file.
- Scaling is part of the model pipeline here, which is appropriate for LSTM.
- Always write the loss explicitly in the final run command if your script has any default inconsistency between config and CLI.

Sensitivity run:
python ./stacked-LSTM/slstm-sen.py --output_dir stacked-LSTM/outputs/sensitivity --fixed_config_json ./stacked-LSTM/outputs/stacked_lstm_best_configs.json --seed_list 42 --ablation_names full,no_climate,no_serotype,no_temporal,no_population_density