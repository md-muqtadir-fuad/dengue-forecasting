MODEL: Attention-Based LSTM (Direct Multi-Horizon District-Panel)

RUN MODE
- Final run: ENABLED
- Hyperparameter tuning: DISABLED
- Reason: best settings already fixed from prior tuning
- Forecasting mode: direct multi-horizon
- Model form: pooled district-level panel, one model per horizon
- Sequence design: lookback-window sequence model with static horizon branch

INPUT
- Preprocessed file: prime_dataset_model_input_with_purge.csv
- Split source: use existing split column from preprocessing
- Required rows in input file: train + val + purge + test must all be kept
- Required core columns: District, Date, Month-year, split, Log_NoOfDenguePatients
- Target: Log_NoOfDenguePatients

FEATURES
- Use selected numeric predictors from the preprocessing pipeline
- Month encoding from preprocessing: Month_sin + Month_cos
- Horizon-specific seasonal features added inside model: TargetMonth_sin + TargetMonth_cos
- District one-hot encoding: ENABLED by default in this file
- Sequence lookback: 6 months by default
- Model-specific scaling: ENABLED inside the script, fitted on training data only for sequence/static branches
- No separate global preprocessing scaling file needed

TRAINING
- Train split: used for fitting
- Validation split: used for early stopping / model selection
- Purge split: excluded from fitting and excluded from evaluation targets, but retained in the input file for correct horizon construction
- Test split: final holdout only
- Horizon loop: 1 to 6 months by default
- Epochs: 300 by default
- Batch size: 32 by default

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
- attention profile outputs
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
  python attention_lstm.py

EXPLICIT OUTPUT FOLDER
- Run with a custom output folder:
  python attention_lstm.py --output_dir attn_lstm_run_outputs

EXPLICIT INPUT FILE
- Run with a specific preprocessed file:
  python attention_lstm.py --input ./data/raw/prime_dataset_model_input_with_purge.csv

OPTIONAL SWITCHES
- Enable tuning:
  python attention_lstm.py --use_tuning

- Change maximum forecast horizon:
  python attention_lstm.py --horizon 6

- Change lookback length:
  python attention_lstm.py --lookback 6

- Change epochs:
  python attention_lstm.py --epochs 300

- Change batch size:
  python attention_lstm.py --batch_size 32

- Explicitly keep district one-hot encoding on:
  python attention_lstm.py --use_district_ohe

- Explicitly turn district one-hot encoding off:
  python attention_lstm.py --no_district_ohe

- Change target column if needed:
  python attention_lstm.py --target_col Log_NoOfDenguePatients

- Change random seed:
  python attention_lstm.py --random_state 42

- Explicitly choose loss:
  python attention_lstm.py --loss_name mse
  python attention_lstm.py --loss_name huber

EXAMPLE FINAL RUN
- Recommended final run:
  python attention_lstm.py --input ./data/raw/prime_dataset_model_input_with_purge.csv --output_dir attn_lstm_run_outputs --horizon 6 --lookback 6 --epochs 300 --batch_size 32 --loss_name mse


  Sensitivity:
  python ./attn-LSTM/alstm-sen.py --output_dir ./attn-LSTM/outputs/sensitivity --fixed_config_json ./attn-LSTM/outputs/attention_lstm_best_configs.json --seed_list 42 --ablation_names full,no_climate,no_serotype,no_temporal,no_population_density