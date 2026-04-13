MODEL: CatBoost (Direct Multi-Horizon District-Panel)

RUN MODE
- Final run: ENABLED
- Hyperparameter tuning: DISABLED
- Reason: best settings already fixed from prior tuning
- Forecasting mode: direct multi-horizon
- Model form: pooled district-level monthly panel
- District handling: included directly as a categorical feature

INPUT
- Preprocessed file: prime_dataset_model_input_with_purge.csv
- Split source: use existing split column from preprocessing
- Required rows in input file: train + val + purge + test must all be kept
- Required core columns: District, Date, split, Log_NoOfDenguePatients
- Optional but supported: Month-year and other preprocessed predictors
- Target: Log_NoOfDenguePatients

FEATURES
- Use predictors from the preprocessing pipeline
- District is used as a categorical feature by default
- Object-type predictors are kept as categorical features by default
- Blocked features by default: Year, Month_sin, Month_cos
- Horizon-specific seasonal features added inside model: TargetMonth_sin + TargetMonth_cos
- No global scaling inside this CatBoost pipeline

TRAINING
- Train split: used for fitting
- Validation split: used for early stopping / model selection
- Purge split: excluded from fitting and excluded from evaluation targets, but retained in the input file for correct horizon construction
- Test split: final holdout only
- Horizon loop: 1 to 6 months by default
- Early stopping: ENABLED
- Outbreak-aware sample weighting: ENABLED

TUNING
- Enabled: NO in final run
- Search method: parameter sampling only when explicitly enabled
- Best params source: fixed final configuration from prior tuning
- Optional tuning mode remains available only for separate tuning experiments

OUTPUTS TO SAVE
- fitted model files for each horizon
- train predictions
- validation predictions
- test predictions
- national predictions from summed district forecasts
- residual / error tables
- per-horizon metrics
- district-wise metrics
- regime-wise metrics
- national summary metrics
- outbreak classification metrics
- split manifest
- row-audit output
- SHAP outputs
- CatBoost prediction-importance outputs
- figures
- catboost_best_params.json
- run_config.json
- run_summary.txt
- zipped output archive

REPRODUCIBILITY
- Random seed: 42
- Save exact config used: YES
- Save split manifest: YES
- Save best-iteration / final parameter record: YES
- Keep purge rows in source file: YES, mandatory

DEPENDENCIES
- Install dependencies first:
  pip install -r requirements.txt

DEFAULT RUN
- Run with default input/output:
  python catboost.py

EXPLICIT OUTPUT FOLDER
- Run with a custom output folder:
  python catboost.py --output_dir catboost_run_outputs

EXPLICIT INPUT FILE
- Run with a specific preprocessed file:
  python catboost.py --input ./data/raw/prime_dataset_model_input_with_purge.csv

OPTIONAL SWITCHES
- Enable tuning:
  python catboost.py --use_tuning

- Change maximum forecast horizon:
  python catboost.py --horizon 6

- Change target column if needed:
  python catboost.py --target_col Log_NoOfDenguePatients

- Change random seed:
  python catboost.py --random_state 42

- Turn district categorical feature off:
  python catboost.py --no_district_cat

- Add extra blocked features:
  python catboost.py --block_feature Rainfall_lag_3
  python catboost.py --block_feature denv4

- Adjust outbreak weighting setup:
  python catboost.py --outbreak_q 0.90 --extreme_outbreak_q 0.97 --outbreak_weight 5.0 --extreme_outbreak_weight 12.0

EXAMPLE FINAL RUN
- Recommended final run:
  python catboost.py --input ./data/raw/prime_dataset_model_input_with_purge.csv --output_dir catboost_run_outputs --horizon 6

IMPORTANT NOTE
- Do not remove purge rows from the modeling input file.
- Do not use old split arguments like:
  --test_frac
  --val_months
  --purge_months
  because this version takes split definitions directly from the preprocessing file.

  sesitivity

python ./CATBoost/cat-sen.py --output_dir ./CatBoost/outputs/sensitivity --fixed_config_json ./CATBoost/outputs/catboost_best_params.json --seed_list 42 --ablation_names full,no_climate,no_serotype,no_temporal,no_population_density