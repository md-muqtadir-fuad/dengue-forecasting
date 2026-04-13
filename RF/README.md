MODEL: Random Forest — Direct Multi-Horizon District-Panel

RUN MODE
- Final run: ENABLED
- Hyperparameter tuning: DISABLED in final run
- Reason: best settings already fixed from prior tuning
- Forecasting mode: direct multi-horizon
- Model form: pooled district-level monthly panel
- One separate Random Forest model is fit for each forecast horizon

INPUT
- Preprocessed file: prime_dataset_model_input_with_purge.csv
- Split source: use existing split column from preprocessing
- Required rows in input file: train + val + purge + test must all be kept
- Required core columns: District, Date, Month-year, split, Log_NoOfDenguePatients
- Target: Log_NoOfDenguePatients

FEATURES
- Use numeric predictors from the preprocessing pipeline
- Month encoding from preprocessing: Month_sin + Month_cos
- Horizon-specific seasonal features added inside model: TargetMonth_sin + TargetMonth_cos
- District one-hot encoding: ENABLED by default
- No global scaling inside this Random Forest pipeline

TRAINING
- Train split: used for fitting
- Validation split: used for model selection
- Purge split: excluded from fitting and excluded from evaluation targets, but retained in the input file for correct horizon construction
- Test split: final holdout only
- Horizon loop: 1 to 6 months by default
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
- residual / error tables
- per-horizon metrics
- district-wise metrics
- district × horizon metrics
- regime-wise metrics
- outbreak classification metrics
- split manifest
- row-audit output
- SHAP outputs
- MDI feature importance outputs
- figures
- rf_best_params.json
- run_config.json
- run_summary.txt
- zipped output archive

REPRODUCIBILITY
- Random seed: 42
- Save exact config used: YES
- Save split manifest: YES
- Save best model-size / final parameter record: YES
- Keep purge rows in source file: YES, mandatory

DEPENDENCIES
- Install dependencies first:
  pip install -r requirements.txt

DEFAULT RUN
- Run with default input/output:
  python rf.py

EXPLICIT OUTPUT FOLDER
- Run with a custom output folder:
  python rf.py --output_dir rf_run_outputs

EXPLICIT INPUT FILE
- Run with a specific preprocessed file:
  python rf.py --input ./data/raw/prime_dataset_model_input_with_purge.csv

OPTIONAL SWITCHES
- Enable tuning:
  python rf.py --use_tuning

- Change maximum forecast horizon:
  python rf.py --horizon 6

- Change tuning iterations:
  python rf.py --tuning_iter 20

- Change target column if needed:
  python rf.py --target_col Log_NoOfDenguePatients

- Change random seed:
  python rf.py --random_state 42

- Explicitly keep district one-hot encoding on:
  python rf.py --use_district_ohe

- Explicitly turn district one-hot encoding off:
  python rf.py --no_district_ohe

EXAMPLE FINAL RUN
- Recommended final run:
  python rf.py --input ./data/raw/prime_dataset_model_input_with_purge.csv --output_dir rf_run_outputs --horizon 6 --use_district_ohe

IMPORTANT NOTES
- Do not remove purge rows from the modeling input file.
- Do not use old split arguments like:
  --test_frac
  --val_months
  --purge_months
  because this version takes split definitions directly from the preprocessing file.
- TargetMonth_sin and TargetMonth_cos are created inside the script from TargetDate for each horizon.

sensitivity:
python ./RF/rf-sen.py --fixed_config_json ./RF/outputs/rf_best_params.json --seed_list 42 --ablation_names full,no_climate,no_serotype,no_temporal,no_population_density