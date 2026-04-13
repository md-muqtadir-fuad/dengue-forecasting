MODEL: Support Vector Regression (SVR) — Direct Multi-Horizon District-Panel

RUN MODE
- Final run: ENABLED
- Hyperparameter tuning: DISABLED in final run
- Reason: best settings already fixed from prior tuning
- Forecasting mode: direct multi-horizon
- Model form: pooled district-level monthly panel
- One separate SVR model is fit for each forecast horizon

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
- Model-specific scaling: ENABLED inside the pipeline using StandardScaler
- No separate global preprocessing scaling file needed
- Default SVR kernel: RBF

TRAINING
- Train split: used for fitting
- Validation split: used for model selection
- Purge split: excluded from fitting and excluded from evaluation targets, but retained in the input file for correct horizon construction
- Test split: final holdout only
- Horizon loop: 1 to 6 months by default
- Outbreak-aware sample weighting: NOT USED
- Scaling is fit inside each horizon model on training data only

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
- permutation importance outputs
- figures
- svr_best_params.json
- run_config.json
- run_summary.txt
- zipped output archive

REPRODUCIBILITY
- Random seed: 42
- Save exact config used: YES
- Save split manifest: YES
- Save best parameter record: YES
- Keep purge rows in source file: YES, mandatory

DEPENDENCIES
- Install dependencies first:
  pip install -r requirements.txt

DEFAULT RUN
- Run with default input/output:
  python svr.py

EXPLICIT OUTPUT FOLDER
- Run with a custom output folder:
  python svr.py --output_dir SVR/outputs

EXPLICIT INPUT FILE
- Run with a specific preprocessed file:
  python svr.py --input ./data/raw/prime_dataset_model_input_with_purge.csv

OPTIONAL SWITCHES
- Enable tuning:
  python svr.py --use_tuning

- Change maximum forecast horizon:
  python svr.py --horizon 6

- Change tuning iterations:
  python svr.py --tuning_iter 24

- Change target column if needed:
  python svr.py --target_col Log_NoOfDenguePatients

- Change random seed:
  python svr.py --random_state 42

- Explicitly keep district one-hot encoding on:
  python svr.py --use_district_ohe

- Explicitly turn district one-hot encoding off:
  python svr.py --no_district_ohe

- Adjust SHAP background/sample settings:
  python svr.py --shap_max_background 80 --shap_max_samples 120 --shap_nsamples 100

EXAMPLE FINAL RUN
- Recommended final run:
  python svr.py --input ./data/raw/prime_dataset_model_input_with_purge.csv --output_dir SVR/outputs --horizon 6 --use_district_ohe

IMPORTANT NOTES
- Do not remove purge rows from the modeling input file.
- Do not use old split arguments like:
  --test_frac
  --val_months
  --purge_months
  because this version takes split definitions directly from the preprocessing file.
- District one-hot encoding is ON by default in this script.
- Scaling is part of the model pipeline here, which is appropriate for SVR.

sensitivity:
python ./SVR/SVR-sen.py --output_dir SVR/outputs/sensitivity --fixed_config_json ./SVR/outputs/svr_best_params.json --seed_list 42 --ablation_names full,no_climate,no_serotype,no_temporal,no_population_density