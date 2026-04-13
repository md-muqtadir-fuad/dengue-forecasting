MODEL: Prophet — Grouped District-Level Monthly Benchmark

RUN MODE
- Final run: ENABLED
- Hyperparameter tuning: ENABLED
- Tuning type: light validation-only tuning
- Reason: this updated script now performs a small, defensible Prophet parameter search using validation performance
- Forecasting mode: rolling-origin monthly forecasting
- Model form: one Prophet model per district
- Aggregation: district forecasts are also summed to national totals for aggregate evaluation

INPUT
- Preprocessed file: ./data/raw/prime_dataset_model_input_with_purge.csv
- Split source: use existing split column from preprocessing
- Required rows in input file: train + val + purge + test must all be kept
- Required core columns: District, Date, split, Log_NoOfDenguePatients
- Optional but supported: Month-year
- Target: Log_NoOfDenguePatients

FEATURES / MODEL DESIGN
- Univariate target series per district
- No external preprocessed predictors used as regression features
- Month dummies: ENABLED by default
- Purpose of month dummies: represent monthly seasonality safely for monthly data
- Built-in yearly seasonality: disabled
- Forecast frequency: monthly start (MS)
- No scaling inside this Prophet pipeline

TRAINING
- Train split: used as initial history
- Validation split: used for tuning and rolling-origin validation evaluation
- Purge split: excluded from evaluation targets, but retained in the input file for correct chronology and target coverage
- Test split: final holdout only
- Horizon loop: 1 to 6 months by default
- Minimum history required before forecasting: 12 months by default
- Forecast intervals: ENABLED

TUNING
- Enabled: YES
- Search scope:
  - changepoint_prior_scale: [0.01, 0.05, 0.10, 0.30]
  - seasonality_prior_scale: [1.0, 5.0, 10.0, 20.0]
  - seasonality_mode: [additive, multiplicative]
- Search basis: validation MAE on district-level rolling forecasts
- Search granularity: district-specific selection
- Test split usage in tuning: NO
- Purge split usage in tuning: NO

OUTPUTS TO SAVE
- train predictions
- validation predictions
- test predictions
- district-level rolling forecast audit tables
- split manifest
- per-horizon metrics
- district-wise metrics
- national summary from summed district forecasts
- Prophet tuning results by district
- selected Prophet parameters by district
- final-train model diagnostics
- regressor coefficient tables
- train fitted residual tables
- ACF / PACF / Ljung-Box diagnostics outputs
- component plots
- figures
- run_config.json
- run_summary.txt
- zipped output archive

REPRODUCIBILITY
- Random seed: 42
- Save exact config used: YES
- Save split manifest: YES
- Save district-level tuning results: YES
- Keep purge rows in source file: YES, mandatory

DEPENDENCIES
- Install dependencies first:
  pip install -r requirements.txt

DEFAULT RUN
- Run with default input/output:
  python prophet.py

EXPLICIT OUTPUT FOLDER
- Run with a custom output folder:
  python prophet.py --output_dir Prophet/outputs

EXPLICIT INPUT FILE
- Run with a specific preprocessed file:
  python prophet.py --input ./data/raw/prime_dataset_model_input_with_purge.csv

OPTIONAL SWITCHES
- Enable tuning:
  python prophet.py --use_tuning

- Change maximum forecast horizon:
  python prophet.py --horizon 6

- Change minimum history before forecasting:
  python prophet.py --min_history_months 12

- Change interval width:
  python prophet.py --interval_width 0.95

- Change default changepoint prior scale:
  python prophet.py --changepoint_prior_scale 0.05

- Change default seasonality prior scale:
  python prophet.py --seasonality_prior_scale 10.0

- Change default seasonality mode:
  python prophet.py --seasonality_mode additive
  python prophet.py --seasonality_mode multiplicative

- Turn month dummies off:
  python prophet.py --no_month_dummies

- Change target column if needed:
  python prophet.py --target_col Log_NoOfDenguePatients

EXAMPLE FINAL RUN
- Recommended final run:
  python prophet.py --input ./data/raw/prime_dataset_model_input_with_purge.csv --output_dir Prophet/outputs --horizon 6 --min_history_months 12 --interval_width 0.95 --use_tuning

IMPORTANT NOTES
- In this updated version, tuning is worth documenting because the script now actually performs a real validation-based Prophet candidate search.
- This is still light tuning, not a large optimization pipeline.
- Do not remove purge rows from the modeling input file.
- Month dummies are ON by default; only use --no_month_dummies if you intentionally want to disable them.
- This is a grouped district-level benchmark, not a pooled global Prophet model.

sensitivity:

python ./Prophet/prop-sensitivity.py --input ./data/raw/prime_dataset_model_input_with_purge.csv --output_dir Prophet/outputs/sensitivity --fixed_tuning_csv ./Prophet/outputs/prophet_tuning_results.csv --ablation_names full,no_month_dummies