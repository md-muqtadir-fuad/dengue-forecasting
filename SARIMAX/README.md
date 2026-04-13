MODEL: SARIMAX — Grouped District-Level Monthly Benchmark

RUN MODE
- Final run: ENABLED
- Hyperparameter tuning: NOT APPLICABLE in the tree/deep-learning sense
- Order search: ENABLED by default
- Reason: this script uses a small internal candidate-order comparison, not a broad tuning search
- Forecasting mode: rolling-origin monthly forecasting
- Model form: one SARIMAX model per district
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
- No exogenous regressors used by default
- Reason: avoids future-exog leakage in multi-step forecasting
- Default non-seasonal order: (1,1,1)
- Default seasonal order: (0,1,1,12)
- Default seasonal period: 12
- Minimum train history before forecasting: 24 months
- Forecasts are made on the log target, then converted back to count scale
- Forecast clipping on log scale: ENABLED (max_pred_log_clip = 12.0)

TRAINING
- Train split: used as the initial fitting history
- Validation split: included in rolling-origin evaluation
- Purge split: excluded from evaluation targets, but retained in the input file for correct chronology and target coverage
- Test split: final holdout only
- Horizon loop: 1 to 6 months by default
- Max optimizer iterations: 200 by default
- Residual diagnostics: ENABLED
- Forecast intervals: ENABLED

ORDER SEARCH
- Enabled: YES by default
- Search type: limited candidate specification comparison
- Selection basis: fit success / convergence / AIC / BIC
- Disable only if you want to force a fixed order:
  python sarimax.py --no_order_search

OUTPUTS TO SAVE
- train predictions
- validation predictions
- test predictions
- district-level rolling forecast audit tables
- split manifest
- row-audit output
- per-horizon metrics
- district-wise metrics
- regime-wise metrics
- national summary from summed district forecasts
- model parameter tables
- AIC / BIC candidate comparison tables
- residual diagnostic outputs
- ACF / PACF figures
- forecast interval outputs
- figures
- run_config.json
- run_summary.txt
- zipped output archive

REPRODUCIBILITY
- Random seed: 42
- Save exact config used: YES
- Save split manifest: YES
- Save district-level audit output: YES
- Keep purge rows in source file: YES, mandatory

DEPENDENCIES
- Install dependencies first:
  pip install -r requirements.txt

DEFAULT RUN
- Run with default input/output:
  python sarimax.py

EXPLICIT OUTPUT FOLDER
- Run with a custom output folder:
  python sarimax.py --output_dir SARIMAX/outputs

EXPLICIT INPUT FILE
- Run with a specific preprocessed file:
  python sarimax.py --input ./data/raw/prime_dataset_model_input_with_purge.csv

OPTIONAL SWITCHES
- Change maximum forecast horizon:
  python sarimax.py --horizon 6

- Change target column if needed:
  python sarimax.py --target_col Log_NoOfDenguePatients

- Change random seed:
  python sarimax.py --random_state 42

- Change non-seasonal order:
  python sarimax.py --p 1 --d 1 --q 1

- Change seasonal order:
  python sarimax.py --P 0 --D 1 --Q 1 --seasonal_period 12

- Change minimum training history:
  python sarimax.py --min_train_points 24

- Change optimizer iterations:
  python sarimax.py --maxiter 200

- Change diagnostic lags:
  python sarimax.py --diagnostics_lags 24

- Disable candidate order search and force the supplied order:
  python sarimax.py --no_order_search

EXAMPLE FINAL RUN
- Recommended final run:
  python sarimax.py --input ./data/raw/prime_dataset_model_input_with_purge.csv --output_dir SARIMAX/outputs --horizon 6

EXAMPLE FIXED-ORDER RUN
- Force one exact SARIMAX specification without candidate search:
  python sarimax.py --input ./data/raw/prime_dataset_model_input_with_purge.csv --output_dir SARIMAX/outputs --horizon 6 --p 1 --d 1 --q 1 --P 0 --D 1 --Q 1 --seasonal_period 12 --no_order_search

IMPORTANT NOTES
- This script is not using feature-based tuning like XGBoost / RF / deep models.
- The only search behavior here is the limited internal SARIMAX order comparison, which is ON by default.
- Do not remove purge rows from the modeling input file.
- This is a grouped district-level benchmark, not a pooled global SARIMAX model.

Package         Version
--------------- -------
cycler          0.12.1
Cython          0.29.17
joblib          1.0.1
kiwisolver      1.4.7  
matplotlib      3.3.4  
numpy           1.18.5 
pandas          1.1.5
patsy           0.5.1
pillow          10.4.0 
pip             19.2.3
pmdarima        1.8.0
pyparsing       3.1.4
python-dateutil 2.8.2
pytz            2020.5
scikit-learn    0.23.2
scipy           1.5.4
seaborn         0.11.1
setuptools      41.2.0
six             1.17.0
statsmodels     0.12.2
threadpoolctl   2.1.0  
urllib3         2.2.3

python 3.8.5

sensitivity:
python ./SARIMAX/sar-sen.py --base_script ./SARIMAX/sarimax_district.py --fixed_config_json ./SARIMAX/outputs/run_config.json --output_dir ./SARIMAX/outputs/sensitivity