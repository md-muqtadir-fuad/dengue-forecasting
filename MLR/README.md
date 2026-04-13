MODEL: Multiple Linear Regression (MLR) — Direct Multi-Horizon District-Panel

RUN MODE
- Final run: ENABLED
- Hyperparameter tuning: NOT APPLICABLE
- Reason: this script uses classical regression, not a tuning-based search pipeline
- Forecasting mode: direct multi-horizon
- Model form: pooled district-level monthly panel
- One separate regression is fit for each forecast horizon

INPUT
- Preprocessed file: prime_dataset_final_selected_features_with_splits.csv
- Split source: use existing split column from preprocessing
- Required core columns: District, Date, Month-year, split, Log_NoOfDenguePatients
- Required rows in input file: train + val + test
- Purge rows: optional in this script, but strongly preferred for better target coverage at larger horizons
- Target: Log_NoOfDenguePatients

FEATURES
- Use numeric predictors from the preprocessing pipeline
- Month encoding from preprocessing: Month_sin + Month_cos
- No global scaling inside this MLR pipeline
- District one-hot encoding: DISABLED by default
- Reason district OHE is off by default: static district covariates such as PopulationDensity become perfectly collinear with district fixed effects
- If district OHE is enabled, PopulationDensity is automatically dropped to avoid exact multicollinearity

TRAINING
- Train split: used for fitting
- Validation split: used for validation reporting
- Purge split: not required for the script to run, but improves horizon coverage if present
- Test split: final holdout only
- Horizon loop: 1 to 6 months by default
- Fitting engine: statsmodels OLS
- Robust covariance: HC3 by default

TUNING
- Enabled: NO
- Search method: N/A
- Best params source: N/A
- Main configurable statistical option: robust covariance type

OUTPUTS TO SAVE
- train predictions
- validation predictions
- test predictions
- residual / error tables
- split manifest
- target coverage manifest
- per-horizon metrics
- district-wise metrics
- district × horizon metrics
- regime-wise metrics
- outbreak classification metrics
- coefficient tables for each horizon
- standardized coefficient tables
- aggregate standardized coefficient summary
- VIF tables
- horizon-wise diagnostics
- model summary text files
- figures
- mlr_data_audit.json
- run_config.json
- run_summary.txt
- zipped output archive

REPRODUCIBILITY
- Random seed: 42
- Save exact config used: YES
- Save split manifest: YES
- Save target coverage report: YES
- Save diagnostic statistics: YES

DEPENDENCIES
- Install dependencies first:
  pip install -r requirements.txt

DEFAULT RUN
- Run with default input/output:
  python mlr.py

EXPLICIT OUTPUT FOLDER
- Run with a custom output folder:
  python mlr.py --output_dir mlr_run_outputs

EXPLICIT INPUT FILE
- Run with a specific preprocessed file:
  python mlr.py --input prime_dataset_final_selected_features_with_splits.csv

OPTIONAL SWITCHES
- Change maximum forecast horizon:
  python mlr.py --horizon 6

- Enable district one-hot encoding:
  python mlr.py --use_district_ohe

- Change target column if needed:
  python mlr.py --target_col Log_NoOfDenguePatients

- Change random seed:
  python mlr.py --random_state 42

- Change robust covariance type:
  python mlr.py --robust_cov_type HC3
  python mlr.py --robust_cov_type HC1

- Change outbreak quantile threshold:
  python mlr.py --outbreak_quantile 0.90

EXAMPLE FINAL RUN
- Recommended final run:
  python mlr.py --input prime_dataset_final_selected_features_with_splits.csv --output_dir mlr_run_outputs --horizon 6 --robust_cov_type HC3

IMPORTANT NOTES
- This script does not have a tuning mode.
- Purge rows are not strictly mandatory here, but if they are missing, early target coverage at larger horizons can shrink.
- District one-hot encoding is OFF by default for a good reason; do not enable it unless you explicitly want district fixed effects and accept the automatic drop of PopulationDensity.

Sensitivity

python MLR/mlr-sen.py --robust_cov_type HC3 --outbreak_quantile 0.9 --ablation_names full,no_climate,no_serotype,no_temporal,no_population_density