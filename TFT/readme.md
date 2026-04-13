Package             Version
------------------- ------------
aiohappyeyeballs    2.4.4
aiohttp             3.10.11
aiosignal           1.3.1
async-timeout       5.0.1
attrs               25.3.0
colorama            0.4.6
contourpy           1.1.1
cycler              0.12.1
filelock            3.16.1
fonttools           4.57.0
frozenlist          1.5.0
fsspec              2025.3.0
idna                3.11
importlib_resources 6.4.5
Jinja2              3.1.6
joblib              1.4.2
kiwisolver          1.4.7
lightning           2.3.3
lightning-utilities 0.11.9
MarkupSafe          2.1.5
matplotlib          3.7.5
mpmath              1.3.0
multidict           6.1.0
networkx            3.1
numpy               1.24.4
packaging           24.2
pandas              2.0.3
pillow              10.4.0
pip                 25.0.1
propcache           0.2.0
pyparsing           3.1.4
python-dateutil     2.9.0.post0
pytorch-forecasting 1.1.1
pytorch-lightning   2.4.0
pytz                2026.1.post1
PyYAML              6.0.3
scikit-learn        1.3.2
scipy               1.10.1
seaborn             0.12.2
setuptools          41.2.0
six                 1.17.0
sympy               1.13.3
threadpoolctl       3.5.0
torch               2.4.1
torchmetrics        1.5.2
tqdm                4.67.3
typing_extensions   4.13.2
tzdata              2026.1
yarl                1.15.2
zipp                3.20.2

pyhton 3.8.5

MODEL: Temporal Fusion Transformer (TFT) — Direct Multi-Horizon District-Panel

RUN MODE
- Final run: ENABLED
- Hyperparameter tuning: DISABLED in final run
- Reason: best settings already fixed from prior tuning
- Forecasting mode: direct multi-horizon
- Model form: pooled district-level monthly panel
- One separate TFT model is fit for each forecast horizon, unless the script internally manages all horizons in one configured run

INPUT
- Preprocessed file: ./data/raw/prime_dataset_model_input_with_purge.csv
- Split source: use existing split column from preprocessing
- Required rows in input file: train + val + purge + test must all be kept
- Required core columns: District, Date, Month-year, split, Log_NoOfDenguePatients
- Target: Log_NoOfDenguePatients

FEATURES
- Use predictors from the preprocessing pipeline
- Month encoding from preprocessing: Month_sin + Month_cos
- Horizon-specific seasonal features added inside model when applicable: TargetMonth_sin + TargetMonth_cos
- District encoding: kept as static series identifier / encoded feature according to script implementation
- Lookback / encoder length: use script default unless explicitly overridden
- Model-specific scaling / normalization: ENABLED inside the script on training data only
- No separate global preprocessing scaling file needed

TRAINING
- Train split: used for fitting
- Validation split: used for early stopping / checkpoint selection / model selection
- Purge split: excluded from fitting and excluded from evaluation targets, but retained in the input file for correct horizon construction
- Test split: final holdout only
- Horizon loop: use script default, typically 1 to 6 months
- Early stopping: ENABLED
- Best-checkpoint loading: ENABLED if supported in the script
- Gradient clipping / regularization: use script defaults unless explicitly changed

TUNING
- Enabled: NO in final run
- Search method: parameter sampling / trial search only when explicitly enabled
- Best params source: fixed final configuration from prior tuning
- Optional tuning mode remains available only for separate tuning experiments

OUTPUTS TO SAVE
- fitted model / checkpoint files
- scaler / normalizer artifacts if the script saves them
- train predictions
- validation predictions
- test predictions
- residual / error tables
- per-horizon metrics
- district-wise metrics
- district × horizon metrics
- outbreak classification metrics if implemented
- split manifest
- row-audit output
- TFT interpretation outputs / variable importance / attention summaries if implemented
- figures
- best-params / best-config record
- run_config.json
- run_summary.txt
- zipped output archive

REPRODUCIBILITY
- Random seed: 42
- Save exact config used: YES
- Save split manifest: YES
- Save best-epoch / checkpoint record: YES
- Keep purge rows in source file: YES, mandatory

DEPENDENCIES
- Install dependencies first:
  pip install -r requirements.txt

DEFAULT RUN
- Run with default input/output:
  python tft.py

EXPLICIT OUTPUT FOLDER
- Run with a custom output folder:
  python tft.py --output_dir TFT/outputs

EXPLICIT INPUT FILE
- Run with a specific preprocessed file:
  python tft.py --input ./data/raw/prime_dataset_model_input_with_purge.csv

OPTIONAL SWITCHES
- Enable tuning:
  python tft.py --use_tuning

- Change maximum forecast horizon:
  python tft.py --horizon 6

- Change lookback / encoder length:
  python tft.py --lookback 6
  python tft.py --max_encoder_length 6

- Change epochs:
  python tft.py --epochs 300

- Change batch size:
  python tft.py --batch_size 32

- Change hidden size:
  python tft.py --hidden_size 32

- Change attention heads:
  python tft.py --attention_head_size 4

- Change dropout:
  python tft.py --dropout 0.1

- Change learning rate:
  python tft.py --learning_rate 0.001

- Change target column if needed:
  python tft.py --target_col Log_NoOfDenguePatients

- Change random seed:
  python tft.py --random_state 42

- Control district encoding if your script supports it:
  python tft.py --use_district_ohe
  python tft.py --no_district_ohe

EXAMPLE FINAL RUN
- Recommended final run:
  python tft.py --input ./data/raw/prime_dataset_model_input_with_purge.csv --output_dir TFT/outputs --horizon 6 --epochs 300 --batch_size 32

IMPORTANT NOTES
- Do not remove purge rows from the modeling input file.
- Do not use old split arguments like:
  --test_frac
  --val_months
  --purge_months
  because this version should take split definitions directly from the preprocessing file.
- Scaling / normalization inside the model is appropriate for TFT, but it must be fit on training data only.
- If the script has separate defaults in Config and CLI for any parameter, document the final run with that parameter written explicitly.

Sensitiviy:
python ./TFT/tft-sen.py --output_dir ./TFT/outputs/sensitivity --fixed_config_json ./TFT/outputs/tft_best_config.json --seed_list 42 --ablation_names full,no_climate,no_serotype,no_temporal,no_population_density