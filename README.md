# Dengue Forecasting
A comprehensive machine learning project for dengue fever outbreak forecasting using multiple predictive models and advanced data preprocessing techniques.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Models](#models)
- [Usage](#usage)
- [Outputs](#outputs)
- [Results](#results)
- [License](#license)
- [Appendix](#appendix)

## Overview

This project implements multiple machine learning models to forecast dengue cases at different time horizons (1-6 months ahead). The models are trained on historical dengue data and include capabilities for:

- Multi-horizon forecasting (1-6 months ahead)
- District-level predictions
- Hyperparameter tuning and optimization
- Comprehensive performance evaluation
- Feature importance analysis
- Outbreak classification and detection

## Project Structure

```bash
dengue-forecasting/                  # Root directory
├── .gitignore
├── README.md
├── LICENSE
├── synthetic-dataset.py
├── district_panel.py
├── dis-den-rep.py
├── dis-selected.py
├── mae-rmse.py
├── preprocessing.ipynb
├── format_utils.py
│
├── data/
│   ├── dataset/
│   │   └── synthetic-dataset.csv
│   └── district/
│       ├── bgd_admin0.geojson
│       ├── bgd_admin1.geojson
│       ├── bgd_admin2.geojson
│       ├── bgd_admin3.geojson
│       ├── bgd_admincapitals.geojson
│       ├── bgd_adminlines.geojson
│       └── bgd_adminpoints.geojson
│
├── CATBoost/
│   ├── README.md
│   ├── cat-sen.py
│   ├── catboost_district.py
│   └── outputs/
│       ├── catboost_best_params.json
│       ├── catboost_per_horizon.csv
│       ├── catboost_split_manifest.csv
│       ├── catboost_summary.csv
│       ├── run_config.json
│       ├── run_summary.txt
│       └── sensitivity/...
│
├── XGBoost/
│   ├── README.md
│   ├── xgb-sen.py
│   ├── xgboost_district_pool.py
│   └── outputs/...
│
├── RF/
│   ├── README.md
│   ├── rf-sen.py
│   ├── random_forest_district.py
│   └── outputs/...
│
├── SVR/
│   ├── README.md
│   ├── SVR-sen.py
│   ├── svr_district.py
│   └── outputs/...
│
├── MLR/
│   ├── README.md
│   ├── mlr-sen.py
│   ├── mlr_district.py
│   └── outputs/...
│
├── Prophet/
│   ├── README.md
│   ├── prophet_district.py
│   ├── prop-sensitivity.py
│   ├── pro_sensitivity_visualization.py
│   └── outputs/...
│
├── SARIMAX/
│   ├── README.md
│   ├── sar-sen.py
│   ├── sarimax_district.py
│   ├── requirements.txt
│   └── outputs/...
│
├── TFT/
│   ├── readme.md
│   ├── tft_district.py
│   ├── tft-sen.py
│   ├── sensitivity_analysis.py
│   ├── requirements.txt
│   └── outputs/...
│
├── attn-LSTM/
│   ├── README.md
│   ├── alstm-sen.py
│   ├── attention_lstm_district.py
│   └── outputs/...
│
├── stacked-LSTM/
│   ├── README.md
│   ├── slstm-sen.py
│   ├── stacked_lstm_district.py
│   └── outputs/...
│
├── Causality/
│   ├── README.md
│   ├── PCMCI.py
│   ├── granger-casuality.py
│   └── outputs/...
│
└── significance_test/
    ├── main_script.py
    └── outputs/
        ├── README.txt
        ├── table_dm_absolute_error_summary.csv
        └── table_dm_squared_error_summary.csv
```

## Installation

### Prerequisites

- Python 3.8+
- pip or conda package manager
- Virtual environment (recommended)

### Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd dengue-forecasting
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   
   For XGBoost model:
   ```bash
   pip install -r XGBoost/requirements.txt
   ```

   Or install core dependencies manually:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn xgboost shap
   ```

## Dataset

The project uses dengue case data stored in CSV format:

- **Raw Data:** `data/raw/prime-dataset.csv`
- **Processed Features:** `data/raw/prime_dataset_final_selected_features.csv`
- **Processed Output:** `data/output/`

The dataset includes:
- Historical dengue cases
- Selected features via K-best feature selection
- District-level information
- Time-series data for multiple horizons

### Data Preprocessing

Run the preprocessing notebook to prepare data:
```bash
jupyter notebook preprocessing.ipynb
```

## Models

The project includes seven distinct modeling approaches:

| Model | Type | Implementation |
|-------|------|-----------------|
| **Random Forest** | Ensemble tree-based | `RF/random_forest_district.py` |
| **XGBoost** | Gradient boosting | `XGBoost/xgboost_district_pool.py` |
| **SARIMAX** | Time series (ARIMA) | `SARIMAX/sarimax_district.py` |
| **SVR** | Support Vector Regression | `SVR/svr_district.py` |
| **MLR** | Multiple Linear Regression | `MLR/mlr_district.py` |
| **Attention LSTM** | Deep learning (RNN) | `attn-LSTM/attention_lstm_district.py` |
| **Stacked LSTM** | Deep learning (RNN) | `stacked-LSTM/stacked_lstm_district.py` |
| **TFT** | Transformer-based | `TFT/tft_district.py` |

Key features across models:
- Hyperparameter tuning and optimization
- SHAP interpretation (XGBoost)
- Multi-horizon predictions
- District-level analysis

## Usage

### Running Models

All models follow a similar CLI structure:

```bash
python <MODEL_DIR>/<MODEL_SCRIPT>.py [OPTIONS]
```

**Common Parameters:**
- `--horizon`: Forecast horizon in months (1-6, default varies by model)
- `--test_frac`: Test set fraction (default: 0.20)
- `--val_months`: Validation period in months (default: 12)
- `--purge_months`: Data purge period in months (default: 12)
- `--use_tuning`: Enable hyperparameter tuning
- `--tuning_iter`: Number of tuning iterations
- `--use_district_ohe`: Use one-hot encoding for districts
- `--output_dir`: Custom output directory

### Examples

**Random Forest (basic):**
```bash
python RF/random_forest_district.py
```

**XGBoost (with tuning):**
```bash
python XGBoost/xgboost_district_pool.py --use_tuning --tuning_iter 200
```

**Custom parameters:**
```bash
python RF/random_forest_district.py --horizon 6 --test_frac 0.20 --use_tuning
```

## Outputs

Each model generates the following outputs in its respective `outputs/` directory:

**Model Artifacts:**
- Model files (in native format, e.g., `.joblib` for RF, `.json` for XGBoost)
- Best hyperparameters (`*_best_params.json`)
- Feature importance (`*_feature_importance_gain.csv`)

**Predictions:**
- `*_train_predictions_long.csv` - Training set predictions
- `*_val_predictions_long.csv` - Validation set predictions
- `*_test_predictions_long.csv` - Test set predictions
- `*_test_residuals_long.csv` - Test set residuals

**Evaluation Metrics:**
- `*_summary.csv` - Overall model summary
- `*_metrics_by_district.csv` - Performance by district
- `*_metrics_by_district_horizon.csv` - Performance by district and horizon
- `*_per_horizon.csv` - Metrics by forecast horizon
- `*_regimewise_metrics.csv` - Metrics by regime (e.g., high/low cases)
- `*_outbreak_classification_metrics.csv` - Outbreak detection metrics

**Interpretation:**
- `shap_aggregate_importance.csv` - Aggregated SHAP feature importance
- `run_summary.txt` - Detailed run information
- `run_config.json` - Configuration used for the run

## Results

### Performance Metrics

Models are evaluated using:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)
- **Outbreak Classification Metrics** (Precision, Recall, F1-Score)

### Model Comparison

Results by model and horizon are available in the respective `outputs/` directories. Summary comparisons are generated during runs.

### Key Findings

- XGBoost and Random Forest show competitive performance across horizons
- Performance degrades with longer forecast horizons
- District-level predictions vary significantly
- Low-case regimes are challenging to forecast

## License

See [LICENSE](LICENSE) file for details.

---

## Appendix

### Data Generation

To generate synthetic datasets:
```bash
python synthetic-dataset.py --input_csv ./data/raw/prime_dataset_model_input_with_purge.csv --output_csv ./data/dataset/synthetic-dataset.csv --seed 42
```

### Environment Configuration

The project supports multiple Python environments based on model requirements:

- **Main Environment:** Python 3.11.9 (RF, XGBoost, CATBoost, Prophet, etc.)
- **TFT-Specific:** Python 3.8.0 (requires PyTorch Lightning)
- **SARIMAX-Specific:** Python 3.8.0 (requires pmdarima, statsmodels)
- **LSTM Models:** Python 3.11.9 (TensorFlow/Keras)

For complete dependency specifications, refer to individual `requirements.txt` files in each model directory.

### Package Versions

**Main Environment: Python 3.11.9**

```bash
array-api-compat    1.14.0
catboost            1.2.10
certifi             2026.2.25
cloudpickle         3.1.2
cmdstanpy           1.3.0
colorama            0.4.6
contourpy           1.3.3
cycler              0.12.1
dcor                0.7
et_xmlfile          2.0.0
fonttools           4.62.1
geopandas           1.1.3
graphviz            0.21
holidays            0.94
importlib_resources 6.5.2
joblib              1.5.3
kiwisolver          1.5.0
llvmlite            0.46.0
matplotlib          3.10.8
narwhals            2.19.0
networkx            3.6.1
numba               0.64.0
numpy               2.4.3
openpyxl            3.1.5
packaging           26.0
pandas              3.0.1
patsy               1.0.2
pillow              12.1.1
plotly              6.7.0
prophet             1.3.0
pyogrio             0.12.1
pyparsing           3.3.2
pyproj              3.7.2
python-dateutil     2.9.0.post0
regex               2026.2.28
scikit-learn        1.8.0
scipy               1.17.1
seaborn             0.13.2
setuptools          65.5.0
shap                0.51.0
shapely             2.1.2
six                 1.17.0
slicer              0.0.8
stanio              0.5.1
statsmodels         0.14.6
threadpoolctl       3.6.0
tigramite           5.2.10.1
tqdm                4.67.3
typing_extensions   4.15.0
tzdata              2025.3
xgboost             3.2.0
```

**TFT-Specific: Python 3.8.0**

```bash
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
```

**SARIMAX-Specific: Python 3.8.0**

```bash
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
```

**LSTM Models: Python 3.11.9**

```bash
absl-py            2.4.0
astunparse         1.6.3
certifi            2026.2.25
charset-normalizer 3.4.7
contourpy          1.3.3
cycler             0.12.1
flatbuffers        25.12.19
fonttools          4.62.1
gast               0.7.0
google-pasta       0.2.0
grpcio             1.80.0
h5py               3.14.0
idna               3.11
joblib             1.5.3
keras              3.14.0
kiwisolver         1.5.0
libclang           18.1.1
markdown-it-py     4.0.0
matplotlib         3.10.8
mdurl              0.1.2
namex              0.1.0
numpy              2.4.4
opt_einsum         3.4.0
optree             0.19.0
packaging          26.0
pandas             3.0.2
pillow             12.2.0
pip                24.0
protobuf           7.34.1
Pygments           2.20.0
pyparsing          3.3.2
python-dateutil    2.9.0.post0
requests           2.33.1
rich               14.3.4
scikit-learn       1.8.0
scipy              1.17.1
seaborn            0.13.2
setuptools         65.5.0
six                1.17.0
tensorflow         2.21.0
termcolor          3.3.0
threadpoolctl      3.6.0
typing_extensions  4.15.0
tzdata              2026.1
urllib3            2.6.3
wheel              0.46.3
wrapt              2.1.2
```