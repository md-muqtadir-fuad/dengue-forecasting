# Granger Causality Test Script

This script performs **conditional linear Granger causality testing** on a preprocessed dengue dataset.

## Script

`granger-casuality.py`

## Purpose

The script tests whether one or more predictor time series **Granger-cause** a dengue target series under a linear framework.

It supports:

- **National mode**: one aggregated time series
- **District mode**: separate tests per district, plus combined summaries
- **Conditional testing**: include control variables such as `denv2` and `denv3`
- **Lag search**: test lags from `1` to `max_lag`
- **Exported results** for manuscript tables and diagnostics

## Typical Use Cases

### 1. National conditional Granger test
Example: test whether temperature and rainfall linearly Granger-cause dengue while conditioning on DENV-2 and DENV-3.

### 2. District-wise Granger test
Run the same test separately for each district, then summarize:
- best lag
- best p-value
- significance conclusion
- combined district-level evidence

---

# Input Requirements

The script expects a **CSV file** with at least:

- a date column
- a target column
- one or more predictor columns
- optional control columns
- optionally a district column for district-wise mode

## Required columns

### For national mode
- `Date`
- target column, for example: `Log_NoOfDenguePatients`
- predictor columns, for example: `AvgTemp`, `Rainfall`
- optional controls, for example: `denv2`, `denv3`

### For district mode
- all of the above, plus:
- `District`

## Important note

If your current selected-feature file only contains variables such as:

- `AvgTemp_lag_3`
- `Rainfall_lag_2`
- `denv1_lag_1`

then the script will still run, but that will **not exactly reproduce** a manuscript table framed as:

- `Temp → Dengue`
- `Rain → Dengue`
- conditioned on `DENV-2`, `DENV-3`

For that manuscript setup, use a richer preprocessed dataset containing current/raw versions of those variables.

---

# Method Summary

For each predictor:

1. the script constructs a time-aligned dataset
2. applies optional conditioning variables
3. tests lags `1` through `max_lag`
4. records the p-value for each lag
5. identifies the strongest lag based on the smallest valid p-value
6. concludes significance at the chosen alpha level

In district mode, the script also produces:
- district-level results
- combined summary tables
- Fisher-style combined evidence across districts, if enabled in the script

---

# Command-Line Arguments

## Core arguments

- `--input`  
  Path to input CSV

- `--mode`  
  Either:
  - `national`
  - `district`

- `--target_col`  
  Name of dengue target column

- `--predictors`  
  One or more predictor columns

- `--controls`  
  Optional conditioning variables

- `--max_lag`  
  Maximum lag to test

## Optional target handling

- `--target_is_log1p`  
  Use this if the target column is already log1p-transformed

## Optional panel arguments

- `--district_col`  
  Default district column name if using district mode

- `--date_col`  
  Default date column name

## Optional significance setting

- `--alpha`  
  Significance threshold, usually `0.05`

---

# Example Commands

## A. National analysis matching manuscript intent

```bash
python granger-casuality.py \
  --input preprocessed_dataset \
  --mode national \
  --date_col Date \
  --target_col Log_NoOfDenguePatients \
  --target_is_log1p \
  --predictors AvgTemp Rainfall \
  --controls denv2 denv3 \
  --max_lag 6 \
  --alpha 0.05
```
## B. District-wise analysis
```bash
python granger-casuality.py \
  --input preprocessed_dataset.csv \
  --mode district \
  --district_col District \
  --date_col Date \
  --target_col Log_NoOfDenguePatients \
  --target_is_log1p \
  --predictors AvgTemp Rainfall \
  --controls denv2 denv3 \
  --max_lag 6 \
  --alpha 0.05

```
## C. Run using currently selected-feature style

```bash
python granger-casuality.py \
  --input preprocessed_dataset \
  --mode district \
  --district_col District \
  --date_col Date \
  --target_col Log_NoOfDenguePatients \
  --target_is_log1p \
  --predictors AvgTemp_lag_3 Rainfall_lag_2 \
  --max_lag 6 \
  --alpha 0.05
```


