Outputs
=======
- run_config.json: exact configuration used
- run_log.json: basic run metadata
- variable_lag_metadata.csv: lag metadata for target / predictors / controls
- lag_interpretation_note.txt: how to interpret effective total lag
- granger_summary_for_manuscript.csv: manuscript-friendly summary table
- national_series_used.csv: national aggregated monthly series (national mode only)
- granger_national_<predictor>.csv: per-lag national results (national mode only)
- district_tests/granger_<DISTRICT>_<predictor>.csv: per-lag district results (district mode only)
- granger_district_summary_<predictor>.csv: best lag by district (district mode only)
- granger_fisher_by_lag_<predictor>.csv: Fisher-combined p-value by lag across districts (district mode only)
- granger_district_summary_all.csv: merged district summaries (district mode only)

Important note
==============
If already-lagged predictors such as AvgTemp_lag_3 are provided,
the script reports both:
- Strongest Granger Lag
- Effective Total Lag

Interpretation rule:
Effective Total Lag = Source Input Offset + Granger Lag - Target Input Offset