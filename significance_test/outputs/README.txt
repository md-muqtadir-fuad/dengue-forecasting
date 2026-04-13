Final significance testing outputs

Protocol used
-------------
Reference model: Prophet
Competitors: Attention-LSTM, Stacked LSTM, XGBoost, TFT

Two significance families:
1. squared error  -> RMSE-oriented
2. absolute error -> MAE-oriented

Testing unit:
- common District-TargetDate-horizon rows, aligned pairwise
- losses averaged across districts within each TargetDate
- DM test with HLN small-sample correction
- Newey-West lag = horizon - 1
- Holm correction within each horizon and loss family

Important note
--------------
If any model has fewer available rows than the others, this script keeps the
comparison valid by using only the pairwise common overlap for that comparison.
See:
- coverage_audit_by_model_horizon.csv
- pairwise_overlap_vs_prophet.csv
- all_models_common_intersection_audit.csv
