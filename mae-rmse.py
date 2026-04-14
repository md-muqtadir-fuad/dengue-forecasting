import matplotlib.pyplot as plt
from pathlib import Path

# Horizons
horizons = [1, 2, 3, 4, 5, 6]

# Hard-coded RMSE values
rmse = {
    "SARIMAX": [3716.948, 8244.282, 7917.783, 7294.117, 5963.015, 4510.714],
    "Prophet": [3930.494, 3952.613, 3973.281, 3988.867, 3999.210, 4004.231],
    "MLR": [4081.671, 4076.681, 4080.298, 4088.084, 4084.170, 4089.436],
    "SVR": [4111.986, 4112.013, 4112.013, 4111.995, 4112.013, 4112.013],
    "RF": [4111.079, 4110.786, 4110.825, 4111.033, 4110.891, 4110.035],
    "XGBoost": [4109.832, 4109.008, 4110.385, 4110.268, 4110.337, 4110.374],
    "CatBoost": [4109.979, 4109.428, 4110.558, 4110.407, 4110.464, 4109.933],
    "Stacked LSTM": [4094.063, 4092.768, 4103.847, 4081.426, 4111.850, 4111.667],
    "Attention-LSTM": [4009.819, 4057.773, 4105.483, 4111.476, 4110.805, 4104.887],
    "TFT": [4111.249, 4111.043, 4110.979, 4110.983, 4110.987, 4110.971],
}

# Hard-coded MAE values
mae = {
    "SARIMAX": [952.653, 1735.993, 1882.174, 1894.360, 1752.683, 1510.294],
    "Prophet": [1226.973, 1233.684, 1239.919, 1245.062, 1249.041, 1251.955],
    "MLR": [1264.802, 1263.477, 1266.597, 1271.018, 1268.560, 1270.226],
    "SVR": [1287.381, 1287.449, 1287.449, 1287.404, 1287.449, 1287.449],
    "RF": [1286.820, 1286.620, 1286.569, 1286.720, 1286.536, 1285.972],
    "XGBoost": [1286.252, 1285.873, 1286.190, 1286.290, 1286.177, 1286.314],
    "CatBoost": [1286.284, 1285.849, 1286.446, 1286.291, 1286.415, 1286.194],
    "Stacked LSTM": [1278.789, 1243.645, 1280.043, 1246.009, 1287.090, 1287.251],
    "Attention-LSTM": [1250.743, 1260.844, 1281.643, 1286.531, 1286.584, 1284.254],
    "TFT": [1287.074, 1286.842, 1286.623, 1286.529, 1286.513, 1286.514],
}

# Consistent styles
model_styles = {
    "SARIMAX": {"color": "#1f77b4", "marker": "o"},
    "Prophet": {"color": "#ff7f0e", "marker": "s"},
    "MLR": {"color": "#2ca02c", "marker": "^"},
    "SVR": {"color": "#d62728", "marker": "D"},
    "RF": {"color": "#9467bd", "marker": "v"},
    "XGBoost": {"color": "#8c564b", "marker": "P"},
    "CatBoost": {"color": "#e377c2", "marker": "X"},
    "Stacked LSTM": {"color": "#7f7f7f", "marker": "<"},
    "Attention-LSTM": {"color": "#bcbd22", "marker": ">"},
    "TFT": {"color": "#17becf", "marker": "*"},
}

sarimax_only = ["SARIMAX"]
other_models = [m for m in rmse.keys() if m != "SARIMAX"]

# Output path
out_dir = Path("./data/output")
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "horizon_performance_split_panel.pdf"

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)

# RMSE - SARIMAX
for model in sarimax_only:
    style = model_styles[model]
    axes[0, 0].plot(
        horizons, rmse[model],
        label=model,
        color=style["color"],
        marker=style["marker"],
        linewidth=2.2,
        markersize=7
    )
axes[0, 0].set_title("RMSE by Forecast Horizon: SARIMAX")
axes[0, 0].set_ylabel("RMSE")
axes[0, 0].set_xticks(horizons)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend(fontsize=9)

# RMSE - Others
for model in other_models:
    style = model_styles[model]
    axes[0, 1].plot(
        horizons, rmse[model],
        label=model,
        color=style["color"],
        marker=style["marker"],
        linewidth=2,
        markersize=6
    )
axes[0, 1].set_title("RMSE by Forecast Horizon: Other Models")
axes[0, 1].set_xticks(horizons)
axes[0, 1].grid(True, alpha=0.3)

# MAE - SARIMAX
for model in sarimax_only:
    style = model_styles[model]
    axes[1, 0].plot(
        horizons, mae[model],
        label=model,
        color=style["color"],
        marker=style["marker"],
        linewidth=2.2,
        markersize=7
    )
axes[1, 0].set_title("MAE by Forecast Horizon: SARIMAX")
axes[1, 0].set_xlabel("Horizon (months)")
axes[1, 0].set_ylabel("MAE")
axes[1, 0].set_xticks(horizons)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend(fontsize=9)

# MAE - Others
for model in other_models:
    style = model_styles[model]
    axes[1, 1].plot(
        horizons, mae[model],
        label=model,
        color=style["color"],
        marker=style["marker"],
        linewidth=2,
        markersize=6
    )
axes[1, 1].set_title("MAE by Forecast Horizon: Other Models")
axes[1, 1].set_xlabel("Horizon (months)")
axes[1, 1].set_xticks(horizons)
axes[1, 1].grid(True, alpha=0.3)

# Shared legend for non-SARIMAX models
handles, labels = axes[0, 1].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="lower center",
    ncol=5,
    fontsize=9,
    frameon=True,
    bbox_to_anchor=(0.5, 0.01)
)

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(out_file, format="pdf", dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved to: {out_file.resolve()}")