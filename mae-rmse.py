import matplotlib.pyplot as plt
import pandas as pd

# 1. DATA PREPARATION
data = {
    'Horizon': [1, 2, 3, 4, 5, 6],

    # RMSE Data
    'RMSE_SARIMAX':   [89.99, 142.63, 168.78, 222.72, 251.12, 251.22],
    'RMSE_RF':        [163.81, 184.07, 199.54, 159.42, 143.21, 176.77],
    'RMSE_XGBoost':   [104.87, 130.84, 175.30, 193.21, 158.45, 199.99],
    'RMSE_Attn_LSTM': [101.19, 122.68, 196.47, 236.00, 255.28, 238.48],
    'RMSE_TFT':       [173.89, 187.22, 190.44, 181.73, 161.09, 137.48],

    # MAE Data
    'MAE_SARIMAX':    [28.12, 44.78, 52.30, 62.15, 69.32, 71.81],
    'MAE_RF':         [42.42, 48.01, 49.84, 41.15, 44.79, 50.42],
    'MAE_XGBoost':    [32.75, 35.98, 50.68, 57.66, 52.89, 59.25],
    'MAE_Attn_LSTM':  [28.92, 33.65, 50.78, 60.36, 66.49, 68.00],
    'MAE_TFT':        [43.27, 47.88, 47.99, 44.33, 42.23, 39.44]
}

df = pd.DataFrame(data)

# 2. DEFINE STYLES
styles = {
    'SARIMAX':   {'color': 'black',   'ls': '-',  'marker': 'o', 'label': 'SARIMAX'},
    'RF':        {'color': '#444444', 'ls': '--', 'marker': '^', 'label': 'RF'},
    'XGBoost':   {'color': '#666666', 'ls': '-.', 'marker': 's', 'label': 'XGBoost'},
    'Attn-LSTM': {'color': 'black',   'ls': ':',  'marker': 'D', 'label': 'Attn-LSTM'},
    'TFT':       {'color': 'black',   'ls': '-',  'marker': 'x', 'label': 'TFT', 'linewidth': 2.5}
}

models = ['SARIMAX', 'RF', 'XGBoost', 'Attn-LSTM', 'TFT']

# --- GENERATE IMAGE 1: RMSE ---
plt.figure(figsize=(8, 6))

for model in models:
    plt.plot(
        df['Horizon'],
        df[f'RMSE_{model.replace("-", "_")}'],
        color=styles[model]['color'],
        linestyle=styles[model]['ls'],
        marker=styles[model]['marker'],
        label=styles[model]['label'],
        linewidth=styles[model].get('linewidth', 1.5),
        markersize=6
    )

#plt.title('Model Performance (RMSE) vs. Forecast Horizon', fontsize=14)
plt.ylabel('RMSE (Lower is Better)', fontsize=12)
plt.xlabel('Forecast Horizon (Months)', fontsize=12)
plt.xticks(df['Horizon'])
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=10, loc='upper left')
plt.tight_layout()
plt.savefig('rmse_plot_bw.pdf', bbox_inches='tight') # Save Image 1
plt.show()

# --- GENERATE IMAGE 2: MAE ---
plt.figure(figsize=(8, 6))

for model in models:
    plt.plot(
        df['Horizon'],
        df[f'MAE_{model.replace("-", "_")}'],
        color=styles[model]['color'],
        linestyle=styles[model]['ls'],
        marker=styles[model]['marker'],
        label=styles[model]['label'],
        linewidth=styles[model].get('linewidth', 1.5),
        markersize=6
    )

#plt.title('Model Performance (MAE) vs. Forecast Horizon', fontsize=14)
plt.ylabel('MAE (Lower is Better)', fontsize=12)
plt.xlabel('Forecast Horizon (Months)', fontsize=12)
plt.xticks(df['Horizon'])
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=10, loc='upper left')
plt.tight_layout()
plt.savefig('mae_plot_bw.pdf', bbox_inches='tight') # Save Image 2
plt.show()