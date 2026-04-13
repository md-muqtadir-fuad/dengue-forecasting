import os
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# =========================================
# FILE PATHS
# =========================================
geojson_path = "./data/district/bgd_admin2.geojson"
csv_path = "./data/raw/prime_dataset_model_input_with_purge.csv"   # change if needed
output_pdf = "./data/output/dengue_aug_2023_district_log_map.pdf"

os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

# =========================================
# NAME NORMALIZATION
# =========================================
def normalize_name(name):
    if pd.isna(name):
        return None

    name = str(name).strip().lower()
    name = name.replace("&", "and")
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name)

    replacements = {
        "barisal": "barishal",
        "jessore": "jashore",
        "chittagong": "chattogram",
        "coxs bazar": "coxs bazar",
        "coxs bazaar": "coxs bazar",
    }
    return replacements.get(name, name)

# =========================================
# READ GEOJSON
# =========================================
gdf = gpd.read_file(geojson_path).copy()
name_col = "adm2_name"
gdf["district_key"] = gdf[name_col].apply(normalize_name)

# =========================================
# READ CSV AND FILTER AUG 2023
# =========================================
df = pd.read_csv(csv_path).copy()
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["district_key"] = df["District"].apply(normalize_name)

aug_2023 = df[(df["Date"].dt.year == 2023) & (df["Date"].dt.month == 8)].copy()

if aug_2023.empty:
    raise ValueError("No rows found for August 2023 in the CSV.")

# =========================================
# DISTRICT-WISE SUM, THEN LOG SCALE
# Proper way:
#   raw_cases = exp(log_cases) - 1
#   summed_raw = sum(raw_cases) by district
#   mapped_log = log(1 + summed_raw)
# =========================================
aug_2023["raw_cases"] = np.expm1(aug_2023["Log_NoOfDenguePatients"])

district_aug = (
    aug_2023.groupby("district_key", as_index=False)["raw_cases"]
    .sum()
    .rename(columns={"raw_cases": "aug_2023_cases"})
)

district_aug["log_cases"] = np.log1p(district_aug["aug_2023_cases"])

# =========================================
# MERGE WITH MAP
# =========================================
plot_gdf = gdf.merge(district_aug, on="district_key", how="left")
plot_gdf["selected"] = plot_gdf["log_cases"].notna()

# =========================================
# CHECK MATCHES
# =========================================
matched = sorted(plot_gdf.loc[plot_gdf["selected"], name_col].tolist())
print("Matched districts:", matched)

missing = sorted(set(district_aug["district_key"]) - set(plot_gdf["district_key"]))
if missing:
    print("Warning: unmatched district keys:", missing)

# =========================================
# LABELS
# =========================================
label_map = {
    "rajshahi": "Rajshahi",
    "mymensingh": "Mymensingh",
    "sylhet": "Sylhet",
    "dhaka": "Dhaka",
    "faridpur": "Faridpur",
    "jashore": "Jessore",
    "khulna": "Khulna",
    "barishal": "Barisal",
    "bhola": "Bhola",
    "chattogram": "Chattogram",
    "coxs bazar": "Cox's Bazar",
}

label_offsets = {
    "rajshahi": (-0.12, 0.02),
    "mymensingh": (0.00, 0.05),
    "sylhet": (0.06, 0.03),
    "dhaka": (0.00, 0.00),
    "faridpur": (0.00, -0.05),
    "jashore": (-0.05, 0.00),
    "khulna": (0.00, -0.10),
    "barishal": (0.02, -0.02),
    "bhola": (0.00, -0.08),
    "chattogram": (0.05, 0.00),
    "coxs bazar": (0.03, -0.12),
}

# =========================================
# PLOT
# =========================================
fig, ax = plt.subplots(figsize=(8, 9))
fig.patch.set_alpha(0)      # transparent figure background
ax.set_facecolor("none")

# Base map: all districts in pale background
plot_gdf.plot(
    ax=ax,
    color="#f1efc8",
    edgecolor="#b7b7a4",
    linewidth=0.7
)

# Legend axis on the right
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="4%", pad=0.08)

# Overlay selected districts with continuous log-scale colors
selected_gdf = plot_gdf[plot_gdf["selected"]].copy()

selected_gdf.plot(
    column="log_cases",
    ax=ax,
    cmap="YlOrRd",
    edgecolor="#8f8f8f",
    linewidth=0.8,
    legend=True,
    cax=cax,
    vmin=0,
    vmax=max(10, selected_gdf["log_cases"].max()),
    legend_kwds={"label": "Cases (log scale)"}
)

# Labels
for _, row in selected_gdf.iterrows():
    key = row["district_key"]
    label = label_map.get(key, row[name_col])

    if (
        "center_lon" in row.index and "center_lat" in row.index
        and pd.notna(row["center_lon"]) and pd.notna(row["center_lat"])
    ):
        x = row["center_lon"]
        y = row["center_lat"]
    else:
        pt = row.geometry.representative_point()
        x, y = pt.x, pt.y

    dx, dy = label_offsets.get(key, (0, 0))

    ax.text(
        x + dx, y + dy,
        label,
        fontsize=8,
        color="black",
        ha="center",
        va="center",
        fontweight="bold"
    )

ax.set_axis_off()
plt.tight_layout()

# =========================================
# SAVE PDF
# =========================================
plt.savefig(
    output_pdf,
    format="pdf",
    dpi=300,
    bbox_inches="tight",
    transparent=True
)
plt.close()

print(f"Saved PDF: {output_pdf}")