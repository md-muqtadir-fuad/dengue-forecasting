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
csv_path = "./data/raw/prime_dataset_model_input_with_purge.csv"  # kept from second script
output_pdf = "./data/output/combined_bd_maps.pdf"

os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

# =========================================
# SELECTED DISTRICTS (from first script)
# =========================================
selected_districts = [
    "Rajshahi",
    "Mymensingh",
    "Sylhet",
    "Dhaka",
    "Faridpur",
    "Jessore",
    "Khulna",
    "Barisal",
    "Bhola",
    "Chattogram",
    "Cox's Bazar"
]

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

selected_keys = [normalize_name(x) for x in selected_districts]

# =========================================
# LABELS TO DISPLAY EXACTLY LIKE FIGURE
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
    "coxs bazar": "Cox's Bazar"
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
    "coxs bazar": (0.03, -0.12)
}

# =========================================
# READ GEOJSON
# =========================================
gdf = gpd.read_file(geojson_path).copy()
name_col = "adm2_name"
gdf["district_key"] = gdf[name_col].apply(normalize_name)

# Flag selected districts for panel 1
gdf["selected"] = gdf["district_key"].isin(selected_keys)

# =========================================
# CHECK MATCHES FOR PANEL 1
# =========================================
matched = set(gdf.loc[gdf["selected"], "district_key"])
missing = set(selected_keys) - matched

print("Matched selected districts:", sorted(matched))
if missing:
    print("Warning: these selected districts were not matched:", sorted(missing))

# =========================================
# READ CSV AND FILTER AUGUST 2023 (kept exactly from second script)
# =========================================
df = pd.read_csv(csv_path).copy()
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["district_key"] = df["District"].apply(normalize_name)

aug_2023 = df[(df["Date"].dt.year == 2023) & (df["Date"].dt.month == 8)].copy()

if aug_2023.empty:
    raise ValueError("No rows found for August 2023 in the CSV.")

# =========================================
# DISTRICT-WISE SUM, THEN LOG SCALE
# =========================================
aug_2023["raw_cases"] = np.expm1(aug_2023["Log_NoOfDenguePatients"])

district_aug = (
    aug_2023.groupby("district_key", as_index=False)["raw_cases"]
    .sum()
    .rename(columns={"raw_cases": "aug_2023_cases"})
)

district_aug["log_cases"] = np.log1p(district_aug["aug_2023_cases"])

# =========================================
# MERGE FOR PANEL 2
# =========================================
plot_gdf = gdf.merge(district_aug, on="district_key", how="left", suffixes=("", "_csv"))
plot_gdf["selected_log"] = plot_gdf["log_cases"].notna()

matched_log = sorted(plot_gdf.loc[plot_gdf["selected_log"], name_col].tolist())
print("Matched log-map districts:", matched_log)

missing_log = sorted(set(district_aug["district_key"]) - set(plot_gdf["district_key"]))
if missing_log:
    print("Warning: unmatched district keys in log map:", missing_log)

# =========================================
# HELPER FOR LABELING
# =========================================
def get_label_xy(row):
    if (
        "center_lon" in row.index and "center_lat" in row.index
        and pd.notna(row["center_lon"]) and pd.notna(row["center_lat"])
    ):
        return row["center_lon"], row["center_lat"]
    pt = row.geometry.representative_point()
    return pt.x, pt.y

def add_labels(ax, label_gdf):
    for _, row in label_gdf.iterrows():
        key = row["district_key"]
        label = label_map.get(key, row[name_col])
        x, y = get_label_xy(row)
        dx, dy = label_offsets.get(key, (0, 0))

        ax.text(
            x + dx, y + dy, label,
            fontsize=8,
            color="black",
            ha="center",
            va="center",
            fontweight="bold"
        )

# =========================================
# PLOT COMBINED FIGURE
# =========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
fig.patch.set_alpha(0)
ax1.set_facecolor("none")
ax2.set_facecolor("none")

# -----------------------------------------
# PANEL 1: SELECTED DISTRICTS MAP
# -----------------------------------------
gdf.plot(
    ax=ax1,
    color="#a8f0a2",
    edgecolor="#7ea57e",
    linewidth=0.6
)

gdf[gdf["selected"]].plot(
    ax=ax1,
    color="#f4a033",
    edgecolor="#7ea57e",
    linewidth=0.8
)

selected_gdf_panel1 = gdf[gdf["selected"]].copy()
add_labels(ax1, selected_gdf_panel1)

ax1.set_axis_off()

# -----------------------------------------
# PANEL 2: LOG-SCALE DENGUE CASE MAP
# -----------------------------------------
plot_gdf.plot(
    ax=ax2,
    color="#f1efc8",
    edgecolor="#b7b7a4",
    linewidth=0.7
)

divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="4%", pad=0.08)

selected_gdf_panel2 = plot_gdf[plot_gdf["selected_log"]].copy()

selected_gdf_panel2.plot(
    column="log_cases",
    ax=ax2,
    cmap="YlOrRd",
    edgecolor="#8f8f8f",
    linewidth=0.8,
    legend=True,
    cax=cax,
    vmin=0,
    vmax=max(10, selected_gdf_panel2["log_cases"].max()),
    legend_kwds={"label": "Cases (log scale)"}
)

add_labels(ax2, selected_gdf_panel2)

ax2.set_axis_off()
ax1.text(
    0.02, 0.98, "A",
    transform=ax1.transAxes,
    fontsize=14,
    fontweight="bold",
    va="top",
    ha="left"
)

ax2.text(
    0.02, 0.98, "B",
    transform=ax2.transAxes,
    fontsize=14,
    fontweight="bold",
    va="top",
    ha="left"
)

# Keep layout clean without adding panel captions into the image
plt.tight_layout()

# =========================================
# SAVE COMBINED PDF
# =========================================
plt.savefig(
    output_pdf,
    format="pdf",
    dpi=300,
    bbox_inches="tight",
    transparent=True
)
plt.close()

print(f"Saved combined PDF: {output_pdf}")