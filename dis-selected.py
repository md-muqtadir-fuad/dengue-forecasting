import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import re

# =========================================
# FILE PATHS
# =========================================
geojson_path = "./data/district/bgd_admin2.geojson"
output_pdf = "./data/output/selected_dengue_districts_bd.pdf"

# =========================================
# SELECTED DISTRICTS (same as figure)
# Note: some names are standardized for matching
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
        "coxs bazaar": "coxs bazar"
    }
    return replacements.get(name, name)

selected_keys = [normalize_name(x) for x in selected_districts]

# =========================================
# READ GEOJSON
# =========================================
gdf = gpd.read_file(geojson_path)

# Use the confirmed district name column
name_col = "adm2_name"

# Normalize district names from geojson
gdf["district_key"] = gdf[name_col].apply(normalize_name)

# Flag selected districts
gdf["selected"] = gdf["district_key"].isin(selected_keys)

# =========================================
# CHECK MATCHES
# =========================================
matched = set(gdf.loc[gdf["selected"], "district_key"])
missing = set(selected_keys) - matched

print("Matched districts:", sorted(matched))
if missing:
    print("Warning: these districts were not matched:", sorted(missing))

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

# Optional manual label offsets to resemble the figure better
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
# PLOT
# =========================================
fig, ax = plt.subplots(figsize=(8, 9))
fig.patch.set_alpha(0) 
ax.set_facecolor("none")

# All districts: light green
gdf.plot(
    ax=ax,
    color="#a8f0a2",
    edgecolor="#7ea57e",
    linewidth=0.6
)

# Selected districts: orange
gdf[gdf["selected"]].plot(
    ax=ax,
    color="#f4a033",
    edgecolor="#7ea57e",
    linewidth=0.8
)

# Labels
selected_gdf = gdf[gdf["selected"]].copy()

for _, row in selected_gdf.iterrows():
    key = row["district_key"]
    label = label_map.get(key, row[name_col])

    # Prefer provided center coords if available
    if "center_lon" in row and "center_lat" in row and pd.notna(row["center_lon"]) and pd.notna(row["center_lat"]):
        x = row["center_lon"]
        y = row["center_lat"]
    else:
        pt = row.geometry.representative_point()
        x, y = pt.x, pt.y

    dx, dy = label_offsets.get(key, (0, 0))

    ax.text(
        x + dx, y + dy, label,
        fontsize=8,
        color="black",
        ha="center",
        va="center",
        fontweight="bold"
    )

# Clean map look
ax.set_axis_off()

# Caption like the figure
#caption = "(a) Selected locations for ML dengue\nforecasting in Bangladesh (orange), overlaid on\ndistrict boundaries"
# fig.text(
#     0.5, 0.03,
#     caption,
#     ha="center",
#     va="bottom",
#     fontsize=11,
#     family="serif"
# )

plt.subplots_adjust(bottom=0.16)

# Save PDF
plt.savefig(
    output_pdf,
    format="pdf",
    dpi=300,
    bbox_inches="tight",
    transparent=True
)
plt.close()

print(f"Saved PDF: {output_pdf}")