# format_utils.py
import re

# manual overrides: these take priority whenever present
column_renaming = {
    'denv4': 'DENV-4',
    'Year': 'Year',
    'AvgTemp_lag_3': 'Avg. Temp (Lag 3)',
    'Month': 'Month',
    'PopulationDensity': 'Pop. Density',
    'MonthlyPrevailingWindDir_ENE': 'Wind Dir. (ENE)',
    'Rainfall_lag_2': 'Rainfall (Lag 2)',
    'MonthlyAvgSunshineHours _lag_1': 'Sunshine Hrs (Lag 1)',
    'denv1': 'DENV-1',
    'Humidity': 'Humidity',
    'Log_NoOfDenguePatients': 'Log(Dengue Cases)'
}

column_renaming['MonthlyAvgSunshineHours_lag_1'] = 'Sunshine Hrs (Lag 1)'

def pretty_column_name(col: str) -> str:
    """
    Global display formatter for EVERY column shown in notebook outputs.
    Manual mapping first, then automatic cleanup/formatting.
    """
    if col in column_renaming:
        return column_renaming[col]

    c = str(col).strip()

    # handle lag suffix nicely
    lag_match = re.match(r"^(.*)_lag_(\d+)$", c)
    if lag_match:
        base, lag_num = lag_match.groups()
        return f"{pretty_column_name(base)} (Lag {lag_num})"

    # common exact replacements
    exact_map = {
        "NoOfDenguePatients": "Dengue Cases",
        "Log_NoOfDenguePatients": "Log(Dengue Cases)",
        "Month-year": "Month-Year",
        "PopulationDensity": "Pop. Density",
        "MonthlyAvgSunshineHours": "Sunshine Hrs",
        "MonthlyPrevailingWindDir": "Wind Direction",
        "MonthlyPrevailingWindSpeed": "Wind Speed",
        "MonthlyAvgSeaLevelPressure": "Sea Level Pressure",
        "MonthlyAvgVisibility": "Visibility",
        "AvgTemp": "Avg. Temp",
        "MinTemp": "Min. Temp",
        "MaxTemp": "Max. Temp",
        "Rainfall": "Rainfall",
        "Humidity": "Humidity",
        "dominant": "Dominant Serotype",
        "Date": "Date",
        "Month": "Month",
        "Year": "Year",
        "Month_sin": "Month (Sin)",
        "District": "District"
    }
    if c in exact_map:
        return exact_map[c]

    if re.fullmatch(r"denv\d+", c.lower()):
        num = re.findall(r"\d+", c)[0]
        return f"DENV-{num}"

    if c.startswith("MonthlyPrevailingWindDir_"):
        suffix = c.replace("MonthlyPrevailingWindDir_", "").strip()
        return f"Wind Dir. ({suffix})"

    c = c.replace("_", " ")
    c = re.sub(r"\s+", " ", c).strip()
    c = re.sub(r"([a-z])([A-Z])", r"\1 \2", c)
    c = c.replace("Avg ", "Avg. ")
    c = c.replace("Min ", "Min. ")
    c = c.replace("Max ", "Max. ")
    c = c.replace("Pop Density", "Pop. Density")

    return c