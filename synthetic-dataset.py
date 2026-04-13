import os
import argparse
import numpy as np
import pandas as pd


def month_angle(month_series):
    return 2 * np.pi * (month_series - 1) / 12.0


def build_district_effects(districts, rng):
    effects = {}
    for d in districts:
        effects[d] = {
            "temp": rng.normal(0.0, 0.8),
            "rain": rng.normal(0.0, 40.0),
            "sun": rng.normal(0.0, 0.35),
            "hum": rng.normal(0.0, 2.5),
            "wind": rng.normal(0.0, 0.06),
        }
    return effects


def synthesize_weather_features(df, seed=42):
    rng = np.random.default_rng(seed)
    out = df.copy()

    print("Loaded columns:")
    print(list(out.columns))

    required_cols = ["District", "Date"]
    for c in required_cols:
        if c not in out.columns:
            raise ValueError(
                f"Required column missing: {c}. "
                f"Available columns: {list(out.columns)}"
            )

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    if out["Date"].isna().any():
        bad_n = out["Date"].isna().sum()
        raise ValueError(f"'Date' parsing failed for {bad_n} row(s). Check date format.")

    out["month_num"] = out["Date"].dt.month

    districts = sorted(out["District"].dropna().astype(str).unique())
    district_fx = build_district_effects(districts, rng)

    theta = month_angle(out["month_num"].values)
    monsoon = np.sin(theta - np.pi / 3)
    dryness = -monsoon
    shared_shock = rng.normal(0, 1, size=len(out))

    temp_fx = out["District"].astype(str).map(lambda d: district_fx[d]["temp"]).values
    rain_fx = out["District"].astype(str).map(lambda d: district_fx[d]["rain"]).values
    sun_fx = out["District"].astype(str).map(lambda d: district_fx[d]["sun"]).values
    hum_fx = out["District"].astype(str).map(lambda d: district_fx[d]["hum"]).values
    wind_fx = out["District"].astype(str).map(lambda d: district_fx[d]["wind"]).values

    temp = (
        27.0
        + 2.8 * np.sin(theta - np.pi / 6)
        + temp_fx
        + 0.35 * shared_shock
        + rng.normal(0, 0.7, size=len(out))
    )
    temp = np.clip(temp, 18.0, 36.0)

    rain = (
        120.0
        + 180.0 * (monsoon + 1.0)
        + rain_fx
        + 30.0 * shared_shock
        + rng.normal(0, 45.0, size=len(out))
    )
    rain = np.clip(rain, 0.0, 900.0)

    sunshine = (
        7.0
        + 1.2 * dryness
        + sun_fx
        - 0.12 * shared_shock
        + rng.normal(0, 0.45, size=len(out))
    )
    sunshine = np.clip(sunshine, 2.0, 10.5)

    humidity = (
        74.0
        + 9.0 * monsoon
        + hum_fx
        + 0.8 * shared_shock
        + rng.normal(0, 2.8, size=len(out))
    )
    humidity = np.clip(humidity, 50.0, 96.0)

    p_ene = 0.22 + 0.12 * np.sin(theta + np.pi / 8) + wind_fx
    p_ene = np.clip(p_ene, 0.05, 0.70)
    wind_ene = rng.binomial(1, p_ene, size=len(out)).astype(int)

    replacement_map = {
        "AvgTemp_lag_3": temp,
        "Rainfall_lag_2": rain,
        "MonthlyAvgSunshineHours_lag_1": sunshine,
        "Humidity_lag_1": humidity,
        "MonthlyPrevailingWindDir_ENE": wind_ene,
    }

    replaced_cols = []
    missing_cols = []

    for col, values in replacement_map.items():
        if col in out.columns:
            out[col] = values
            replaced_cols.append(col)
        else:
            missing_cols.append(col)

    out.drop(columns=["month_num"], inplace=True)

    print("\nReplaced columns:")
    for c in replaced_cols:
        print(f"  - {c}")

    print("\nWeather columns not found in file:")
    for c in missing_cols:
        print(f"  - {c}")

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Reading: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    print(f"Input shape: {df.shape}")

    out_df = synthesize_weather_features(df, seed=args.seed)

    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    out_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved synthetic dataset to: {args.output_csv}")
    print(f"Output shape: {out_df.shape}")


if __name__ == "__main__":
    main()