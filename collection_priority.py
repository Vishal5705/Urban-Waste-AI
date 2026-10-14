
from pathlib import Path
import argparse

import joblib
import numpy as np
import pandas as pd


# ============================================================
# AI-DRIVEN URBAN WASTE MANAGEMENT
# Collection-Priority Engine
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR / "dataset" / "Waste_Management_India_20K.csv"
)

MODEL_PATH = (
    BASE_DIR / "models" / "waste_forecasting_pipeline.pkl"
)

OUTPUT_PATH = (
    BASE_DIR / "outputs" / "collection_priorities.csv"
)

CITY_NAME = "Chennai"
TARGET_COLUMN = "Daily_Waste_Generation_Tons"
CITY_COLUMN = "City"
WASTE_TYPE_COLUMN = "Waste_Type"


# ============================================================
# 1. COLLECTION-PRIORITY RULES
# ============================================================

def assign_priority(percentile):
    """
    Assign relative collection priority.

    High:   percentile >= 75
    Medium: percentile >= 25 and < 75
    Low:    percentile < 25

    These are prototype rules, not official
    municipal collection standards.
    """
    if percentile >= 75:
        return "High"
    elif percentile >= 25:
        return "Medium"
    return "Low"


# ============================================================
# 2. SEASON HELPER
# ============================================================

def get_season(month):
    """Return the season label used by this prototype."""
    if month in (12, 1, 2):
        return "Winter"
    elif month in (3, 4, 5):
        return "Summer"
    elif month in (6, 7, 8, 9):
        return "Monsoon"
    return "Post-Monsoon"


# ============================================================
# 3. LOAD DATASET AND SAVED MODEL
# ============================================================


def load_resources():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved model not found:\n{MODEL_PATH}\n"
            "Please check that train_model.py has been run."
        )

    df = pd.read_csv(DATASET_PATH)
    saved_objects = joblib.load(MODEL_PATH)

    # The saved file contains a dictionary.
    model = saved_objects["model"]
    preprocessor = saved_objects["preprocessor"]
    feature_names = saved_objects["features"]

    required_columns = {
        CITY_COLUMN,
        WASTE_TYPE_COLUMN,
        TARGET_COLUMN,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required dataset columns: "
            f"{sorted(missing_columns)}"
        )

    missing_features = [
        name for name in feature_names
        if name not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "The dataset is missing model features: "
            f"{missing_features}"
        )

    return df, (preprocessor, model), feature_names



# ============================================================
# 4. PREPARE CHENNAI FORECAST SCENARIOS
# ============================================================

def build_scenarios(
    df,
    feature_names,
    forecast_year,
    forecast_month,
):
    """
    Select the latest available Chennai record for each
    waste type and use it as a scenario template.

    Only Year, Month and Season are updated where those
    features are required by the saved model.

    Other features retain their values from the source
    records. Therefore, these are scenario estimates,
    not validated future forecasts.
    """

    chennai = df[
        df[CITY_COLUMN]
        .astype(str)
        .str.strip()
        .str.casefold()
        .eq(CITY_NAME.casefold())
    ].copy()

    if chennai.empty:
        raise ValueError(
            f"No records found for {CITY_NAME}."
        )

    chennai = chennai.dropna(
        subset=[WASTE_TYPE_COLUMN]
    )

    date_columns = [
        col for col in ["Year", "Month"]
        if col in chennai.columns
    ]

    if date_columns:
        chennai = chennai.sort_values(date_columns)

    # Latest source record for each waste type.
    latest_records = (
        chennai.groupby(
            WASTE_TYPE_COLUMN,
            sort=False,
            group_keys=False,
        )
        .tail(1)
        .copy()
        .reset_index(drop=True)
    )

    if latest_records.empty:
        raise ValueError(
            "No usable Chennai waste-type records found."
        )

    # Update time-related features only when required.
    if "Year" in feature_names:
        latest_records["Year"] = forecast_year

    if "Month" in feature_names:
        latest_records["Month"] = forecast_month

    if "Season" in feature_names:
        latest_records["Season"] = get_season(
            forecast_month
        )

    X = latest_records[feature_names].copy()

    return latest_records, X


# ============================================================
# 5. GENERATE WASTE ESTIMATES
# ============================================================


def generate_predictions(preprocessor, model, X):
    # Apply the saved preprocessing steps first
    transformed_X = preprocessor.transform(X)

    predictions = np.asarray(
        model.predict(transformed_X),
        dtype=float,
    ).reshape(-1)

    # Prevent negative waste predictions
    predictions = np.maximum(predictions, 0)

    return predictions



# ============================================================
# 6. CREATE COLLECTION-PRIORITY REPORT
# ============================================================

def create_priority_report(
    records,
    predictions,
    forecast_year,
    forecast_month,
):
    report = pd.DataFrame({
        "City": records[CITY_COLUMN].astype(str),
        "Waste_Type": records[
            WASTE_TYPE_COLUMN
        ].astype(str),
        "Forecast_Year": forecast_year,
        "Forecast_Month": forecast_month,
        "Raw_Predicted_Tons_Per_Day": predictions,
    })

    # Do not report negative estimated waste.
    report["Estimated_Tons_Per_Day"] = (
        report["Raw_Predicted_Tons_Per_Day"]
        .clip(lower=0)
        .round(2)
    )

    # Relative percentile among the selected waste types.
    report["Priority_Percentile"] = (
        report["Estimated_Tons_Per_Day"]
        .rank(method="average", pct=True)
        .mul(100)
        .round(2)
    )

    report["Collection_Priority"] = (
        report["Priority_Percentile"]
        .apply(assign_priority)
    )

    report["Priority_Rule"] = (
        "High >= 75; Medium >= 25 and < 75; Low < 25"
    )

    report["Interpretation"] = (
        "City-level waste-type scenario; "
        "not an individual-bin estimate"
    )

    # Display highest estimated waste first.
    report = report.sort_values(
        "Estimated_Tons_Per_Day",
        ascending=False,
    ).reset_index(drop=True)

    return report


# ============================================================
# 7. MAIN PROGRAM
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate Chennai waste estimates and "
            "relative collection priorities."
        )
    )

    parser.add_argument(
        "--year",
        type=int,
        default=2026,
        help="Scenario year (default: 2026)",
    )

    parser.add_argument(
        "--month",
        type=int,
        choices=range(1, 13),
        default=10,
        help="Scenario month, 1-12 (default: 10)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(" AI-DRIVEN URBAN WASTE MANAGEMENT")
    print(" COLLECTION-PRIORITY ENGINE")
    print("=" * 60)

    print("\nLoading dataset and trained pipeline...")

    df, (preprocessor, model), feature_names = load_resources()

    print("Dataset loaded successfully.")
    print("Saved forecasting pipeline loaded successfully.")
    print(f"Required model features: {feature_names}")

    print(
        f"\nPreparing Chennai scenarios for "
        f"{args.year}-{args.month:02d}..."
    )

    records, X = build_scenarios(
        df,
        feature_names,
        args.year,
        args.month,
    )

    print(f"Scenarios generated: {len(X)}")

    predictions = generate_predictions(preprocessor, model, X)

    report = create_priority_report(
        records,
        predictions,
        args.year,
        args.month,
    )

    # Create the output directory automatically.
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    display_columns = [
        "City",
        "Waste_Type",
        "Estimated_Tons_Per_Day",
        "Priority_Percentile",
        "Collection_Priority",
    ]

    print("\nCOLLECTION-PRIORITY RESULTS")
    print("-" * 60)
    print(
        report[display_columns].to_string(index=False)
    )

    print("\nPRIORITY COUNTS")
    print("-" * 30)

    counts = (
        report["Collection_Priority"]
        .value_counts()
        .reindex(
            ["High", "Medium", "Low"],
            fill_value=0,
        )
    )

    print(counts.to_string())

    print("\nCSV export completed.")
    print(f"Output file: {OUTPUT_PATH}")

    print(
        "\nIMPORTANT: These are scenario-based, "
        "city-level waste-type estimates. Priorities "
        "are relative rankings, not validated "
        "bin-level collection instructions."
    )


if __name__ == "__main__":
    main()
