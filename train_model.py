import pandas as pd
import joblib
from pathlib import Path

df = pd.read_csv("dataset/Waste_Management_India_20K.csv")

print("Dataset shape:", df.shape)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

chennai = df[df["City"] == "Chennai"]

print("\nChennai records:", len(chennai))

print("\nChennai years:")
print(sorted(chennai["Year"].unique()))

print("\nChennai waste types:")
print(chennai["Waste_Type"].unique())

print("\nChennai records by year:")
print(chennai.groupby("Year").size())

print("\nChennai records by year and month:")
print(chennai.groupby(["Year", "Month"]).size())

print("\nChennai sample records:")
print(
    chennai[
        ["Year", "Month", "Waste_Type", "Daily_Waste_Generation_Tons"]
    ].sort_values(["Year", "Month"]).head(30)
)

print("\nChennai waste type counts:")
print(chennai["Waste_Type"].value_counts())

print("\nTarget statistics:")
print(chennai["Daily_Waste_Generation_Tons"].describe())

print("\nChennai target and waste-per-capita:")
print(
    chennai[
        [
            "Year",
            "Month",
            "Waste_Type",
            "Population",
            "Daily_Waste_Generation_Tons",
            "Waste_Per_Capita_KG"
        ]
    ].head(20)
)

print("\nChecking Waste Per Capita relationship:")

chennai["Calculated_Waste_Per_Capita"] = (
    chennai["Daily_Waste_Generation_Tons"] * 1000
    / chennai["Population"]
)

print(
    chennai[
        [
            "Daily_Waste_Generation_Tons",
            "Population",
            "Waste_Per_Capita_KG",
            "Calculated_Waste_Per_Capita"
        ]
    ].head(20)
)

print("\nPotential target-related features:")

features_to_check = [
    "Collection_Efficiency_Percentage",
    "Waste_Segregation_Rate",
    "Recycling_Rate",
    "Composting_Rate",
    "Landfill_Dependency_Rate",
    "Waste_Processing_Capacity",
    "Municipal_Efficiency_Score",
    "Waste_Management_Budget_INR",
    "Cost_Per_Ton_INR",
    "Carbon_Emission_From_Waste",
    "Landfill_Capacity_Tons",
    "Landfill_Remaining_Capacity",
    "Sustainability_Index"
]

print(chennai[features_to_check].describe().T)

print("\nChennai yearly waste generation:")

yearly_waste = (
    chennai
    .groupby(["Year", "Waste_Type"])["Daily_Waste_Generation_Tons"]
    .mean()
    .reset_index()
)

print(yearly_waste.to_string(index=False))

print("\nChennai records by Year and Month:")

monthly_records = (
    chennai
    .groupby(["Year", "Month"])
    .size()
    .reset_index(name="Record_Count")
)

print(monthly_records.to_string(index=False))

print("\nChecking Year-Month-Waste Type uniqueness:")

duplicates = (
    chennai
    .groupby(["Year", "Month", "Waste_Type"])
    .size()
    .reset_index(name="Count")
)

print("Total combinations:", len(duplicates))
print("Duplicate combinations:")
print(duplicates[duplicates["Count"] > 1].to_string(index=False))

safe_features = [
    "Year",
    "Month",
    "Season",
    "Population",
    "Population_Density",
    "Urbanization_Rate",
    "Waste_Type",
    "Latitude",
    "Longitude",
    "Air_Quality_Index",
    "Rainfall_MM",
    "Tourism_Impact_Score",
    "Industrial_Activity_Level",
    "Population_Growth_Rate"
]

print("\nPotential forecasting features:")
print(chennai[safe_features].head(20).to_string(index=False))

print("\nUnique values:")
for col in safe_features:
    print(f"{col}: {chennai[col].nunique()}")

print("\nObservations per Waste Type:")

waste_type_counts = (
    chennai["Waste_Type"]
    .value_counts()
    .sort_values(ascending=False)
)

print(waste_type_counts)

print("\nCorrelation with target:")

numeric_features = [
    "Year",
    "Month",
    "Population",
    "Population_Density",
    "Urbanization_Rate",
    "Air_Quality_Index",
    "Rainfall_MM",
    "Tourism_Impact_Score",
    "Industrial_Activity_Level",
    "Population_Growth_Rate",
    "Daily_Waste_Generation_Tons"
]

correlation = chennai[numeric_features].corr()["Daily_Waste_Generation_Tons"].sort_values(
    ascending=False
)

print(correlation)

train = chennai[chennai["Year"] <= 2023].copy()
test = chennai[chennai["Year"] >= 2024].copy()

print("Training records:", len(train))
print("Testing records:", len(test))

print("\nTraining years:")
print(sorted(train["Year"].unique()))

print("\nTesting years:")
print(sorted(test["Year"].unique()))



from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

baseline_prediction = train["Daily_Waste_Generation_Tons"].mean()

y_test = test["Daily_Waste_Generation_Tons"]

baseline_mae = mean_absolute_error(
    y_test,
    np.full(len(y_test), baseline_prediction)
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        np.full(len(y_test), baseline_prediction)
    )
)

baseline_r2 = r2_score(
    y_test,
    np.full(len(y_test), baseline_prediction)
)

print("\nBaseline Results")
print("----------------")
print("Baseline prediction:", round(baseline_prediction, 2))
print("MAE:", round(baseline_mae, 2))
print("RMSE:", round(baseline_rmse, 2))
print("R2:", round(baseline_r2, 4))

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

features = [
    "Year",
    "Month",
    "Season",
    "Waste_Type",
    "Population",
    "Population_Density",
    "Urbanization_Rate",
    "Air_Quality_Index",
    "Rainfall_MM",
    "Tourism_Impact_Score",
    "Industrial_Activity_Level",
    "Population_Growth_Rate"
]

target = "Daily_Waste_Generation_Tons"

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

categorical_features = [
    "Season",
    "Waste_Type"
]

numeric_features = [
    "Year",
    "Month",
    "Population",
    "Population_Density",
    "Urbanization_Rate",
    "Air_Quality_Index",
    "Rainfall_MM",
    "Tourism_Impact_Score",
    "Industrial_Activity_Level",
    "Population_Growth_Rate"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Original training shape:", X_train.shape)
print("Processed training shape:", X_train_processed.shape)

print("Original testing shape:", X_test.shape)
print("Processed testing shape:", X_test_processed.shape)


from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    max_depth=None,
    min_samples_split=2,
    n_jobs=-1
)

rf_model.fit(X_train_processed, y_train)

rf_predictions = rf_model.predict(X_test_processed)

rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
rf_r2 = r2_score(y_test, rf_predictions)

print("\nRandom Forest Results")
print("---------------------")
print("MAE:", round(rf_mae, 2))
print("RMSE:", round(rf_rmse, 2))
print("R2:", round(rf_r2, 4))


print("\nActual vs Predicted:")
print("---------------------")

results = test[
    ["Year", "Month", "Waste_Type", "Daily_Waste_Generation_Tons"]
].copy()

results["Predicted_Waste"] = rf_predictions

results["Absolute_Error"] = (
    results["Daily_Waste_Generation_Tons"]
    - results["Predicted_Waste"]
).abs()

print(
    results.sort_values(["Year", "Month"]).to_string(index=False)
)


feature_names = preprocessor.get_feature_names_out()

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\nRandom Forest Feature Importance")
print("---------------------------------")
print(importance.to_string(index=False))

type_means = (
    train
    .groupby("Waste_Type")["Daily_Waste_Generation_Tons"]
    .mean()
)

type_predictions = test["Waste_Type"].map(type_means)

type_mae = mean_absolute_error(
    y_test,
    type_predictions
)

type_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        type_predictions
    )
)

type_r2 = r2_score(
    y_test,
    type_predictions
)

print("\nWaste-Type Baseline Results")
print("----------------------------")
print("MAE:", round(type_mae, 2))
print("RMSE:", round(type_rmse, 2))
print("R2:", round(type_r2, 4))


# ==========================================
# SAVE MODEL AND PREPROCESSING PIPELINE
# ==========================================

model_dir = Path("models")
model_dir.mkdir(parents=True, exist_ok=True)

model_artifact = {
    "model": rf_model,
    "preprocessor": preprocessor,
    "features": features,
    "target": target,
    "categorical_features": categorical_features,
    "numeric_features": numeric_features,
    "training_years": [int(year) for year in sorted(train["Year"].unique())],
    "test_years": [int(year) for year in sorted(test["Year"].unique())],
    "evaluation": {
        "MAE": float(rf_mae),
        "RMSE": float(rf_rmse),
        "R2": float(rf_r2)
    }
}

model_path = model_dir / "waste_forecasting_pipeline.pkl"

joblib.dump(model_artifact, model_path)

print("\nModel Saving")
print("------------")
print(f"Model and preprocessor saved successfully: {model_path}")
print(f"File exists: {model_path.exists()}")
