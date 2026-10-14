import math
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"


# ==========================================
# 1. CACHED DATA & MODEL LOADERS
# ==========================================

@st.cache_resource
def load_forecasting_pipeline():
    """Loads the trained waste forecasting pipeline dictionary."""
    model_path = MODEL_DIR / "waste_forecasting_pipeline.pkl"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


@st.cache_data
def load_bin_data():
    """Loads the Chennai Garbage Bin dataset."""
    bin_path = DATASET_DIR / "Chennai_Bins.csv"
    if not bin_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(bin_path)
    df["ZONE_CODE"] = df["ZONE_CODE"].astype(str).str.strip()
    df["DIV_CODE"] = df["DIV_CODE"].astype(str).str.strip()
    return df


@st.cache_data
def load_waste_data():
    """Loads the India 20K Waste Management dataset."""
    waste_path = DATASET_DIR / "Waste_Management_India_20K.csv"
    if not waste_path.exists():
        return pd.DataFrame()
    return pd.read_csv(waste_path)


@st.cache_data
def load_clean_collection_data():
    """Loads the cleaned garbage collection dataset."""
    coll_path = DATASET_DIR / "Garbage_collection_clean.csv"
    if not coll_path.exists():
        return pd.DataFrame()
    return pd.read_csv(coll_path, encoding="latin1")


# ==========================================
# 2. HAVERSINE DISTANCE & MATRIX GENERATOR
# ==========================================

def calculate_haversine_km(lat1, lon1, lat2, lon2):
    """Calculates straight-line Haversine distance in km between two lat/lon points."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    a = max(0.0, min(1.0, a))
    return 6371.0 * (2 * math.asin(math.sqrt(a)))


def build_distance_matrix_metres(lats, lons):
    """Builds a symmetric N x N distance matrix in integer metres with zero diagonal."""
    n = len(lats)
    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(0)
            else:
                dist_km = calculate_haversine_km(lats[i], lons[i], lats[j], lons[j])
                row.append(int(round(dist_km * 1000)))
        matrix.append(row)
    return matrix


def compute_route_distance_metres(route, distance_matrix):
    """Computes total route distance in metres for a node sequence."""
    total = 0
    for i in range(len(route) - 1):
        total += distance_matrix[route[i]][route[i + 1]]
    return total


# ==========================================
# 3. ROUTE SOLVERS
# ==========================================

def solve_csv_order_route(num_locations, depot=0, distance_matrix=None):
    """CSV Order Baseline Route."""
    route = list(range(num_locations)) + [depot]
    dist_m = compute_route_distance_metres(route, distance_matrix) if distance_matrix else 0
    return {
        "route": route,
        "distance_km": dist_m / 1000.0,
        "locations_visited": num_locations,
        "method": "CSV-order baseline"
    }


def solve_nearest_neighbor_route(num_locations, depot=0, distance_matrix=None):
    """Nearest Neighbor Greedy Baseline Route."""
    unvisited = set(range(num_locations))
    unvisited.remove(depot)
    route = [depot]
    current = depot

    while unvisited:
        nxt = min(unvisited, key=lambda node: distance_matrix[current][node])
        route.append(nxt)
        unvisited.remove(nxt)
        current = nxt

    route.append(depot)
    dist_m = compute_route_distance_metres(route, distance_matrix)
    return {
        "route": route,
        "distance_km": dist_m / 1000.0,
        "locations_visited": num_locations,
        "method": "Nearest-neighbor baseline"
    }


def solve_ortools_tsp(lats, lons, depot=0, time_limit_sec=5):
    """Solves TSP using Google OR-Tools Guided Local Search with input validation."""
    if lats is None or lons is None or len(lats) != len(lons):
        raise ValueError("Latitude and longitude lists must be non-null and equal in length.")

    num_locations = len(lats)
    if num_locations < 2:
        raise ValueError("At least two locations are required for route optimization.")

    if not (0 <= depot < num_locations):
        raise ValueError(f"Depot index {depot} is out of bounds for {num_locations} locations.")

    # Validate numeric ranges
    lats_arr = np.array(lats, dtype=float)
    lons_arr = np.array(lons, dtype=float)

    if not np.isfinite(lats_arr).all() or not np.isfinite(lons_arr).all():
        raise ValueError("Latitude or longitude contains non-numeric or infinite values.")

    if not ((lats_arr >= -90) & (lats_arr <= 90)).all():
        raise ValueError("Invalid latitude range found (must be between -90 and 90).")

    if not ((lons_arr >= -180) & (lons_arr <= 180)).all():
        raise ValueError("Invalid longitude range found (must be between -180 and 180).")

    start_time = time.time()
    distance_matrix = build_distance_matrix_metres(lats, lons)

    manager = pywrapcp.RoutingIndexManager(num_locations, 1, depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = int(max(1, time_limit_sec))

    solution = routing.SolveWithParameters(search_parameters)
    exec_time = time.time() - start_time

    if solution is None:
        return None

    optimized_route = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        optimized_route.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    optimized_route.append(depot)

    # Route integrity checks
    if len(set(optimized_route[:-1])) != num_locations:
        raise ValueError(f"Route did not visit every bin exactly once: visited {len(set(optimized_route[:-1]))} of {num_locations}.")

    if optimized_route[0] != depot or optimized_route[-1] != depot:
        raise ValueError(f"Route start/end depot mismatch: expected depot {depot}, got start {optimized_route[0]} and end {optimized_route[-1]}.")

    opt_dist_m = compute_route_distance_metres(optimized_route, distance_matrix)

    # Calculate baselines for comparison using identical distance matrix and depot
    csv_res = solve_csv_order_route(num_locations, depot, distance_matrix)
    nn_res = solve_nearest_neighbor_route(num_locations, depot, distance_matrix)

    opt_dist_km = opt_dist_m / 1000.0
    csv_dist_km = csv_res["distance_km"]
    nn_dist_km = nn_res["distance_km"]

    csv_reduction = ((csv_dist_km - opt_dist_km) / csv_dist_km * 100) if csv_dist_km > 0 else 0
    nn_reduction = ((nn_dist_km - opt_dist_km) / nn_dist_km * 100) if nn_dist_km > 0 else 0

    return {
        "optimized_route": optimized_route,
        "optimized_distance_km": opt_dist_km,
        "csv_route": csv_res["route"],
        "csv_distance_km": csv_dist_km,
        "nn_route": nn_res["route"],
        "nn_distance_km": nn_dist_km,
        "csv_reduction_pct": csv_reduction,
        "nn_reduction_pct": nn_reduction,
        "execution_time_sec": round(exec_time, 3),
        "distance_matrix": distance_matrix,
        "locations_visited": num_locations,
        "depot": depot,
        "status": "Optimal/Feasible solution found"
    }


# ==========================================
# 4. FORECAST INPUT & PRIORITY ENGINE
# ==========================================

def get_season(month):
    if month in (12, 1, 2):
        return "Winter"
    elif month in (3, 4, 5):
        return "Summer"
    elif month in (6, 7, 8, 9):
        return "Monsoon"
    return "Post-Monsoon"


def assign_priority_label(percentile):
    if percentile >= 75:
        return "High"
    elif percentile >= 25:
        return "Medium"
    return "Low"


def build_forecast_input_df(pipeline_dict, df_waste, forecast_year, forecast_month, waste_type_select, user_overrides=None):
    """
    Constructs a guaranteed valid, non-empty pandas DataFrame for forecasting containing
    every required feature column specified by the saved model metadata.
    """
    if pipeline_dict is None:
        raise ValueError("Pipeline dictionary is required.")

    feature_names = pipeline_dict["features"]
    numeric_features = pipeline_dict.get("numeric_features", [])
    categorical_features = pipeline_dict.get("categorical_features", [])

    if df_waste is not None and not df_waste.empty:
        chennai = df_waste[df_waste["City"].astype(str).str.strip().str.casefold() == "chennai"].copy()
    else:
        chennai = pd.DataFrame()

    if not chennai.empty:
        chennai = chennai.dropna(subset=["Waste_Type"]).sort_values(["Year", "Month"])
        base_records = chennai.groupby("Waste_Type", sort=False, group_keys=False).tail(1).copy().reset_index(drop=True)
    else:
        base_records = pd.DataFrame()

    # Filter to specific waste type if requested
    if waste_type_select and waste_type_select != "All Waste Types":
        if not base_records.empty and waste_type_select in base_records["Waste_Type"].values:
            base_records = base_records[base_records["Waste_Type"] == waste_type_select].copy()
        else:
            # Fallback 1-row DataFrame if selected waste type is not in baseline dataset
            row_data = {
                "Year": forecast_year,
                "Month": forecast_month,
                "Season": get_season(forecast_month),
                "Waste_Type": waste_type_select,
                "Population": 7100000,
                "Population_Density": 17000,
                "Urbanization_Rate": 87.5,
                "Air_Quality_Index": 110,
                "Rainfall_MM": 125.0,
                "Tourism_Impact_Score": 6.5,
                "Industrial_Activity_Level": 8.0,
                "Population_Growth_Rate": 1.0
            }
            base_records = pd.DataFrame([row_data])

    if base_records.empty:
        # Emergency fallback if base_records is empty to guarantee shape[0] >= 1
        known_types = ["Organic", "Construction", "Plastic", "Paper", "Glass", "Metal", "Biomedical", "Hazardous"]
        target_types = [waste_type_select] if waste_type_select and waste_type_select != "All Waste Types" else known_types
        rows = []
        for wt in target_types:
            rows.append({
                "Year": forecast_year,
                "Month": forecast_month,
                "Season": get_season(forecast_month),
                "Waste_Type": wt,
                "Population": 7100000,
                "Population_Density": 17000,
                "Urbanization_Rate": 87.5,
                "Air_Quality_Index": 110,
                "Rainfall_MM": 125.0,
                "Tourism_Impact_Score": 6.5,
                "Industrial_Activity_Level": 8.0,
                "Population_Growth_Rate": 1.0
            })
        base_records = pd.DataFrame(rows)

    # Apply date & season
    if "Year" in base_records.columns:
        base_records["Year"] = forecast_year
    if "Month" in base_records.columns:
        base_records["Month"] = forecast_month
    if "Season" in base_records.columns:
        base_records["Season"] = get_season(forecast_month)

    # Apply user overrides if provided
    if user_overrides:
        for k, v in user_overrides.items():
            if k in base_records.columns:
                base_records[k] = v

    # Ensure every expected feature in model metadata exists
    for feat in feature_names:
        if feat not in base_records.columns:
            base_records[feat] = 0

    # Extract exact required features in exact column order
    X_in = base_records[feature_names].copy().reset_index(drop=True)

    # Type safety enforcement
    for num_col in numeric_features:
        if num_col in X_in.columns:
            X_in[num_col] = pd.to_numeric(X_in[num_col], errors="coerce").fillna(0)

    for cat_col in categorical_features:
        if cat_col in X_in.columns:
            X_in[cat_col] = X_in[cat_col].astype(str)

    if X_in.empty or len(X_in) == 0:
        raise ValueError("Forecast input DataFrame has 0 sample(s).")

    return base_records, X_in


def generate_forecast_priorities(year, month):
    """Generates waste generation forecast and priority classification using trained ML model."""
    pipeline_dict = load_forecasting_pipeline()
    df_waste = load_waste_data()

    if pipeline_dict is None:
        return None

    model = pipeline_dict["model"]
    preprocessor = pipeline_dict["preprocessor"]

    records, X_in = build_forecast_input_df(
        pipeline_dict=pipeline_dict,
        df_waste=df_waste,
        forecast_year=year,
        forecast_month=month,
        waste_type_select="All Waste Types"
    )

    X_proc = preprocessor.transform(X_in)
    preds = np.maximum(model.predict(X_proc), 0)

    report = pd.DataFrame({
        "City": "Chennai",
        "Waste_Type": records["Waste_Type"].astype(str),
        "Forecast_Year": year,
        "Forecast_Month": month,
        "Estimated_Tons_Per_Day": np.round(preds, 2)
    })

    report["Priority_Percentile"] = report["Estimated_Tons_Per_Day"].rank(method="average", pct=True).mul(100).round(2)
    report["Collection_Priority"] = report["Priority_Percentile"].apply(assign_priority_label)

    report = report.sort_values("Estimated_Tons_Per_Day", ascending=False).reset_index(drop=True)
    return report


# ==========================================
# 5. GEOGRAPHIC OUTLIER DETECTOR
# ==========================================

def detect_isolated_bins(df_sub, threshold_km=2.0):
    """Detects bin locations further than `threshold_km` from the area center."""
    if df_sub.empty or len(df_sub) < 3:
        return pd.DataFrame()

    center_lat = df_sub["LATITUDE"].median()
    center_lon = df_sub["LONGTITUDE"].median()

    dists = df_sub.apply(
        lambda row: calculate_haversine_km(center_lat, center_lon, row["LATITUDE"], row["LONGTITUDE"]),
        axis=1
    )

    outliers = df_sub[dists > threshold_km].copy()
    if not outliers.empty:
        outliers["Distance_From_Center_km"] = dists[dists > threshold_km].round(3)
    return outliers
