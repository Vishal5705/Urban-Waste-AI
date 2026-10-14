
import pandas as pd
import matplotlib.pyplot as plt

from math import radians, sin, cos, asin, sqrt

from ortools.constraint_solver import (
    pywrapcp,
    routing_enums_pb2
)

# ==========================================
# 1. LOAD AND VALIDATE T. NAGAR DATA
# ==========================================

INPUT_PATH = "dataset/Route_Test_TNagar.csv"

df = pd.read_csv(INPUT_PATH)

required_columns = [
    "OBJECTID",
    "DIV_CODE",
    "AREA_NAME",
    "LOC_NAME",
    "STREET_NAM",
    "LATITUDE",
    "LONGTITUDE"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if df[["LATITUDE", "LONGTITUDE"]].isnull().any().any():
    raise ValueError("Latitude or longitude contains missing values.")

if len(df) < 2:
    raise ValueError("At least two bin locations are required.")

if not df["LATITUDE"].between(-90, 90).all():
    raise ValueError("Invalid latitude values found.")

if not df["LONGTITUDE"].between(-180, 180).all():
    raise ValueError("Invalid longitude values found.")

if df[["LATITUDE", "LONGTITUDE"]].duplicated().any():
    raise ValueError("Duplicate coordinates found.")

df = df.reset_index(drop=True)

num_locations = len(df)
depot = 0  # Temporary depot: first bin in the dataset

print("=" * 55)
print("T. NAGAR WASTE COLLECTION ROUTE OPTIMIZER")
print("=" * 55)
print("Division:", df["DIV_CODE"].iloc[0])
print("Area:", df["AREA_NAME"].iloc[0])
print("Bin locations:", num_locations)
print("Temporary depot: first bin in the dataset")
print("Dataset validation: successful")

# ==========================================
# 2. HAVERSINE DISTANCE CALCULATION
# ==========================================

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate straight-line distance between
    two coordinates in kilometres.
    """

    lat1, lon1, lat2, lon2 = map(
        radians,
        [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1)
        * cos(lat2)
        * sin(dlon / 2) ** 2
    )

    # Protect against small floating-point errors
    a = max(0.0, min(1.0, a))

    c = 2 * asin(sqrt(a))

    return 6371 * c


# Store each bin's coordinates
locations = list(
    zip(df["LATITUDE"], df["LONGTITUDE"])
)

# Build the distance matrix in integer metres
distance_matrix = []

for lat1, lon1 in locations:
    row = []

    for lat2, lon2 in locations:
        km = calculate_distance(
            lat1, lon1, lat2, lon2
        )

        row.append(round(km * 1000))

    distance_matrix.append(row)

print("\nDistance matrix created:")
print("Dimensions:", len(distance_matrix), "x", len(distance_matrix[0]))
print("Distance from first bin to itself:", distance_matrix[0][0], "metres")
print(
    "Distance from bin 1 to bin 2:",
    distance_matrix[0][1],
    "metres"
)

# ==========================================
# 3. ROUTE METRIC HELPERS
# ==========================================

def route_distance(route):
    """
    Calculate total route distance in metres.
    The route includes the return to the depot.
    """

    return sum(
        distance_matrix[route[i]][route[i + 1]]
        for i in range(len(route) - 1)
    )


def display_route(route):
    """
    Display bin visit order using 1-based numbering.
    """

    return " -> ".join(
        str(node + 1) for node in route
    )


# Quick sanity checks
test_route = [0, 1, 0]

print("\nRoute helper checks:")
print("Test route:", display_route(test_route))
print(
    "Test route distance:",
    round(route_distance(test_route) / 1000, 3),
    "km"
)

# ==========================================
# 4. BASELINE: CSV ORDER
# ==========================================

# Visit every bin in its original dataset order,
# then return to the temporary depot.
csv_route = list(range(num_locations)) + [depot]

csv_distance = route_distance(csv_route)

print("\n" + "=" * 55)
print("BASELINE ROUTE: CSV ORDER")
print("=" * 55)
print("Locations visited:", len(csv_route) - 1)
print("Returns to depot: Yes")
print(
    "Total straight-line distance:",
    round(csv_distance / 1000, 3),
    "km"
)
print("First 10 visits:", display_route(csv_route[:10]))

# ==========================================
# 5. BASELINE: NEAREST NEIGHBOR
# ==========================================

unvisited = set(range(num_locations))
unvisited.remove(depot)

nearest_route = [depot]
current = depot

while unvisited:
    next_location = min(
        unvisited,
        key=lambda node: distance_matrix[current][node]
    )

    nearest_route.append(next_location)
    unvisited.remove(next_location)
    current = next_location

# Return to the temporary depot
nearest_route.append(depot)

nearest_distance = route_distance(nearest_route)

print("\n" + "=" * 55)
print("BASELINE ROUTE: NEAREST NEIGHBOR")
print("=" * 55)
print("Locations visited:", len(nearest_route) - 1)
print("Returns to depot: Yes")
print(
    "Total straight-line distance:",
    round(nearest_distance / 1000, 3),
    "km"
)
print("First 10 visits:", display_route(nearest_route[:10]))

# ==========================================
# 6. OR-TOOLS ROUTE OPTIMIZATION
# ==========================================

manager = pywrapcp.RoutingIndexManager(
    num_locations,
    1,       # One vehicle for this test
    depot
)

routing = pywrapcp.RoutingModel(manager)


def distance_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)

    return distance_matrix[from_node][to_node]


callback_index = routing.RegisterTransitCallback(
    distance_callback
)

routing.SetArcCostEvaluatorOfAllVehicles(callback_index)

search_parameters = pywrapcp.DefaultRoutingSearchParameters()

search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
)

search_parameters.local_search_metaheuristic = (
    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
)

# Allow up to 10 seconds to improve the route
search_parameters.time_limit.seconds = 10

print("\nRunning OR-Tools optimization...")

solution = routing.SolveWithParameters(search_parameters)

if solution is None:
    raise RuntimeError("OR-Tools could not find a route.")

optimized_route = []
index = routing.Start(0)

while not routing.IsEnd(index):
    optimized_route.append(manager.IndexToNode(index))
    index = solution.Value(routing.NextVar(index))

optimized_route.append(depot)

optimized_distance = route_distance(optimized_route)

print("\n" + "=" * 55)
print("OR-TOOLS OPTIMIZED ROUTE")
print("=" * 55)
print("Locations visited:", len(optimized_route) - 1)
print("Returns to depot: Yes")
print(
    "Total straight-line distance:",
    round(optimized_distance / 1000, 3),
    "km"
)
print("First 10 visits:", display_route(optimized_route[:10]))

# ==========================================
# 7. COMPARE ROUTING METHODS
# ==========================================

results = [
    {
        "Method": "CSV-order baseline",
        "Distance_km": csv_distance / 1000
    },
    {
        "Method": "Nearest-neighbor baseline",
        "Distance_km": nearest_distance / 1000
    },
    {
        "Method": "OR-Tools optimized",
        "Distance_km": optimized_distance / 1000
    }
]

results_df = pd.DataFrame(results)

print("\n" + "=" * 55)
print("T. NAGAR ROUTE DISTANCE COMPARISON")
print("=" * 55)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Distance_km": "{:.3f}".format
        }
    )
)

for name, baseline_distance in [
    ("CSV-order baseline", csv_distance),
    ("Nearest-neighbor baseline", nearest_distance)
]:
    reduction = (
        (baseline_distance - optimized_distance)
        / baseline_distance * 100
    ) if baseline_distance else 0

    print(
        f"\nDistance reduction vs {name}: "
        f"{reduction:.2f}%"
    )

# ==========================================
# 8. SAVE ROUTE RESULTS
# ==========================================

from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

results_df.to_csv(
    OUTPUT_DIR / "tnagar_route_comparison.csv",
    index=False
)

route_df = pd.DataFrame({
    "Visit_Order": range(1, len(optimized_route) + 1),
    "OBJECTID": [
        df.iloc[node]["OBJECTID"]
        for node in optimized_route
    ],
    "DIV_CODE": [
        df.iloc[node]["DIV_CODE"]
        for node in optimized_route
    ],
    "AREA_NAME": [
        df.iloc[node]["AREA_NAME"]
        for node in optimized_route
    ],
    "LOC_NAME": [
        df.iloc[node]["LOC_NAME"]
        for node in optimized_route
    ],
    "STREET_NAM": [
        df.iloc[node]["STREET_NAM"]
        for node in optimized_route
    ],
    "LATITUDE": [
        df.iloc[node]["LATITUDE"]
        for node in optimized_route
    ],
    "LONGTITUDE": [
        df.iloc[node]["LONGTITUDE"]
        for node in optimized_route
    ]
})

route_df.to_csv(
    OUTPUT_DIR / "tnagar_optimized_route.csv",
    index=False
)

print("\nRoute files saved successfully.")


# ==========================================
# 9. VISUALIZE THE OPTIMIZED ROUTE
# ==========================================

route_lats = [
    df.iloc[node]["LATITUDE"]
    for node in optimized_route
]

route_lons = [
    df.iloc[node]["LONGTITUDE"]
    for node in optimized_route
]

plt.figure(figsize=(12, 9))

plt.plot(
    route_lons,
    route_lats,
    marker="o",
    markersize=3,
    linewidth=1,
    label="OR-Tools route"
)

plt.scatter(
    route_lons[0],
    route_lats[0],
    marker="*",
    s=200,
    label="Temporary depot"
)

plt.title("T. Nagar Waste Collection Route")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "tnagar_route_visualization.png",
    dpi=300
)


# ==========================================
# 10. VISUALIZE ROUTE COMPARISON
# ==========================================

plt.figure(figsize=(10, 6))

plt.bar(
    results_df["Method"],
    results_df["Distance_km"]
)

plt.ylabel("Straight-line distance (km)")
plt.title("T. Nagar Route Method Comparison")
plt.xticks(rotation=10)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "tnagar_route_comparison.png",
    dpi=300
)

print("\nGenerated files:")
print("- outputs/tnagar_route_comparison.csv")
print("- outputs/tnagar_optimized_route.csv")
print("- outputs/tnagar_route_visualization.png")
print("- outputs/tnagar_route_comparison.png")

plt.show()

# ==========================================
# CHECK GEOGRAPHICALLY ISOLATED LOCATIONS
# ==========================================

print("\nChecking geographically isolated locations...")

# Approximate center of the main T. Nagar cluster
center_lat = df["LATITUDE"].median()
center_lon = df["LONGTITUDE"].median()

# Approximate distance from the center using Haversine
distances_from_center = df.apply(
    lambda row: calculate_distance(
        center_lat,
        center_lon,
        row["LATITUDE"],
        row["LONGTITUDE"]
    ),
    axis=1
)

# Flag locations more than 2 km from the center
isolated_locations = df[
    distances_from_center > 2
].copy()

isolated_locations["Distance_From_Center_km"] = (
    distances_from_center[distances_from_center > 2].round(3)
)

print(f"Locations checked: {len(df)}")
print(f"Potentially isolated locations: {len(isolated_locations)}")

if not isolated_locations.empty:
    print("\nReview these locations:")
    print(
        isolated_locations[
            [
                "OBJECTID",
                "AREA_NAME",
                "LOC_NAME",
                "STREET_NAM",
                "LATITUDE",
                "LONGTITUDE",
                "Distance_From_Center_km"
            ]
        ].to_string(index=False)
    )

    isolated_locations.to_csv(
        "outputs/tnagar_isolated_locations_review.csv",
        index=False
    )

    print(
        "\nSaved review file: "
        "outputs/tnagar_isolated_locations_review.csv"
    )
