
import pandas as pd
import matplotlib.pyplot as plt
from math import radians, sin, cos, asin, sqrt
from ortools.constraint_solver import (
    pywrapcp,
    routing_enums_pb2
)

# ==========================================
# 1. LOAD AND VALIDATE DATA
# ==========================================

df = pd.read_csv("dataset/Route_Test_Division_1.csv")

required_columns = ["LATITUDE", "LONGTITUDE"]
if df[required_columns].isnull().any().any():
    raise ValueError("Latitude or longitude contains missing values.")

if len(df) < 2:
    raise ValueError("At least two locations are required.")

# Validate latitude and longitude ranges
if not df["LATITUDE"].between(-90, 90).all():
    raise ValueError("Invalid latitude values found.")

if not df["LONGTITUDE"].between(-180, 180).all():
    raise ValueError("Invalid longitude values found.")

num_locations = len(df)
depot = 0  # Temporary test depot: location 1

print("=" * 48)
print("WASTE COLLECTION ROUTE EVALUATION")
print("=" * 48)
print("Locations:", num_locations)
print("Temporary depot: Location 1")

# ==========================================
# 2. HAVERSINE DISTANCE
# ==========================================

def calculate_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(
        radians, [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    )

    c = 2 * asin(sqrt(a))
    return 6371 * c  # Kilometres


locations = list(zip(df["LATITUDE"], df["LONGTITUDE"]))

# Distances stored as integer metres for OR-Tools
distance_matrix = []

for lat1, lon1 in locations:
    row = []

    for lat2, lon2 in locations:
        km = calculate_distance(lat1, lon1, lat2, lon2)
        row.append(round(km * 1000))

    distance_matrix.append(row)

print("Distance matrix:", num_locations, "x", num_locations)

# ==========================================
# 3. COMMON ROUTE METRICS
# ==========================================

def route_distance(route):
    return sum(
        distance_matrix[route[i]][route[i + 1]]
        for i in range(len(route) - 1)
    )


def display_route(route):
    return " -> ".join(str(node + 1) for node in route)


# ==========================================
# 4. BASELINE 1: CSV ORDER
# ==========================================

csv_route = list(range(num_locations)) + [depot]
csv_distance = route_distance(csv_route)

# ==========================================
# 5. BASELINE 2: NEAREST NEIGHBOR
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

nearest_route.append(depot)
nearest_distance = route_distance(nearest_route)

# ==========================================
# 6. OR-TOOLS ROUTE OPTIMIZATION
# ==========================================

manager = pywrapcp.RoutingIndexManager(
    num_locations,
    1,       # One test vehicle
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

# Search for route improvements for up to 5 seconds
search_parameters.local_search_metaheuristic = (
    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
)
search_parameters.time_limit.seconds = 5

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

# ==========================================
# 7. COMPARE ALL THREE ROUTES
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

print("\n" + "=" * 48)
print("ROUTE DISTANCE COMPARISON")
print("=" * 48)
print(results_df.to_string(
    index=False,
    formatters={"Distance_km": "{:.3f}".format}
))

for name, baseline_distance in [
    ("CSV-order baseline", csv_distance),
    ("Nearest-neighbor baseline", nearest_distance)
]:
    reduction = (
        (baseline_distance - optimized_distance)
        / baseline_distance * 100
    ) if baseline_distance else 0

    print(f"\nReduction vs {name}: {reduction:.2f}%")

print("\n" + "=" * 48)
print("GENERATED ROUTE")
print("=" * 48)
print(display_route(optimized_route))
print(f"Locations visited: {len(optimized_route) - 1}")
print(f"Return to depot: Yes")
print(f"Total distance: {optimized_distance / 1000:.3f} km")

# ==========================================
# 8. SAVE RESULTS
# ==========================================

results_df.to_csv("route_comparison.csv", index=False)

route_df = pd.DataFrame({
    "Visit_Order": range(1, len(optimized_route) + 1),
    "Location_Number": [node + 1 for node in optimized_route],
    "Latitude": [
        df.iloc[node]["LATITUDE"] for node in optimized_route
    ],
    "Longitude": [
        df.iloc[node]["LONGTITUDE"] for node in optimized_route
    ]
})

route_df.to_csv("optimized_route.csv", index=False)

# ==========================================
# 9. PLOT THE OPTIMIZED ROUTE
# ==========================================

route_lats = [
    df.iloc[node]["LATITUDE"] for node in optimized_route
]
route_lons = [
    df.iloc[node]["LONGTITUDE"] for node in optimized_route
]

plt.figure(figsize=(10, 8))
plt.plot(
    route_lons,
    route_lats,
    marker="o",
    linewidth=1.5,
    label="OR-Tools route"
)

plt.scatter(
    route_lons[0],
    route_lats[0],
    color="red",
    marker="*",
    s=180,
    label="Temporary depot"
)

for order, node in enumerate(optimized_route[:-1], start=1):
    plt.annotate(
        str(node + 1),
        (df.iloc[node]["LONGTITUDE"], df.iloc[node]["LATITUDE"]),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8
    )

plt.title("OR-Tools Waste Collection Route")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("route_visualization.png", dpi=300)

# Save a comparison chart too
plt.figure(figsize=(9, 5))
plt.bar(
    results_df["Method"],
    results_df["Distance_km"]
)
plt.ylabel("Straight-line route distance (km)")
plt.title("Comparison of Routing Methods")
plt.xticks(rotation=10)
plt.tight_layout()
plt.savefig("route_comparison.png", dpi=300)

print("\nSaved files:")
print("- route_comparison.csv")
print("- optimized_route.csv")
print("- route_visualization.png")
print("- route_comparison.png")

plt.show()
