import pandas as pd

# Load datasets
bins = pd.read_csv("dataset/Chennai_Bins.csv")

collection = pd.read_csv(
    "dataset/Garbage_collection_clean.csv",
    encoding="latin1"
)

# Clean whitespace from zone/division codes
bins["ZONE_CODE"] = bins["ZONE_CODE"].astype(str).str.strip()
bins["DIV_CODE"] = bins["DIV_CODE"].astype(str).str.strip()

# Convert N01 -> 1, N088 -> 88
bins["ZONE_NUM"] = (
    bins["ZONE_CODE"]
    .str.extract(r"(\d+)")
    .astype(int)
)

bins["DIV_NUM"] = (
    bins["DIV_CODE"]
    .str.extract(r"(\d+)")
    .astype(int)
)

# Get divisions covered by collection data
collection_divisions = set(collection["Div"].unique())

# Find matching bin records
matching_bins = bins[
    bins["DIV_NUM"].isin(collection_divisions)
].copy()

print("Total bin records:", len(bins))

print("Bin records in collection divisions:", len(matching_bins))

print(
    "Collection divisions:",
    len(collection_divisions)
)

print(
    "Matching divisions found:",
    matching_bins["DIV_NUM"].nunique()
)

print("\nZones in matching bin records:")
print(sorted(matching_bins["ZONE_NUM"].unique()))

print("\nFirst 10 matching bin records:")
print(
    matching_bins[
        ["ZONE_CODE", "DIV_CODE", "ZONE_NUM", "DIV_NUM",
         "BIN_NO", "LATITUDE", "LONGTITUDE"]
    ].head(10)
)