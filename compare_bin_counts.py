import pandas as pd

# Load bin dataset
bins = pd.read_csv("dataset/Chennai_Bins.csv")

# Load cleaned collection dataset
collection = pd.read_csv(
    "dataset/Garbage_collection_clean.csv",
    encoding="latin1"
)

# Clean zone/division codes
bins["ZONE_CODE"] = bins["ZONE_CODE"].astype(str).str.strip()
bins["DIV_CODE"] = bins["DIV_CODE"].astype(str).str.strip()

# Convert N01 -> 1 and N001 -> 1
bins["Zone"] = (
    bins["ZONE_CODE"]
    .str.extract(r"(\d+)")
    .astype(int)
)

bins["Div"] = (
    bins["DIV_CODE"]
    .str.extract(r"(\d+)")
    .astype(int)
)

# Count physical bin records per Zone + Division
bin_counts = (
    bins.groupby(["Zone", "Div"])
    .size()
    .reset_index(name="KML_BIN_COUNT")
)

# Aggregate collection records by Zone + Division
collection_counts = (
    collection.groupby(["Zone", "Div"])["NO.OF BINS"]
    .sum()
    .reset_index(name="COLLECTION_BIN_COUNT")
)

# Compare
comparison = pd.merge(
    collection_counts,
    bin_counts,
    on=["Zone", "Div"],
    how="inner"
)

comparison["DIFFERENCE"] = (
    comparison["KML_BIN_COUNT"]
    - comparison["COLLECTION_BIN_COUNT"]
)

print("Total Zone + Division combinations:", len(comparison))

print("\nFirst 20 comparisons:")
print(comparison.head(20).to_string(index=False))

print("\nExact matches:")
print(
    (comparison["DIFFERENCE"] == 0).sum()
)

print("\nKML count greater than collection count:")
print(
    (comparison["DIFFERENCE"] > 0).sum()
)

print("\nCollection count greater than KML count:")
print(
    (comparison["DIFFERENCE"] < 0).sum()
)

print("\nDifference statistics:")
print(comparison["DIFFERENCE"].describe())