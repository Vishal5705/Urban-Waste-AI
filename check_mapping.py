import pandas as pd

# Load bin data
bins = pd.read_csv("dataset/Chennai_Bins.csv")

# Load cleaned collection data
collection = pd.read_csv(
    "dataset/Garbage_collection_clean.csv",
    encoding="latin1"
)

# Extract numeric part from bin zone/division codes
bins["ZONE_NUM"] = bins["ZONE_CODE"].str.extract(r"(\d+)").astype(int)
bins["DIV_NUM"] = bins["DIV_CODE"].str.extract(r"(\d+)").astype(int)

print("BIN DATA")
print("Unique zones:", sorted(bins["ZONE_NUM"].unique()))
print("Number of zones:", bins["ZONE_NUM"].nunique())

print("\nUnique divisions:", bins["DIV_NUM"].nunique())
print("Division range:", bins["DIV_NUM"].min(), "to", bins["DIV_NUM"].max())

print("\nCOLLECTION DATA")
print("Unique zones:", sorted(collection["Zone"].unique()))
print("Number of zones:", collection["Zone"].nunique())

print("\nUnique divisions:", collection["Div"].nunique())
print("Division range:", collection["Div"].min(), "to", collection["Div"].max())

# Compare mappings
bin_zones = set(bins["ZONE_NUM"].unique())
collection_zones = set(collection["Zone"].unique())

bin_divisions = set(bins["DIV_NUM"].unique())
collection_divisions = set(collection["Div"].unique())

print("\nZONE COMPARISON")
print("Zones in both:", sorted(bin_zones & collection_zones))
print("Zones only in bin data:", sorted(bin_zones - collection_zones))
print("Zones only in collection data:", sorted(collection_zones - bin_zones))

print("\nDIVISION COMPARISON")
print("Divisions in both:", len(bin_divisions & collection_divisions))
print("Divisions only in bin data:", sorted(bin_divisions - collection_divisions))
print("Divisions only in collection data:", sorted(collection_divisions - bin_divisions))

print("\nALL ORIGINAL ZONE CODES:")
print(sorted(bins["ZONE_CODE"].unique()))

print("\nALL ORIGINAL DIVISION CODES:")
print(sorted(bins["DIV_CODE"].unique()))