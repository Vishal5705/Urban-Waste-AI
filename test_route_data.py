import pandas as pd

# Load Chennai bin data
bins = pd.read_csv("dataset/Chennai_Bins.csv")

# Clean Zone and Division codes
bins["ZONE_CODE"] = bins["ZONE_CODE"].astype(str).str.strip()
bins["DIV_CODE"] = bins["DIV_CODE"].astype(str).str.strip()

# Select Zone 1, Division 1
division_bins = bins[
    (bins["ZONE_CODE"] == "N01") &
    (bins["DIV_CODE"] == "N001")
].copy()

print("Zone 1 - Division 1")
print("Number of bin locations:", len(division_bins))

print("\nBin locations:")
print(
    division_bins[
        ["BIN_NO", "LATITUDE", "LONGTITUDE", "LOC_NAME", "STREET_NAM"]
    ].to_string(index=False)
)

# Save the test routing dataset
division_bins.to_csv(
    "dataset/Route_Test_Division_1.csv",
    index=False
)

print("\nRoute_Test_Division_1.csv created successfully!")