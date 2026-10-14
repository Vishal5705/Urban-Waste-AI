import pandas as pd

df = pd.read_csv("dataset/Chennai_Bins.csv")

print("Total records:", len(df))

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nUnique zones:")
print(df["ZONE_CODE"].nunique())

print("\nUnique divisions:")
print(df["DIV_CODE"].nunique())

print("\nLatitude range:")
print(df["LATITUDE"].min(), "to", df["LATITUDE"].max())

print("\nLongitude range:")
print(df["LONGTITUDE"].min(), "to", df["LONGTITUDE"].max())

print("\nSample records:")
print(df.head(10))