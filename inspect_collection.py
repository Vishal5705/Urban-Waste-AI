import pandas as pd

df = pd.read_csv(
    "dataset/Garbage_collection.csv",
    encoding="latin1"
)

print("Original records:", len(df))

# Remove exact duplicate rows
df_clean = df.drop_duplicates().copy()

print("Cleaned records:", len(df_clean))
print("Duplicates removed:", len(df) - len(df_clean))

# Save cleaned dataset
df_clean.to_csv(
    "dataset/Garbage_collection_clean.csv",
    index=False
)

print("\nGarbage_collection_clean.csv created successfully!")