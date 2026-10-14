import xml.etree.ElementTree as ET
import pandas as pd

kml_file = "dataset/Chennai_Bins.kml"

namespace = {
    "k": "http://www.opengis.net/kml/2.2"
}

tree = ET.parse(kml_file)
root = tree.getroot()

records = []

for placemark in root.findall(".//k:Placemark", namespace):

    schema_data = placemark.find(
        ".//k:SchemaData",
        namespace
    )

    if schema_data is None:
        continue

    values = {}

    for simple_data in schema_data.findall(
        "k:SimpleData",
        namespace
    ):
        field_name = simple_data.get("name")
        field_value = simple_data.text
        values[field_name] = field_value

    records.append(values)

df = pd.DataFrame(records)

print("Total bin records:", len(df))

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())

df.to_csv(
    "dataset/Chennai_Bins.csv",
    index=False
)

print("\nChennai_Bins.csv created successfully!")