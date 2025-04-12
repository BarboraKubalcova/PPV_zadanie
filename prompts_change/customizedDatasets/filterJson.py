import json
import pandas as pd

# Define file paths
json_path = "/home/js389cw/dataset/slake/Slake1.0/validate.json"
filtered_json_path = "/home/js389cw/dataset/slake/Slake1.0/filtered_xrays_validate.json"

# Load JSON file
with open(json_path, "r") as file:
    data = json.load(file)

# Convert to DataFrame
df = pd.DataFrame(data)

# Print available modality values
print("Unique modality values:", df["modality"].unique())

# Check the correct spelling of X-ray in the dataset
# Example: If X-rays are stored as "X-Ray", update the filter accordingly
filtered_df = df[df["modality"].str.lower() == "x-ray"]  # Normalize for case sensitivity

# Convert back to JSON format
filtered_data = filtered_df.to_dict(orient="records")

# Save the filtered dataset
with open(filtered_json_path, "w") as file:
    json.dump(filtered_data, file, indent=4)

print(f"✅ Filtered dataset saved as '{filtered_json_path}' with {len(filtered_data)} X-ray images.")
