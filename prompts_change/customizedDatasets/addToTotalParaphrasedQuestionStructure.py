import json
import os

# Define file paths (modify as needed)
input_json_path = "/home/js389cw/dataset/slake/Slake1.0/filtered_xrays_test_en_open_plus_paraphrased.json"  # JSON file containing questions
paraphrased_json_path = "/home/js389cw/dataset/slake/Slake1.0/unique_questions_open_map.json"  # JSON file containing paraphrased versions
output_json_path = "/home/js389cw/dataset/slake/Slake1.0/filtered_xrays_test_en_open_plus_paraphrasedDone.json"  # Output file with added question2 field

# Check if input files exist
if not os.path.exists(input_json_path):
    print(f"Error: Input JSON file '{input_json_path}' not found.")
    exit(1)

if not os.path.exists(paraphrased_json_path):
    print(f"Error: Paraphrased JSON file '{paraphrased_json_path}' not found.")
    exit(1)

# Load the input JSON data
with open(input_json_path, "r", encoding="utf-8") as f:
    input_data = json.load(f)

# Load the paraphrased questions JSON data
with open(paraphrased_json_path, "r", encoding="utf-8") as f:
    paraphrased_data = json.load(f)

# Create a dictionary mapping original questions to paraphrased ones (normalize for consistency)
paraphrase_dict = {entry["original"].strip().lower(): entry["paraphrased"] for entry in paraphrased_data}

# Debugging: Print some mappings
print("Sample Mappings from Paraphrased JSON:")
for key in list(paraphrase_dict.keys())[:5]:
    print(f"Original: {key} -> Paraphrased: {paraphrase_dict[key]}")

# Update input data by adding "question2" parameter
missing_questions = []
for entry in input_data:
    normalized_question = entry["question"].strip().lower()  # Normalize question lookup
    if normalized_question in paraphrase_dict:
        entry["question2"] = paraphrase_dict[normalized_question]
    else:
        entry["question2"] = entry["question"]  # Fallback to original if no paraphrase found
        missing_questions.append(entry["question"])  # Collect missing mappings

# Debugging: Print missing questions
if missing_questions:
    print("\n⚠️ The following questions were not found in the paraphrase dictionary:")
    for missing_q in missing_questions[:5]:  # Show only first few to avoid clutter
        print(f"  - {missing_q}")

# Save the updated data to a new JSON file
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(input_data, f, indent=4, ensure_ascii=False)

print(f"\n✅ Updated JSON file saved as: {output_json_path}")
