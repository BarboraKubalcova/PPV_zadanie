import json

# Input file name (Replace with your actual file name)
input_file = "/home/js389cw/dataset/slake/Slake1.0/filtered_xrays_validate_en.json"  

# Output file names
output_open_file = "/home/js389cw/dataset/slake/Slake1.0/filtered_xrays_validate_en_open.json"
output_closed_file = "/home/js389cw/dataset/slake/Slake1.0/filtered_xrays_validate_en_close.json"

# Load JSON data
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Separate entries into OPEN and CLOSED
open_ended = [entry for entry in data if entry.get("answer_type") == "OPEN"]
closed_ended = [entry for entry in data if entry.get("answer_type") == "CLOSED"]

# Save the filtered data into separate files
with open(output_open_file, "w", encoding="utf-8") as f:
    json.dump(open_ended, f, indent=4, ensure_ascii=False)

with open(output_closed_file, "w", encoding="utf-8") as f:
    json.dump(closed_ended, f, indent=4, ensure_ascii=False)

print(f"Filtered files saved as:\n- {output_open_file} (OPEN-ended questions)\n- {output_closed_file} (CLOSED-ended questions)")

