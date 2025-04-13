import json

def remove_zh_entries(input_file, output_file):
    # Load the JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter out entries where q_lang is 'zh'
    filtered_data = [entry for entry in data if entry.get("q_lang") != "zh"]
    
    # Save the cleaned data to a new file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=4)
    
    print(f"Filtered JSON saved to {output_file}")

# Example usage
input_file = "/home/js389cw/dataset/slake/Slake1.0/filtered_xrays.json"   # Replace with your input file
output_file = "/home/js389cw/dataset/slake/Slake1.0/filtered_xrays_train_en.json"  # Replace with your desired output file
remove_zh_entries(input_file, output_file)