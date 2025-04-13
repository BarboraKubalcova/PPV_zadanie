import json

def extract_unique_questions(json_file_path, output_file="/home/js389cw/DP/mimicCxr/CheXagentDev/uniformDataset/unique_questions_validate.txt"):
    with open(json_file_path, "r", encoding="utf-8") as file:
        json_data = json.load(file)
    
    unique_questions = set()
    
    for item in json_data:
        unique_questions.add(item["question"])
    
    with open(output_file, "w", encoding="utf-8") as file:
        for question in unique_questions:
            file.write(question + "\n")
    
    print(f"Unique questions saved to {output_file}")

# Example usage
json_file_path = "/home/js389cw/dataset/slake/Slake1.0/inverted_en_close_validate.json"  # Change this to the path of your JSON file
extract_unique_questions(json_file_path)