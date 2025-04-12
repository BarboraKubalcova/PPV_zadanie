import json
from collections import defaultdict

def count_questions_by_group(questions_file, groups_file, output_file="/home/js389cw/DP/mimicCxr/CheXagentDev/uniformDataset/group_counts.json"):

    # Load the questions data
    with open(questions_file, "r", encoding="utf-8") as f:
        questions_data = json.load(f)
    
    # Load the question groups
    with open(groups_file, "r", encoding="utf-8") as f:
        question_groups = json.load(f)
    
    # Create a dictionary to count occurrences of questions in each group
    group_counts = {group: 0 for group in question_groups.keys()}
    
    # Iterate through the dataset and categorize questions
    for question_entry in questions_data:
        question_text = question_entry["question"]
        
        # Check which group the question belongs to and increment count
        for group, questions in question_groups.items():
            if question_text in questions:
                group_counts[group] += 1
                break  # Stop checking once we find the group
    
    # Save the result to a JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(group_counts, f, indent=4)
    
    print(f"Question counts per group saved to {output_file}")


# Example usage
questions_file = "/home/js389cw/dataset/slake/Slake1.0/inverted_en_close_total.json"  # Path to JSON file with all unique questions
groups_file = "/home/js389cw/DP/mimicCxr/CheXagentDev/uniformDataset/grouped_unique_questions.json" # Path to JSON file with semantic groups
count_questions_by_group(questions_file, groups_file)