import json
json_path = "data/test_data_original_and_inverted.json"
output_path = "data"

def filter_data(data):
    filtered_data = [
        entry for entry in data if entry['answer'].strip().lower() in {'yes', 'no'}
    ]
    return filtered_data


def read_vqa_json(json_input, from_file=True):
    if from_file:
        with open(json_input, 'r', encoding='utf-8') as file:
            data = json.load(file)
    else:
        data = json.loads(json_input)
    data = filter_data(data)
    return data


def save_json_to_file(data, filename):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Data successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving file: {e}")


def create_dataset():
    """
    Funkcia spoji invertovany a originalny testovaci dataset
    """
    test_dataset_inverted_path = "../prompts_change/external_data/inverted_en_close_test.json"
    test_dataset_original_path = "../dataset/Slake/Slake1.0/test.json"


    inverted_data = read_vqa_json(test_dataset_inverted_path)
    selected_qid = list(set([entry['qid'] for entry in inverted_data]))

    original_test_data_all = read_vqa_json(test_dataset_original_path)
    original_data = [
        entry for entry in original_test_data_all
        if entry['qid'] in selected_qid
    ]
    data_merged = inverted_data + original_data
    save_json_to_file(data_merged, json_path)
    save_json_to_file(original_data, f"{output_path}/original_test_data_filtered.json")
    save_json_to_file(original_data, f"{output_path}/original_test_data_filtered.json")
    save_json_to_file(inverted_data, f"{output_path}/inverted_test_data_filtered.json")


def main():
    # filtered data znamena, ze su odfitrovane len tie otazky, na ktore je odpoved yes/no
    create_dataset()


if __name__ == "__main__":
    main()
