import json


def read_json(json_input, from_file=True):
    if from_file:
        with open(json_input, 'r', encoding='utf-8') as file:
            data = json.load(file)
    else:
        data = json.loads(json_input)
    return data


def save_json_to_file(data, filename):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Data successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving file: {e}")


def main():
    splits = ["test", "train", "validate"]
    external_data_path = "./external_data/"
    output_path = "./custom_dataset/"
    dataset_path = "../dataset/Slake/Slake1.0/"

    for split in splits:
        file = external_data_path + f"inverted_en_close_{split}.json"
        inverted_data = read_json(file)
        original_data = read_json(dataset_path+f"{split}.json")
        new_dataset = []
        original_lookup = {entry["qid"]: entry for entry in original_data if "qid" in entry}

        for entry in inverted_data:
            inverted_qid = entry.get("qid")  # question id
            original_entry = original_lookup.get(inverted_qid)
            # print(f"{inverted_qid=} {original_entry=}")
            if original_entry:
                # print("Original question:", original_entry.get("question"))
                # print("Inverted variation:", entry.get("question"))
                new_data_element = {
                    "question_id": inverted_qid,
                    "original": original_entry.get("question"),
                    "variation": entry.get("question"),
                    "is_negated": True
                }
                new_dataset.append(new_data_element)
        save_json_to_file(new_dataset, output_path+f"inverted_{split}.json")


if __name__ == "__main__":
    main()
