import os
import sys
import json
import copy as cp
from prompt_toolkit import prompt

def specify_item_var(itm):
    os.system('cls' if os.name == 'nt' else 'clear')
    print(json.dumps(itm, indent=4))
    value = prompt("Please enter correct variation: ", default=itm['variation'])
    if value:
        itm['variation'] = value.strip()
    print(json.dumps(itm, indent=4))

    return itm

if __name__ == '__main__':
    prefix = os.path.dirname(os.path.realpath(sys.argv[0]))
    json_input = prefix + "/custom_dataset/variations.json"
    json_output = prefix + "/custom_dataset/variations.fixed.json"
    lower_id = 200
    upper_id = 1200

    if os.path.isfile(json_output + ".idx_file"):
        with open(json_output + ".idx_file", 'r') as f:
            load_index = int(f.read())
    else:
        load_index = 0

    out_index = cp.deepcopy(load_index)

    with open(json_input, 'r', encoding='utf-8') as file:
        data = json.load(file)

    data = list(sorted(filter(lambda dataset_item: lower_id <= dataset_item['question_id'] < upper_id, data), key=lambda entry: entry['question_id']))
    full_data_length = len(data)
    data = data[load_index:]

    should_break = False
    for idx, item in enumerate(data):
        cont = "n"
        while cont == "n":
            new_item = specify_item_var(cp.deepcopy(item))
            cont = input("Confirm ((y or just enter)/n/(q to write current last index (not including this one) and write the changes to file)): ")
            if cont == "y" or not cont:
                data[idx] = new_item
                out_index += 1
            elif cont == "q":
                should_break = True
                break
        if should_break:
            print(f"Last index: {out_index}")
            break

    if os.path.exists(json_output):
        with open(json_output, 'r', encoding='utf-8') as file:
            output_data = json.load(file)
    else:
        output_data = []

    output_data = output_data + data[:out_index - load_index]
    print(f"Processed {len(output_data)} items, {full_data_length - len(output_data)} items left.")

    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    with open(json_output + ".idx_file", 'w', encoding='utf-8') as file:
        file.write(str(out_index))