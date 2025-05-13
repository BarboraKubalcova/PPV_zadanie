import sys
sys.path.append('../models')
sys.path.append('../model')
sys.path.append('..')
import json
import os

from models.chexagent import CheXagent
from model.preprocessing_model import T5WithInversionHead
from task_prompting.T5PreprocessModel import T5PreprocessModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
from colours import bcolors

base_image_path = "../dataset/Slake/Slake1.0/imgs/"
mixed_file = "data/test_data_original_and_inverted.json"
inverted_data = "data/inverted_test_data_filtered.json"
original_data = "data/original_test_data_filtered.json"

# model = T5WithInversionHead.from_pretrained("../task_prompting/weights/weights_9.5.25")
t5_pre = T5PreprocessModel('../task_prompting/weights/weights_9.5.25')
vq_model = CheXagent()


def read_vqa_json(json_input, from_file=True):
    if from_file:
        with open(json_input, 'r', encoding='utf-8') as file:
            data = json.load(file)
    else:
        data = json.loads(json_input)
    return data

def flip_answer(ans):
    ans = ans.lower()
    if ans == "no":
        return "Yes"
    elif ans == "yes":
        return "No"
    else:
        raise Exception("Unsupported question type - yes/no questions only!")


def test_xeagent(data, output_file_name, with_correction=False):

    inverted = False
    results = []
    total_questions = 0
    correct_predictions = 0

    for entry in data:
        if total_questions % 10 == 0:
            print(f"Question {total_questions}/{len(data)}")

        if with_correction:
            entry_q = entry['question']
            canonize_str = f'<canonize> <variation> {entry_q}'
            neggated_str = f'<check_if_negated> <variation> {entry_q}'
            question = t5_pre.get_output(canonize_str)
            inverted = t5_pre.get_output(neggated_str)
        else:
            question = entry["question"]

        correct_answer = entry["answer"]
        image_path = os.path.join(base_image_path, entry["img_name"])

        if not os.path.exists(image_path):
            print(f"Warning: Image not found - {image_path}")
            continue

        prompt = f"Provide a direct answer to this question:\n{question}"
        model_answer = vq_model.generate(image_path, prompt).strip()
        model_answer = model_answer if not inverted else flip_answer(model_answer)

        is_correct = correct_answer.lower() in model_answer.lower()
        total_questions += 1
        if is_correct:
            correct_predictions += 1

        results.append({
            "question": entry["question"],
            "correct_answer": entry["answer"],
            "model_answer": model_answer,
            "match": is_correct
        })
        entry_q = entry["question"]
        entry_a = entry["answer"]
        # print(f"{bcolors.OKGREEN} Question:{bcolors.ENDC} {entry_q} ")
        # print(f"{bcolors.OKBLUE} Answer: {bcolors.ENDC} {model_answer}")
        # print(f"{bcolors.OKBLUE} Correct Answer: {bcolors.ENDC} {entry_a}\n")

    accuracy = (correct_predictions / total_questions) * 100 if total_questions > 0 else 0
    print(f"\nTotal Questions: {total_questions}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%\n")

    output_data = {
        "accuracy": accuracy,
        "total_questions": total_questions,
        "correct_predictions": correct_predictions,
        "results": results
    }

    with open(f"test_output/{output_file_name}", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    return output_data


def main():
    print("Testing Xagent on original testing data")
    data = read_vqa_json(original_data)
    output_on_original = test_xeagent(data, "original_test_results.json")

    print("\nTesting Xagent on inverted testing data without correction")
    data = read_vqa_json(inverted_data)
    output_without_correction = test_xeagent(data, "inverted_test_results_without_correction.json")

    print("\nTesting Xagent on inverted testing data with correction")
    output_with_correction = test_xeagent(data, "inverted_test_results_with_correction.json", with_correction=True)

    print("\nTesting Xagent on mixed testing data without correction")
    data = read_vqa_json(mixed_file)
    output_without_correction_mixed = test_xeagent(data, "mixed_test_results_without_correction.json")

    print("\nTesting Xagent on mixed testing data with correction")
    output_with_correction_mixed = test_xeagent(data, "mixed_test_results_with_correction.json", with_correction=True)

    out_orig = output_on_original["accuracy"]
    out_no_correction = output_without_correction["accuracy"]
    out_correction = output_with_correction["accuracy"]
    out_no_correction_mixed = output_without_correction_mixed["accuracy"]
    out_correction_mixed = output_with_correction_mixed["accuracy"]
    print("-------------------------------------------------------")
    print(f"{out_orig}: Accuracy for original questions without negation.")
    print(f"{out_no_correction}: Accuracy for inverted questions without correction.")
    print(f"{out_correction}: Accuracy for inverted questions with correction.")
    print(f"{out_no_correction_mixed}: Accuracy for mixed questions without correction.")
    print(f"{out_correction_mixed}: Accuracy for mixed questions with correction.")
    print("-------------------------------------------------------")


if __name__ == "__main__":
    main()
