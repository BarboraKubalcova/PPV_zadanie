import sys
sys.path.append('../models')
sys.path.append('..')

import requests, os, json
import ast
from colours import bcolors

from chexagent import CheXagent


base_image_path = "../dataset/Slake/Slake1.0/imgs/"
mixed_file = "data/test_data_original_and_inverted.json"
inverted_data = "data/inverted_test_data_filtered.json"

vq_model = CheXagent()
ollama_model = "gemma3:4b"


def chat_with_model(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": ollama_model,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]


def read_vqa_json(json_input, from_file=True):
    if from_file:
        with open(json_input, 'r', encoding='utf-8') as file:
            data = json.load(file)
    else:
        data = json.loads(json_input)
    return data


def parse_model_output(output_str):
    try:
        parsed = ast.literal_eval(output_str.strip())
        if isinstance(parsed, list) and len(parsed) == 2:
            question, flag = parsed
            if isinstance(question, str) and isinstance(flag, int):
                return [question, flag]
        print("Output format is invalid.")
        return None
    except (ValueError, SyntaxError):
        print(f"Failed to parse model output. {output_str}")
        return None


def canonicalize_and_classify_with_prompting(original_prompt):
    prompt = f"""
        Task: Convert only **negated** diagnostic questions into their positive equivalent. If the question is already positive, do not change it.
        Negation includes: "no", "not", "without", "unhealthy", "abnormal", "non-", "ill", etc.
        
        **Rules:**
        - Flip only if negation is explicitly present.
        - Leave already-positive sentences untouched (e.g. "Is the scan normal?").
        - Return format: ['<new sentence>', <1 if changed, else 0>]

        Examples:
        
        Input: Is the lung unhealthy?  
        Output: ['Is the lung healthy?', 1]

        Input: Does this image look normal?  
        Output: ['Does this image look normal?', 0]
        
        Now transform the following input accordingly. Answer with the predefined format only.
        Input: {original_prompt}
    """

    reply = chat_with_model(prompt)
    processed_reply = parse_model_output(reply)
    if not processed_reply:
        return None
    return processed_reply


def flip_answer(ans):
    ans = ans.lower()
    if ans == "no":
        return "Yes"
    elif ans == "yes":
        return "No"
    else:
        raise Exception("Unsupported question type - yes/no questions only!")


def test_xeagent(data, output_file_name):
    results = []
    total_questions = 0
    correct_predictions = 0

    for i, entry in enumerate(data):
        if i % 5 == 0:
            print(f"Question {i}/{len(data)}")

        reply = canonicalize_and_classify_with_prompting(entry["question"])
        if not reply:
            continue

        question = reply[0]
        inverted = reply[1]

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
        print(f"{bcolors.OKGREEN} Question:{bcolors.ENDC} {entry["question"]} ")
        print(f"{bcolors.OKBLUE} Answer: {bcolors.ENDC} {model_answer}")
        print(f"{bcolors.OKBLUE} Correct Answer: {bcolors.ENDC} {entry["answer"]}\n")

    accuracy = (correct_predictions / total_questions) * 100 if total_questions > 0 else 0
    print(f"\nTotal Questions: {total_questions}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%\n")

    output_data = {
        "accuracy": accuracy,
        "answered_questions": total_questions,
        "total_questions": len(data),
        "correct_predictions": correct_predictions,
        "results": results
    }

    with open(f"test_output/{output_file_name}", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    return output_data


def main():
    # original_question = "Does this image look normal?"
    # raw_answer = canonicalize_and_classify_with_prompting(original_question)
    # print(raw_answer)

    print("\nTesting Xagent on inverted testing data with prompt correction")
    data = read_vqa_json(inverted_data)
    output_with_correction = test_xeagent(data, "inverted_test_results_with_correction_prompting.json")

    print("\nTesting Xagent on mixed testing data with prompt correction")
    data = read_vqa_json(mixed_file)
    output_with_correction_mixed = test_xeagent(data, "mixed_test_results_with_correction_prompting.json")

    print("-------------------------------------------------------")
    print(f"{output_with_correction["accuracy"]}: Accuracy for inverted questions with prompt correction.")
    print(f"{output_with_correction_mixed["accuracy"]}: Accuracy for mixed questions with prompt correction.")
    print("-------------------------------------------------------")


if __name__ == "__main__":
    main()
