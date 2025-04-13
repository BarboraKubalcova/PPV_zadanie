import requests
import json

ollama_model = "mistral"


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


def variate_questions(original_prompt):
    prompt = f"""
        You are a medical language assistant working with chest X-ray related questions. Given a question, produce 3 
        different outputs for each of the following tasks:
    
        Negation – Transform the question into a 3 logically negated version.
        Paraphrasing – Rewrite the question in simpler terms, but ensure that each version maintains the **same logical 
        meaning and polarity**, so that a 'Yes' or 'No' answer to the paraphrased question would be 
        **exactly the same** as the original.
        Synonym Substitution – Replace key words with appropriate synonyms to form 3 new versions of the question,
        ensuring the new questions retain the original intent and logical polarity.
        
        **Important guidelines**:
        - Keep the form as a question (not an answer).
        - Use simple, clear, non-fancy language.
        - All questions must be CLOSED-ended – they should be answerable with 'yes' or 'no'.
        - Avoid rewordings that invert or flip the expected answer logic.
        
        Please keep the form as the question instead of an answer and try to introduce same variance from the original in 
        those ways. Also, provide them in following format and do not reply with anything else: 
        {{
            negated_question: ["[list of your 3 negated questions]"],
            paraphrased_questions: ["[list of your 3 paraphrased questions]"],
            synonymous_questions: ["[list of your 3 synonymous questions]"],
        }}
        
        Here is the question: {original_prompt}.
    """

    reply = chat_with_model(prompt)
    return reply


def correct_json_format(corrupted_json):
    prompt = ("Correct brackets, parentheses and commas in the given json. Answer only with the corrected json, "
              f"nothing else. Here is the json: {corrupted_json}"
              )
    corrected_json = chat_with_model(prompt)
    return corrected_json


def filter_data(data, lang="zh"):
    filtered_json = [entry for entry in data if entry.get("q_lang") != lang and entry.get("answer_type") == "CLOSED"]
    return filtered_json


def read_vqa_json(json_input, from_file=True):
    if from_file:
        with open(json_input, 'r', encoding='utf-8') as file:
            data = json.load(file)
    else:
        data = json.loads(json_input)
    data = filter_data(data)
    return data


def parse_variations(variations_str):
    try:
        return json.loads(variations_str)
    except json.JSONDecodeError:
        try:
            corrected = correct_json_format(variations_str)
            return json.loads(corrected)
        except json.JSONDecodeError:
            return None


def create_extended_json(data):
    new_dataset = []
    entry_id = 0
    for entry in data:
        original_question = entry.get("question")
        raw_variations = variate_questions(original_question)
        parsed_variations = parse_variations(raw_variations)

        if not parsed_variations:
            print(f"Unable to parse variations for question: {original_question}")
            continue

        for key, val in parsed_variations.items():
            is_negated = (key == "negated_question")
            for variation in val:
                new_data_element = {
                    "question_id": entry_id,
                    "original": original_question,
                    "variation": variation,
                    "is_negated": is_negated
                }
                new_dataset.append(new_data_element)
        entry_id += 1
        print(f"Question {entry_id}/{len(data)}")
        if entry_id >= 10:
            break

    return new_dataset


def save_json_to_file(data, filename):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Data successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving file: {e}")


def main():
    dataset_val_path = "../dataset/Slake/Slake1.0/validate.json"
    output_file = "./variated_dataset.json"

    data = read_vqa_json(dataset_val_path)
    new_dataset = create_extended_json(data)
    print(new_dataset)
    save_json_to_file(new_dataset, output_file)


if __name__ == "__main__":
    main()

