import sys
sys.path.append('models/')

import json
import os
from models.chexagent import CheXagent
from colours import bcolors

base_image_path = "dataset/Slake/Slake1.0/imgs/"
input_file = "dataset/Slake/Slake1.0/validate.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

model = CheXagent()

i = 0

for entry in data:
    if i == 5:
        break
    original_prompt = entry["question"]
    correct_answer = entry["answer"]  # Ground truth answer
    image_path = os.path.join(base_image_path, entry["img_name"])

    model_answer = model.generate(image_path, original_prompt).strip()
    print(f"{bcolors.OKGREEN} Question:{bcolors.ENDC} {original_prompt} ")
    print(f"{bcolors.OKBLUE} Answer: {bcolors.ENDC} {model_answer}\n")
    i += 1

