import torch
from typing import Callable

from models.chexagent import CheXagent


class CheXAgentWrapper:
    def __init__(self, preprocessing: Callable[[str], tuple[str, bool]], postprocessing: Callable[[str], str], device: torch.device = torch.device("cpu")):
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing

        self.chexagent  = CheXagent(device=device)

    def __call__(self, prompt, img_path):
        canonized, is_inverted = self.preprocessing(prompt)

        output = self.chexagent.generate(img_path, canonized, do_sample=False)

        if is_inverted:
            output = self.postprocessing(output)

        return output