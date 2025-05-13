import re
import os
import sys
import torch
import uvicorn
import tempfile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Form

from model.CheXAgentWrapper import CheXAgentWrapper
from model.preprocessing_model import T5WithInversionHead


sys.path.append('models/')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def post_processing(prompt: str) -> str:
    if prompt.lower() == "no":
        return "Yes"
    elif prompt.lower() == "yes":
        return "No"
    else:
        leading_ws = re.match(r'^\s*', prompt).group(0)
        body = prompt[len(leading_ws):]

        m = re.match(r'([A-Za-z]+)(.*)', body, re.DOTALL)

        if m:
            first, rest = m.group(1), m.group(2)

            if not first.isupper():
                first = first.lower()

            new_body = f"Not {first}{rest}"
        else:
            new_body = f"Not {body}"

        return leading_ws + new_body

@app.on_event("startup")
def load_model():
    preprocessing_model = T5WithInversionHead.from_pretrained('/home/mm637oq/PycharmProjects/PPV_zadanie/utils/vital-meadow-45-postproc-checkpoints/checkpoint-12915')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    app.state.model = CheXAgentWrapper(lambda x: preprocessing_model.canonicalize_and_classify_from_text(x, device=device), post_processing, device=device)

@app.post("/process")
async def process_with_tempdir(
    image: UploadFile = File(None),
    message: str = Form("")
):
    img_bytes = await image.read() if image else None

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "upload.jpg")
        if img_bytes:
            with open(path, "wb") as f:
                f.write(img_bytes)

        model = app.state.model
        result = model(message, path)

    return JSONResponse({"result": result})

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=3000, reload=True)