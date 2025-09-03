# modal_app.py
import base64, io, re
from typing import Literal

import modal
from PIL import Image

app = modal.App("doodle-to-magic")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "diffusers", "transformers", "peft", "accelerate",
        "fastapi[standard]", "pillow"
    )
    .add_local_python_source("inference_2d")
)

MODELS_DIR = "/vol/models"
weights_vol = modal.Volume.from_name("sd-lora-weights", create_if_missing=True)

@app.cls(image=image, gpu="T4", volumes={MODELS_DIR: weights_vol}, timeout=3600)
class SDInference:
    _pokemon = None
    _amateur = None

    @modal.method()
    def initialize(self):
        from inference_2d.pokemon_inference import load_pokemon_pipeline
        from inference_2d.amateur_inference import load_amateur_pipeline
        self._pokemon = load_pokemon_pipeline()
        self._amateur = load_amateur_pipeline()

    @modal.method()
    def infer(self, mode: Literal["pokemon", "amateur"], prompt: str, data_url_or_b64: str) -> str:
        m = re.match(r"^data:image/\w+;base64,(.*)$", data_url_or_b64)
        b64 = m.group(1) if m else data_url_or_b64

        img_bytes = base64.b64decode(b64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        if mode == "pokemon":
            from inference_2d.pokemon_inference import run_pokemon_inference
            out_img = run_pokemon_inference(self._pokemon, image, prompt)
        else:
            from inference_2d.amateur_inference import run_amateur_inference
            out_img = run_amateur_inference(self._amateur, image, prompt)

        buf = io.BytesIO()
        out_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def generate(body: dict):
    """
    요청:
    { "image": "<dataURL or base64>", "prompt": "...", "theme": "pokemon|amateur" }

    응답:
    { "resultImage": "<base64 png>" }
    """
    theme = body.get("theme", "pokemon")
    prompt = body.get("prompt", "")
    img    = body.get("image", "")

    inferencer = SDInference()
    inferencer.initialize.remote()
    result_b64 = inferencer.infer.remote(
        "pokemon" if theme == "pokemon" else "amateur",
        prompt,
        img,
    )
    return {"resultImage": f"data:image/png;base64,{result_b64}"}

@app.local_entrypoint()
def main():
    """
    테스트용: modal run modal_app.py::main
    """
    import os
    sample_path = "sample_scribble.png"
    if not os.path.exists(sample_path):
        print("[INFO] Put a small PNG as 'sample_scribble.png' to test.")
        return

    with open(sample_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    inferencer = SDInference()
    inferencer.initialize.remote()
    out = inferencer.infer.remote("pokemon", "monster", f"data:image/png;base64,{b64}")
    print("result b64 length:", len(out))