# --- inference_2d/pokemon_inference.py ---

import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DPMSolverMultistepScheduler
from peft import PeftModel
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LORA_WEIGHTS_PATH = os.getenv("POKEMON_LORA_PATH", "/vol/models/model/pokemon")
# LORA_WEIGHTS_PATH = "/home/aikusrv04/Doodle/FINAL/model/pokemon"
SEED = 456
PROMPT_PREFIX = "pokemon style, realistic"
NEGATIVE_PROMPT = "background, blurry, low quality, distorted, ugly, bad anatomy, extra limbs"


def load_pokemon_pipeline():
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_scribble",
        torch_dtype=torch.float16
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    )

    if os.path.exists(LORA_WEIGHTS_PATH):
        pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_WEIGHTS_PATH)
    else:
        print("[WARN] Pokemon LoRA not found. Using base model.")

    pipe = pipe.to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe


def run_pokemon_inference(pipe, image: Image.Image, user_prompt: str) -> Image.Image:
    prompt = f"{PROMPT_PREFIX}, cute {user_prompt} pokemon character, no background"
    image = image.convert("RGB").resize((512, 512))

    result = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        image=image,
        num_inference_steps=20,
        guidance_scale=7.5,
        controlnet_conditioning_scale=1.0,
        generator=torch.manual_seed(SEED)
    )

    return result.images[0]