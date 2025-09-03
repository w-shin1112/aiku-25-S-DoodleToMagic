import logging
import sys
import os
import time
import numpy as np
import rembg
import torch
from PIL import Image
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground

# config
if len(sys.argv) > 1:
    NUM = sys.argv[1]
else:
    NUM = input("Enter the input number: ")
INPUT_IMAGE = f"output_{NUM}.png"
INPUT_PATH = f"/home/aikusrv04/Doodle/FINAL/outputs_2d/pokemon/{INPUT_IMAGE}"
MODEL= "stabilityai/TripoSR"
OUTPUT_PATH = "/home/aikusrv04/Doodle/FINAL/outputs_3d/pokemon"

chunk_size = 8192
mc_resolution = 400
no_remove_bg = False
foreground_ratio = 0.85

if torch.cuda.is_available():
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
else:
    device = torch.device("cpu")

class Timer:
    def __init__(self):
        self.items = {}
        self.time_scale = 1000.0
        self.time_unit = "ms"

    def start(self, name: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.items[name] = time.time()
        logging.info(f"{name} ...")

    def end(self, name: str) -> float:
        if name not in self.items:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = self.items.pop(name)
        delta = time.time() - start_time
        t = delta * self.time_scale
        logging.info(f"{name} finished in {t:.2f}{self.time_unit}.")

timer = Timer()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

timer.start("🏀 Initializing model")
model = TSR.from_pretrained(
    MODEL,
    config_name="config.yaml",
    weight_name="model.ckpt",
)
model.renderer.set_chunk_size(chunk_size)
model.to(device)
timer.end("✅ Initializing model")

timer.start("📕 Processing images")

if no_remove_bg:
    rembg_session = None
else:
    rembg_session = rembg.new_session()

file_name = os.path.basename(INPUT_PATH)
base_name = os.path.splitext(file_name)[0]
images = []

if no_remove_bg:
    rembg_session = None
else:
    rembg_session = rembg.new_session()

if no_remove_bg:
    image = np.array(Image.open(INPUT_PATH).convert("RGB"))
else:
    image = remove_background(Image.open(INPUT_PATH), rembg_session)
    image = resize_foreground(image, foreground_ratio)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
    image = Image.fromarray((image * 255.0).astype(np.uint8))
    
timer.end("Processing images")

# 모델 실행
logging.info(f"💫 Running model on {file_name} ...")

timer.start("Running model")
with torch.no_grad():
    scene_codes = model([image], device=device)
timer.end("Running model")

timer.start("Extracting mesh")
meshes = model.extract_mesh(scene_codes, True, resolution=mc_resolution)
timer.end("Extracting mesh")

# OBJ 저장
out_obj_path = os.path.join(OUTPUT_PATH, f"{base_name}.obj")

timer.start("Exporting mesh")
meshes[0].export(out_obj_path)
timer.end("Exporting mesh")

logging.info(f"🐳 Export finished: {out_obj_path}")