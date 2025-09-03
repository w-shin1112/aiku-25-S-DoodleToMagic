import gradio as gr
from PIL import Image
import torch
import os
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DPMSolverMultistepScheduler
from peft import PeftModel

# ------------------------------------------------------
# 모델 경로 및 설정
# ------------------------------------------------------
POKEMON_LORA_PATH = "/home/aikusrv04/Doodle/FINAL/model/pokemon"
AMATEUR_LORA_PATH = "/home/aikusrv04/Doodle/FINAL/model/amateur"

guidance_scale = 7.5
controlnet_scale = 1.0
steps = 20
POKEMON_SEED = 456
AMATEUR_SEED = 42

# ------------------------------------------------------
# 디바이스 설정
# ------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------
# ControlNet + Stable Diffusion 파이프라인 초기화
# ------------------------------------------------------
def load_pipeline(lora_path=None):
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

    # LoRA 적용
    if lora_path and os.path.exists(lora_path):
        print(f"🔄 Loading LoRA weights from {lora_path}")
        pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
    else:
        print(f"⚠️ LoRA weights not found at {lora_path}, using base model")

    pipe = pipe.to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe

# ------------------------------------------------------
# 모델 캐싱
# ------------------------------------------------------
pokemon_pipe = load_pipeline(POKEMON_LORA_PATH)
amateur_pipe = load_pipeline(AMATEUR_LORA_PATH)

# ------------------------------------------------------
# Inference 함수 (실시간 진행률 표시)
# ------------------------------------------------------
def run_inference(category, mode, sketch, uploaded, description):
    try:
        # 입력 이미지 확인
        image = None
        if mode == "스케치 사용":
            if sketch is not None:
                if isinstance(sketch, dict):
                    if "composite" in sketch and sketch["composite"] is not None:
                        image = sketch["composite"]
                    elif "layers" in sketch and len(sketch["layers"]) > 0:
                        image = sketch["layers"][0]
                else:
                    image = sketch
        elif mode == "업로드 사용" and uploaded is not None:
            image = uploaded

        if image is None:
            yield "⚠️ 입력 이미지를 선택해주세요.", None
            return

        if description is None or description.strip() == "":
            yield "⚠️ 이미지 설명을 입력해주세요.", None
            return

        # 입력 이미지 전처리
        image = image.convert("RGB").resize((512, 512))

        # 카테고리별 설정
        if category == "포켓몬":
            user_prompt = f"pokemon style, cute {description.strip()} pokemon character, no background"
            pipe = pokemon_pipe
            seed = POKEMON_SEED
        else:
            user_prompt = f"a childlike crayon drawing, cute {description.strip()} character, no background"
            pipe = amateur_pipe
            seed = AMATEUR_SEED

        # 진행률 표시 (steps 기준)
        for i in range(1, steps + 1):
            progress = int((i / steps) * 100)
            yield f"⏳ 이미지 생성 중... ", None

        # 이미지 생성
        result = pipe(
            prompt=user_prompt,
            negative_prompt="background, blurry, low quality, distorted, ugly, bad anatomy, extra limbs",
            image=image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_scale,
            generator=torch.manual_seed(seed)
        )

        generated_image = result.images[0]
        yield "✅ 이미지 생성 완료!", generated_image

    except Exception as e:
        yield f"❌ 오류 발생: {e}", None

# ------------------------------------------------------
# Gradio UI 구성
# ------------------------------------------------------
CUSTOM_CSS = """
#centered-row {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 50px;
}

#big_sketch {
    width: 700px !important;
    height: 700px !important;
    max-width: 1000px !important;
    max-height: 1000px !important;
}

#small_upload {
    width: 500px !important;
    height: 500px !important;
    max-width: 500px !important;
    max-height: 500px !important;
}
"""

with gr.Blocks(css=CUSTOM_CSS) as demo:
    gr.Markdown("## 🎨 AIKU - Pokemon & Crayon Doodle Diffusion")

    category = gr.Radio(
        choices=["포켓몬", "아마추어"],
        value="포켓몬",
        label="생성 카테고리 선택"
    )

    mode = gr.Radio(
        choices=["스케치 사용", "업로드 사용"],
        value="업로드 사용",
        label="입력 방식 선택"
    )

    with gr.Row(elem_id="centered-row"):
        sketch = gr.Sketchpad(
            label="🖌️ 흑백 스케치",
            canvas_size=(700, 700),
            type="pil",
            visible=False,
            elem_id="big_sketch"
        )
        uploaded = gr.Image(
            label="이미지 업로드",
            type="pil",
            visible=True,
            elem_id="small_upload"
        )

    description = gr.Textbox(
        label="이미지 설명 입력 (단어로 입력)",
        placeholder="예: tiger"
    )

    submit = gr.Button("제출")
    result = gr.Textbox(label="진행 상태", interactive=True)
    result_image = gr.Image(label="생성된 이미지", type="pil")

    def toggle_inputs(mode):
        if mode == "스케치 사용":
            return gr.update(visible=True, value=None), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)

    mode.change(
        fn=toggle_inputs,
        inputs=[mode],
        outputs=[sketch, uploaded]
    )

    submit.click(
        fn=run_inference,
        inputs=[category, mode, sketch, uploaded, description],
        outputs=[result, result_image]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
