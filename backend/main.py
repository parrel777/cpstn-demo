from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch
import uuid
import os
from dotenv import load_dotenv

load_dotenv()
# hf_token = os.getenv("HF_TOKEN") 새로 모델 로드시 토큰 인증

app = FastAPI()

# CORS 설정 (프론트엔드와 통신 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포 시 프론트엔드 도메인으로 변경
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = 'parrel777/hanbok-LoRA-ver2' # LoRA
base_model = 'Bingsu/my-korean-stable-diffusion-v1-5'

pipe = DiffusionPipeline.from_pretrained(
    base_model, 
    torch_dtype=torch.float16
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.scheduler.use_karras_sigmas = True
pipe.load_lora_weights(model_path)
pipe.to("cuda")

# 입력 데이터 모델 정의
class GenerateRequest(BaseModel):
    prompt: str
    steps: int = 25
    guidance_scale: float = 7.5

@app.post("/generate")
async def generate_image(request: GenerateRequest):
    prompt = request.prompt
    steps = request.steps
    guidance_scale = request.guidance_scale

    if not prompt:
        return {"error": "프롬프트를 입력하세요."}

    image = pipe(
        prompt, 
        num_inference_steps=steps, 
        guidance_scale=guidance_scale,
        height=512,
        width=512,
    ).images[0]

    os.makedirs("generated_images", exist_ok=True)
    filename = f"{uuid.uuid4()}.jpg"
    image_path = os.path.join("generated_images", filename)
    image.save(image_path)

    return FileResponse(image_path, media_type="image/jpg")