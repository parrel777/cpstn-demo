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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

hanbok = 'parrel777/hanbok-LoRA-ver2' 
hanok = 'parrel777/hanok-LoRA-ver1' 
base_model = 'Bingsu/my-korean-stable-diffusion-v1-5'

pipe = DiffusionPipeline.from_pretrained(
    base_model, 
    torch_dtype=torch.float16
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.scheduler.use_karras_sigmas = True
pipe.unet.load_attn_procs(hanbok)
pipe.unet.load_attn_procs(hanok)
pipe.to("cuda")

quality_prompt ="""최고 화질, 초고해상도, 정밀한 묘사, 영화 같은 조명, 섬세한 디테일, 명작, 
선명한 얼굴, 자세한 얼굴, 자연스러운 피부 질감, 또렷한 눈동자, 균형 잡힌 얼굴, 아름다운 눈, 고해상도 피부, 자연스러운 눈
"""

# 부정 프롬프트
negative_prompt = """못생긴, 흐릿한, 저화질, 비현실적인, 얼굴 왜곡, 이상한 눈, 부자연스러운 피부, 
비정상적인 손가락, 기괴한 자세, 왜곡된 신체, 불명확한 배경, 다중 얼굴, 
과도한 장신구, 글씨 존재, 잘린 얼굴, 분할된, 텍스트, 글자, 글자가 있는, 텍스트 포함, 이상한 글자, 텍스트 왜곡"""

class GenerateRequest(BaseModel):
    prompt: str
    steps: int = 25
    guidance_scale: float = 7.5

@app.post("/generate")
async def generate_image(request: GenerateRequest):
    final_prompt = f"{request.prompt}, {quality_prompt}"
    prompt = final_prompt
    steps = request.steps
    guidance_scale = request.guidance_scale

    if not prompt:
        return {"error": "프롬프트를 입력하세요."}

    image = pipe(
        prompt, 
        num_inference_steps=steps, 
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        height=512,
        width=512,
    ).images[0]

    os.makedirs("generated_images", exist_ok=True)
    filename = f"{uuid.uuid4()}.jpg"
    image_path = os.path.join("generated_images", filename)
    image.save(image_path)

    return FileResponse(image_path, media_type="image/jpg")