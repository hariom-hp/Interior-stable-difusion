from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from utils_func import create_scheduler

# Device detection: MPS for Mac, CUDA for NVIDIA, CPU fallback
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def load_model_base(model_path: str):
    pipe = StableDiffusionPipeline.from_single_file(model_path).to(device)

    # Enable memory optimizations for MPS
    if device == "mps":
        pipe.enable_attention_slicing("max")

    # pipe.load_lora_weights("TrgTuan10/Interior", weight_name="xsarchitectural-7.safetensors", adapter_name="architecture")
    # trigger_words = " ,VERRIERES, DAYLIGHTINDIRECT, LIGHTINGAD, MAGAZINE8K, CINEMATIC LIGHTING, EPIC COMPOSITION"

    return pipe

def gen_base(pipe, 
                 prompt, 
                 trigger_words="", 
                 neg="", 
                 seed=42, 
                 num_images=1, 
                 num_infer_steps=35, 
                 height=768, 
                 width=1024):
    
    prompt = prompt + " high quality, lightning, luxury" + trigger_words
    neg = neg + " soft line,curved line,sketch,ugly,logo,pixelated,lowres,text,word,cropped,low quality,normal quality,username,watermark,signature,blurry,soft"
    
    scheduler = create_scheduler()
    
    generator = torch.Generator(device=device).manual_seed(seed)
    images = pipe(
        prompt=prompt,
        negative_prompt=neg,
        do_classifier_free_guidance=True,
        num_images_per_prompt=num_images,
        num_inference_steps=num_infer_steps,
        height=height,
        width=width,
        guidance_scale=7,
        generator=generator,
        scheduler=scheduler
    )
    return images.images

if __name__ == "__main__":
    pipe = load_model_base("checkpoints/Interior.safetensors")
    pipe.load_lora_weights("checkpoints", weight_name="Interior_lora.safetensors")
    pipe.fuse_lora(lora_scale=0.7)
    
    print(pipe.get_active)
    
    prompts = "A modern living room"
    negative_prompt = ""
    images = gen_base(pipe, prompts, neg=negative_prompt, trigger_words="", num_images=1)
    image = images[0]
    image.save("interior.jpg")
    print("Image saved as interior.jpg")