from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DPMSolverMultistepScheduler
import torch
from utils_func import create_scheduler
import cv2
from PIL import Image
import numpy as np

# Device detection: MPS for Mac, CUDA for NVIDIA, CPU fallback
if torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.float32  # MPS works better with float32
elif torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.float16
else:
    DEVICE = "cpu"
    DTYPE = torch.float32

def load_controlnet_model(model_path: str):
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", torch_dtype=DTYPE
    )
    
    pipe_controlnet = StableDiffusionControlNetPipeline.from_single_file(
        model_path,
        controlnet=controlnet, 
        safety_checker=None,
        torch_dtype=DTYPE
    ).to(DEVICE)
    
    # Enable memory optimizations for MPS
    if DEVICE == "mps":
        pipe_controlnet.enable_attention_slicing("max")
    
    return pipe_controlnet

def preprocessor_image(image):
    image = np.array(image)

    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    return image
    

def gen_controlnet(pipe_controlnet,
                            prompt,
                            trigger_words="",
                            neg="",
                            seed=42,
                            num_inference_steps=35,
                            height=512, 
                            width=512,
                            num_images=1,
                            image=None):
    prompt = prompt + "photography, minimalism, cinematic, canon EOS 5d, natural sunlight, studio lights –ar 9:16 –s 140 --stylize 750" + trigger_words
    neg = neg + " ignature, soft, blurry, drawing, sketch, poor quality, ugly, text, type, word, logo, pixelated, low resolution, saturated, high contrast, oversharpened"
    
    scheduler = create_scheduler()
    image_control = preprocessor_image(image)
    # image_control.save("control.jpg")
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    
    images = pipe_controlnet(
        prompt=prompt, 
        negative_prompt=neg,
        num_inference_steps=num_inference_steps,
        generator=generator,
        num_images_per_prompt=num_images,
        image = image_control, 
        height=height, 
        width=width,
        scheduler=scheduler,
    ).images
    
    return images


if __name__ == "__main__":
    pipe_controlnet = load_controlnet_model("checkpoints/Interior.pt",)
    prompt = "a living room with a TV, wooden floor, a sofa, a nice glass table and a flower in the table"
    negative_prompt = "(multiple outlets:1.9),carpets,(multiple tv screens:1.9),2 tables,lamps,lightbuble,(plants:1.6)bad-hands-5, ng_deepnegative_v1_75t, EasyNegative, bad_prompt_version2, bad-artist-anime, bad-artist, bad-image-v2-39000, verybadimagenegative_v1.3, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
    image = Image.open("/home/ubuntu/old_code/trgtuan/Interior-stable-difusion/interior.jpg")
    images = gen_controlnet(pipe_controlnet, prompt, trigger_words="", neg=negative_prompt, num_images=1, image=image)
    image = images[0]
    image.save("interior_control.jpg")
    print("Image saved as interior_control.jpg")
    
                            
                            
