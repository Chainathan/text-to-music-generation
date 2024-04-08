import os
import torch
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDPMScheduler

def generate_images(prompt, model_id, num_images=1, guidance_scale=7.5, inference_steps=50, noise_scheduler=None):
    """
    Generate images from a text prompt using a fine-tuned Stable Diffusion model.

    Parameters:
    - prompt (str): The text prompt to generate images from.
    - model_id (str): The model ID on Hugging Face's Model Hub or the path to the local model.
    - num_images (int): The number of images to generate.
    - guidance_scale (float): The guidance scale to control the adherence to the prompt.

    Returns:
    - images (List[PIL.Image]): The list of generated images.
    """
    accelerator = Accelerator()
    pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(accelerator.device)
    if noise_scheduler is not None:
        pipeline.scheduler = noise_scheduler

    # Load fine-tuned unet
    if model_id.find("riffusion") == -1:
        pipeline.load_lora_weights("temp/musiccaps/checkpoint-3000/pytorch_lora_weights.safetensors")

    images = []
    # with autocast(accelerator.device):
    for _ in range(num_images):
        # Generate an image from the prompt
        generated = pipeline(prompt, guidance_scale=guidance_scale,
                             width=256, height=256, num_inference_steps=inference_steps).images[0]
        images.append(generated)

    return images

if __name__ == '__main__':
    target_dir = "temp/Results/"
    prompt = "joyous, happy, michael jackson, high-tempo, beat, fart"  # Example prompt
    noise_scheduler = None

    # model_id = "runwayml/stable-diffusion-v1-5"  # Replace with your model's ID or local path

    # Riffusion model
    model_id = "riffusion/riffusion-model-v1"
    if model_id.find("riffusion") != -1:
        target_dir = os.path.join(target_dir, "riffusion")

    # Get DDIM scheduler
    # noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    inference_steps = 100

    generated_images = generate_images(prompt, model_id=model_id, num_images=1, guidance_scale=7.5,
           inference_steps = inference_steps, noise_scheduler=noise_scheduler)
    # Display or save the generated images
    for i, img in enumerate(generated_images):
        i_path = os.path.join(target_dir, f"generated_image_{i}.png")
        img.save(i_path)