import os
import torch
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDPMScheduler
import tinytorchutil

def generate_images(prompt, model_id, num_images=1, guidance_scale=7.5, inference_steps=50,
    noise_scheduler=None, seed=0, width=256, height=256):
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
    # Fix seed
    tinytorchutil.set_seed(seed)

    accelerator = Accelerator()
    pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(accelerator.device)
    if noise_scheduler is not None:
        pipeline.scheduler = noise_scheduler

    images = []
    for _ in range(num_images):
        # Generate an image from the prompt
        generated = pipeline(prompt, guidance_scale=guidance_scale,
                             width=width, height=height, num_inference_steps=inference_steps).images[0]
        images.append(generated)

    return images

if __name__ == '__main__':
    target_dir = "temp/Results/"
    prompt = "random cat image"  # Example prompt
    noise_scheduler = None
    width = 256
    height = 256

    model_id = "runwayml/stable-diffusion-v1-5"  # Replace with your model's ID or local path

    # Get DDIM scheduler
    # noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    inference_steps = 50
    seed=42

    generated_images = generate_images(prompt, model_id=model_id, num_images=1, guidance_scale=7.5,
           inference_steps = inference_steps, noise_scheduler=noise_scheduler, seed=seed,
           width=width, height=height)
    # Display or save the generated images
    for i, img in enumerate(generated_images):
        i_path = os.path.join(target_dir, f"generated_image_{i}.png")
        img.save(i_path)