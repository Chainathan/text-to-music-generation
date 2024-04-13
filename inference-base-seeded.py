import os
import torch
from PIL import Image
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDPMScheduler
import tinytorchutil
from riffusion.riffusion_pipeline import preprocess_image
from torchvision.transforms import transforms

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
dtype = torch.float16

def generate_images(prompt, model_id, num_images=1, guidance_scale=7.5, inference_steps=50,
    noise_scheduler=None, seed=0, init_image="og_beat", width=256, height=256):
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

    # Load fine-tuned unet
    if model_id.find("riffusion") == -1:
        pipeline.load_lora_weights("temp/musiccaps/checkpoint-4500/pytorch_lora_weights.safetensors")

    # Setup init_image
    if model_id.find("riffusion") == -1:
        init_image_path = os.path.join("seed_images", f"{init_image}.png")
    else:
        init_image_path = os.path.join("seed_images", "riffusion", f"{init_image}.png")

    # Get seed latent
    init_image = Image.open(init_image_path).convert("RGB").resize((width, height))
    if model_id.find("riffusion") == -1:    # Our preprocessing
        init_image_tensor = transforms.ToTensor()(init_image).to(device=device, dtype=dtype)  # Convert to tensor
        init_image_tensor = transforms.Normalize(mean=[0.5], std=[0.5])(init_image_tensor).unsqueeze(0)   # Our normalization
        # init_image_tensor = preprocess_image(init_image).to(device=device, dtype=dtype)
    else:
        init_image_tensor = preprocess_image(init_image).to(device=device, dtype=dtype)
    latents = pipeline.vae.encode(init_image_tensor).latent_dist.sample()

    images = []
    for _ in range(num_images):
        # Generate an image from the prompt
        generated = pipeline(prompt, guidance_scale=guidance_scale,
                             width=width, height=height, num_inference_steps=inference_steps, latents=latents).images[0]
        images.append(generated)

    return images

if __name__ == '__main__':
    target_dir = "temp/Results/"
    prompt = "piano funk"  # Example prompt
    init_image = "agile"
    noise_scheduler = None
    width=256
    height=256

    model_id = "runwayml/stable-diffusion-v1-5"  # Replace with your model's ID or local path

    # # Riffusion model
    # model_id = "riffusion/riffusion-model-v1"
    # if model_id.find("riffusion") != -1:
    #     target_dir = os.path.join(target_dir, "riffusion")

    # Get DDIM scheduler
    # noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    inference_steps = 50
    seed=42

    generated_images = generate_images(prompt, model_id=model_id, num_images=1, guidance_scale=7.5,
           inference_steps = inference_steps, noise_scheduler=noise_scheduler, seed=seed, init_image=init_image,
           width=width, height=height)
    # Display or save the generated images
    for i, img in enumerate(generated_images):
        i_path = os.path.join(target_dir, f"generated_image_{i}.png")
        img.save(i_path)