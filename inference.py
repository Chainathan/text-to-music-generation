import torch
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from torch import autocast
from safetensors.torch import load_model

def generate_images(prompt, model_id="your_model_id_here", num_images=1, guidance_scale=7.5):
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

    # Load fine-tuned unet
    pipeline.load_lora_weights("sd-k80/")

    images = []
    # with autocast(accelerator.device):
    for _ in range(num_images):
        # Generate an image from the prompt
        generated = pipeline(prompt, guidance_scale=guidance_scale).images[0]
        images.append(generated)

    return images
if __name__ == '__main__':
    prompt = "Hard Rock"  # Example prompt
    model_id = "runwayml/stable-diffusion-v1-5"  # Replace with your model's ID or local path
    generated_images = generate_images(prompt, model_id=model_id, num_images=3, guidance_scale=7.5)

    # Display or save the generated images
    for i, img in enumerate(generated_images):
        img.save(f"Results/generated_image_{i}.png")