from __future__ import annotations
import typing as T
import os
import argparse
import torch
import numpy as np
from PIL import Image
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, DDIMScheduler
from pydub import AudioSegment
from riffusion.datatypes import InferenceInput, PromptInput
from riffusion.riffusion_pipeline import RiffusionPipeline
from utils.ImageUtils import save_images
from utils.AudioUtilsRiffusion import reconstruct_audio_from_spectrograms, stitch_segments

def infer_prompts(
        pipeline: RiffusionPipeline,
        prompt_input_a_text: str,
        prompt_input_b_text: str,
        target_dir: str = "Results/",
        num_interpolation_steps: int = 4,
        num_inference_steps: int = 50,
        guidance: float = 7.0,
        init_image_name: str = "og_beat",
        alpha_power: float = 1.0,
        prompt_input_a_seed: int = 42,
        prompt_input_a_denoising: float = 0.5,
        prompt_input_b_seed: int = 42,
        prompt_input_b_denoising: float = 0.5,
        mask_image: Image = None,
        save_audio: bool = True,
        save_image: bool = True,
) -> tuple[list[Image], list[AudioSegment], AudioSegment] | None:
    """
    Function to run inference on prompts and return list of spectrogram images and audio segments.
    """

    alphas = np.linspace(0, 1, num_interpolation_steps)

    # Apply power scaling to alphas to customize the interpolation curve
    alphas_shifted = alphas * 2 - 1
    alphas_shifted = (np.abs(alphas_shifted) ** alpha_power * np.sign(alphas_shifted) + 1) / 2
    alphas = alphas_shifted

    # Define PromptInput for each prompt
    prompt_input_a = PromptInput(
        prompt=prompt_input_a_text,
        seed=prompt_input_a_seed,
        denoising=prompt_input_a_denoising,
        guidance=guidance,
    )

    prompt_input_b = PromptInput(
        prompt=prompt_input_b_text,
        seed=prompt_input_b_seed,
        denoising=prompt_input_b_denoising,
        guidance=guidance,
    )

    if not prompt_input_a.prompt or not prompt_input_b.prompt:
        print("Enter both prompts to interpolate between them")
        return

    seed_path = "seed_images/riffusion/"
    if init_image_name is not None:
        init_image = Image.open(f'{os.path.join(seed_path, init_image_name)}.png').convert("RGB")
    else:
        init_image = None

    image_list = interpolate(
        pipeline=pipeline,
        alphas=alphas,
        num_inference_steps=num_inference_steps,
        prompt_input_a=prompt_input_a,
        prompt_input_b=prompt_input_b,
        init_image=init_image,
        mask_image=mask_image,
    )

    target_dir = os.path.join(target_dir, prompt_input_a_text.split()[0] + '_' + prompt_input_b_text.split()[0])
    if save_image:
        spectrogram_path = os.path.join(target_dir, 'Spectrogram/')
        save_images(image_list, spectrogram_path)
        print(f"Saved all Spectrogram images to: {spectrogram_path}")

    audio_segments, combined_audio = reconstruct_audio_from_spectrograms(list_imgs=image_list,
                                                                         save_results=save_audio,
                                                                         target_dir=target_dir)

    return image_list, audio_segments, combined_audio


def generate_image(pipeline: RiffusionPipeline,
                   prompt_input_a: PromptInput,
                   prompt_input_b: PromptInput,
                   alpha: float = 1.0,
                   num_inference_steps: int = 50,
                   init_image: Image = None,
                   mask_image: Image = None) -> Image:
    inputs = InferenceInput(
        alpha=float(alpha),
        num_inference_steps=num_inference_steps,
        seed_image_id="og_beat",
        start=prompt_input_a,
        end=prompt_input_b,
    )

    image = pipeline.riffuse(
        inputs,
        init_image=init_image,
        mask_image=mask_image
    )
    return image

def interpolate(
        pipeline: RiffusionPipeline,
        alphas: np.ndarray,
        num_inference_steps: int,
        prompt_input_a: PromptInput,
        prompt_input_b: PromptInput,
        init_image: Image,
        mask_image: Image = None,
) -> T.List[Image]:
    """
    Interpolate between two prompts and return the generated images and audio bytes.
    """

    image_list = []
    # Perform interpolation
    for alpha in alphas:
        image = generate_image(
            pipeline=pipeline,
            alpha=alpha,
            num_inference_steps=num_inference_steps,
            prompt_input_a=prompt_input_a,
            prompt_input_b=prompt_input_b,
            init_image=init_image,
            mask_image=mask_image,
        )

        image_list.append(image)

    return image_list


def init_pipeline(lora_weights_path: str = "lewtun-3000") -> RiffusionPipeline:
    lora_weights_path = os.path.join("checkpoints", lora_weights_path)
    model_id = "runwayml/stable-diffusion-v1-5"
    accelerator = Accelerator()
    pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(accelerator.device)
    pipeline.load_lora_weights(lora_weights_path)
    noise_scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipeline = RiffusionPipeline(pipeline.vae,
                                 pipeline.text_encoder,
                                 pipeline.tokenizer,
                                 pipeline.unet,
                                 # noise_scheduler,
                                 pipeline.scheduler,
                                 pipeline.safety_checker,
                                 pipeline.feature_extractor)
    return pipeline


def init_riffusion_pipeline() -> RiffusionPipeline:
    checkpoint = "riffusion/riffusion-model-v1"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return RiffusionPipeline.load_checkpoint(checkpoint, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    cap1 = "soft female voice"
    cap2 = "techno music"

    parser.add_argument("--prompt_input_a_text", type=str, default=cap1)
    parser.add_argument("--prompt_input_b_text", type=str, default=cap2)
    parser.add_argument("--lora_weights_path", type=str, default="musiccaps-4500")
    parser.add_argument("--init_image_name", type=str, default="og_beat")
    parser.add_argument("--target_dir", type=str, default="temp/Results")
    parser.add_argument("--num_inference_steps", type=int, default=38)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--prompt_input_a_seed", type=int, default=42)
    parser.add_argument("--prompt_input_a_denoising", type=float, default=0.75)
    parser.add_argument("--prompt_input_b_seed", type=int, default=42)
    parser.add_argument("--prompt_input_b_denoising", type=float, default=0.75)
    parser.add_argument("--mask_image", type=str, default=None)
    parser.add_argument("--num_interpolation_steps", type=int, default=5)
    parser.add_argument("--alpha_power", type=float, default=1.0)
    parser.add_argument("--save_audio", type=bool, default=True)
    parser.add_argument("--save_image", type=bool, default=True)
    args = parser.parse_args()

    # pipeline = init_pipeline(args.lora_weights_path)
    pipeline = init_riffusion_pipeline()

    infer_prompts(
        pipeline=pipeline,
        prompt_input_a_text=args.prompt_input_a_text,
        prompt_input_b_text=args.prompt_input_b_text,
        target_dir=args.target_dir,
        num_interpolation_steps=args.num_interpolation_steps,
        num_inference_steps=args.num_inference_steps,
        guidance=args.guidance,
        init_image_name=args.init_image_name,
        alpha_power=args.alpha_power,
        prompt_input_a_seed=args.prompt_input_a_seed,
        prompt_input_a_denoising=args.prompt_input_a_denoising,
        prompt_input_b_seed=args.prompt_input_b_seed,
        prompt_input_b_denoising=args.prompt_input_b_denoising,
        mask_image=args.mask_image,
        save_audio=args.save_audio,
        save_image=args.save_image,
    )