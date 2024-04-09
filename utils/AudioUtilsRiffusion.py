# Function to reconstruct audio snippets from spectrogram image objects
import glob
import os
import typing as T
import pydub
import soundfile as sf
import torch
from PIL import Image
import platform
from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams

if platform.system() == "Windows":
    # Path to your FFmpeg bin directory
    ffmpeg_bin_path = r'C:\ffmpeg\bin'  # Update this path
    # Add FFmpeg to PATH
    os.environ["PATH"] += os.pathsep + ffmpeg_bin_path

# Riffusion dependencies
params = SpectrogramParams(
    min_frequency=0,
    max_frequency=10000,
    num_frequencies=512, # Fixed as we don't control the num_mel_bins in interpolation script for Riffusion
)
device_type = "cuda" if torch.cuda.is_available() else "cpu"
converter = SpectrogramImageConverter(params=params, device=device_type)

def reconstruct_audio_from_spectrograms(
        list_imgs: T.List[Image],
        sr: int = 44100,
        n_fft: T.Optional[int] = 2048,
        hop_length: T.Optional[int] = 512,
        n_mels: T.Optional[int] = 256,
        save_results: T.Optional[bool] = False,
        target_dir: T.Optional[str] = "temp/Reconstructed_Audio"
) -> T.Tuple[T.List[pydub.AudioSegment], pydub.AudioSegment]:
    audio_segments = []
    # Process each spectrogram image in the source folder
    for i, img in enumerate(list_imgs):
        snippet_reconstructed_pydub = image_to_audio(spectrogram_img=img, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                                     n_mels=n_mels)
        # Add to list for stitching
        audio_segments.append(snippet_reconstructed_pydub)
        print(f"Processed reconstruction of Image:{i}")

    # Stitch audio segments
    combined_segment = stitch_segments(audio_segments, crossfade_s=0)
    # Optionally save results
    if save_results:
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        sf.write(os.path.join(target_dir, "combined-output.wav"), combined_segment.get_array_of_samples(), sr)

    return audio_segments, combined_segment

def image_to_audio(spectrogram_path=None, spectrogram_img=None, sr=44100, n_fft=2048, hop_length=512, n_mels=256):
    if spectrogram_path is None and spectrogram_img is None:
        raise ValueError("Either spectrogram_path or spectrogram_img must be provided")
    if spectrogram_img is None:
        # Load the spectrogram image. Load all channels.
        loaded_image = Image.open(spectrogram_path).convert("RGB")
    else:
        loaded_image = spectrogram_img
    segment = converter.audio_from_spectrogram_image(
        loaded_image,
        apply_filters=True,
    )
    return segment

def stitch_segments(
        segments: T.Sequence[pydub.AudioSegment], crossfade_s: float
) -> pydub.AudioSegment:
    """
    Stitch together a sequence of audio segments with a crossfade between each segment.
    """
    crossfade_ms = int(crossfade_s * 1000)
    combined_segment = segments[0]
    for segment in segments[1:]:
        combined_segment = combined_segment.append(segment, crossfade=crossfade_ms)
    return combined_segment

if __name__ == "__main__":
    source_folder = "temp/Results/Riffusion"
    imgs = [Image.open(img_path) for img_path in glob.glob(os.path.join(source_folder, "*.png"))]
    output = reconstruct_audio_from_spectrograms(list_imgs=imgs, save_results=True)