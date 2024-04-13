import glob
import os
import soundfile as sf
import torch
from PIL import Image
from riffusion.spectrogram_params import SpectrogramParams
from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from utils.Reconstruct_Audio import stitch_segments

# Riffusion dependencies
params = SpectrogramParams(
    min_frequency=0,
    max_frequency=10000,
    num_frequencies=256,
)
device_type = "cuda" if torch.cuda.is_available() else "cpu"
converter = SpectrogramImageConverter(params=params, device=device_type)

# Function to reconstruct audio snippets from spectrogram images in a folder
def reconstruct_audio_from_spectrograms(source_folder, target_folder, sr=44100, n_fft=2048, hop_length=512, n_mels=256):
    # Create the target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    audio_segments = []
    # Process each spectrogram image in the source folder
    for spectrogram_file in glob.glob(os.path.join(source_folder,"*.png")):
        # spectrogram_path = os.path.join(source_folder, spectrogram_file)
        spectrogram_path = spectrogram_file

        # Load the RGB spectrogram image
        loaded_image = Image.open(spectrogram_path).convert("RGB")
        segment = converter.audio_from_spectrogram_image(
            loaded_image,
            apply_filters=True,
        )
        audio_segments.append(segment)

        # Save the reconstructed audio snippet
        reconstructed_audio_file = f"{os.path.split(os.path.splitext(spectrogram_file)[0])[1]}.wav"
        reconstructed_audio_path = os.path.join(target_folder, reconstructed_audio_file)
        sf.write(reconstructed_audio_path, segment.get_array_of_samples(), sr)

    # Stitch audio segments
    combined_segment = stitch_segments(audio_segments, crossfade_s=0)

    # Save the combined audio snippet
    reconstructed_audio_file = f"{os.path.split(os.path.splitext(spectrogram_file)[0])[1][:-2]}.wav"
    # reconstructed_audio_path = os.path.join(target_folder, reconstructed_audio_file)
    reconstructed_audio_path = os.path.join(target_folder, reconstructed_audio_file)
    sf.write(reconstructed_audio_path, combined_segment.get_array_of_samples(), sr)

if __name__ == "__main__":
    # Example usage
    source_folder = '../temp/Results/riffusion/'
    target_folder = '../temp/Results/riffusion/Reconstructed_Audio'
    reconstruct_audio_from_spectrograms(source_folder, target_folder)