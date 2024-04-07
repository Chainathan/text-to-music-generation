# Convert hugging face dataset lewtun to spectrogram images

from datasets import load_dataset
import csv
import os
import librosa
import numpy as np
from PIL import Image

def convert_to_spectrogram(dataset, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Set parameters for Mel spectrogram computation
    sr = 44100  # Sample rate
    n_fft = 2048  # Number of FFT points
    hop_length = 512  # Hop length for FFT
    n_mels = 256  # Number of Mel frequency bins

    for split in ['train']:
        split_folder = os.path.join(target_folder, split)
        if not os.path.exists(split_folder):
            os.makedirs(split_folder)
            
        metadata_file = os.path.join(split_folder, 'metadatat3.csv')
        # Check if the file exists; write headers only if it doesn't
        if not os.path.exists(metadata_file):
            with open(metadata_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['file_name', 'text'])

        # for i, row in enumerate(dataset[split]):
        for j in range(3102,4000):
            row = dataset[split][j]
            i = row['song_id']
            audio = row['audio']
            y = audio['array']
            if sr != audio['sampling_rate']:
                continue
        
            # Generate Mel spectrogram 
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

            # Convert to decibels
            S_db = librosa.power_to_db(S, ref=np.max)

            # Normalize
            s_normalized = np.clip((S_db + 80) / 80, 0, 1)

            # Stack to create a 3-channel image (Mono, Mono, Mono)
            mono_image = np.stack([s_normalized, s_normalized, s_normalized], axis=-1)

            # Convert to an RGB image
            mono_image_rgb = np.uint8(mono_image * 255)

            # Save the image with id as filename
            spectrogram_filename = f"{i}.png"
            spectrogram_path = os.path.join(split_folder, spectrogram_filename)
            Image.fromarray(mono_image_rgb).save(spectrogram_path)

            print(f"Saved mono spectrogram to: {spectrogram_path} for id {i}")

            # Append entry to metadata CSV
            with open(metadata_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([spectrogram_filename, row['genre']])

if __name__ == "__main__":
    target_folder = "spectrograms"
    dataset = load_dataset("lewtun/music_genres")
    convert_to_spectrogram(dataset, target_folder)