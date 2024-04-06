import glob
import os
import librosa
import numpy as np
from PIL import Image


# Function to generate padded spectrograms for audio clips in a folder
def generate_padded_spectrograms(source_folder, target_folder):
    # Create the target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Set parameters for Mel spectrogram computation
    sr = 44100  # Sample rate
    n_fft = 2048  # Number of FFT points
    hop_length = 512  # Hop length for FFT
    n_mels = 256  # Number of Mel frequency bins
    snippet_length_sec = 5  # Length of each snippet in seconds

    # Process each audio clip in the source folder
    for audio_file in glob.glob(os.path.join(source_folder, '*.wav')):
        # audio_path = os.path.join(source_folder, audio_file)
        audio_path = audio_file

        # Load the audio file
        y, _ = librosa.load(audio_path, sr=sr)

        if y.ndim == 2:  # Check if the audio is stereo
            y_left, y_right = y[0], y[1]

            # Generate Mel spectrogram for each channel
            S_left = librosa.feature.melspectrogram(y=y_left, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            S_right = librosa.feature.melspectrogram(y=y_right, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                                     n_mels=n_mels)

            # Convert to decibels
            S_left_db = librosa.power_to_db(S_left, ref=np.max)
            S_right_db = librosa.power_to_db(S_right, ref=np.max)

            # Normalize
            S_left_normalized = np.clip((S_left_db + 80) / 80, 0, 1)
            S_right_normalized = np.clip((S_right_db + 80) / 80, 0, 1)

            # Stack to create a 3-channel image (Left, Right, Right)
            stereo_image = np.stack([S_left_normalized, S_right_normalized, S_right_normalized], axis=-1)

            # Convert to an RGB image
            stereo_image_rgb = np.uint8(stereo_image * 255)

            # Save the image
            spectrogram_filename = f"{os.path.splitext(audio_file)[0]}.png"
            spectrogram_path = os.path.join(target_folder, spectrogram_filename)
            Image.fromarray(stereo_image_rgb).save(spectrogram_path)

            print(f"Saved stereo spectrogram to: {spectrogram_path}")
        else:
            # Generate Mel spectrogram for each channel
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

            # Convert to decibels
            S_db = librosa.power_to_db(S, ref=np.max)

            # Normalize
            s_normalized = np.clip((S_db + 80) / 80, 0, 1)

            # Stack to create a 3-channel image (Mono, Mono, Mono)
            mono_image = np.stack([s_normalized, s_normalized, s_normalized], axis=-1)

            # Convert to an RGB image
            stereo_image_rgb = np.uint8(mono_image * 255)

            # Save the image
            spectrogram_filename = f"{os.path.splitext(audio_file)[0]}.png"
            spectrogram_path = os.path.join(target_folder, spectrogram_filename)
            Image.fromarray(stereo_image_rgb).save(spectrogram_path)

            print(f"Saved stereo spectrogram to: {spectrogram_path}")

if __name__ == "__main__":
    #  usage
    source_folder = 'Dataset/'
    target_folder = 'Dataset/Spectrogram'
    generate_padded_spectrograms(source_folder, target_folder)