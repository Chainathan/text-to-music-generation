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

        # Length of each snippet in samples
        snippet_length_samples = snippet_length_sec * sr

        # Calculate total number of snippets
        total_snippets = len(y) // snippet_length_samples

        # Handle the last snippet if there's remaining audio
        if len(y) % snippet_length_samples != 0:
            total_snippets += 1

        # Create a folder for the current audio clip
        clip_name = os.path.splitext(audio_file)[0]
        clip_folder = os.path.join(target_folder, clip_name)
        if not os.path.exists(clip_folder):
            os.makedirs(clip_folder)

        # Process each snippet
        for i in range(total_snippets):
            # Determine the start and end index of the current snippet
            start_index = i * snippet_length_samples
            end_index = min((i + 1) * snippet_length_samples, len(y))

            # Extract the snippet
            snippet = y[start_index:end_index]

            # Generate Mel spectrogram
            S = librosa.feature.melspectrogram(y=snippet, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
                                               window='hamming')

            # Add padding to the spectrogram if necessary
            if S.shape[1] < (snippet_length_samples // hop_length) + 1:
                pad_width = ((0, 0), (0, (snippet_length_samples // hop_length) + 1 - S.shape[1]))
                S = np.pad(S, pad_width=pad_width, mode='constant', constant_values=0)

            S_db = librosa.power_to_db(S, ref=np.max)

            # Normalize and convert to an image
            S_normalized = np.clip((S_db + 80) / 80, 0, 1)
            mel_image = np.uint8(S_normalized * 255)

            # Save the spectrogram as an image
            spectrogram_filename = f"{clip_name}_{i}.tiff"
            # spectrogram_path = os.path.join(clip_folder, spectrogram_filename)
            spectrogram_path = os.path.join('Data/Spectrogram', os.path.split(spectrogram_filename)[1])
            Image.fromarray(mel_image).save(spectrogram_path, format='TIFF')

            print(f"Saved padded spectrogram snippet to: {spectrogram_path}")

if __name__ == "__main__":
    #  usage
    source_folder = 'Data/'
    target_folder = 'Data/Spectrogram'
    generate_padded_spectrograms(source_folder, target_folder)