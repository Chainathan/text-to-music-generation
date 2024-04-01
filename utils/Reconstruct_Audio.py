import glob
import io
import os
import librosa
import numpy as np
import pydub
import soundfile as sf
import typing as T
from PIL import Image
from scipy.io import wavfile

# Function to reconstruct audio snippets from spectrogram images in a folder
def reconstruct_audio_from_spectrograms(source_folder, target_folder, sr=44100, n_fft=2048, hop_length=512, n_mels=256):
    # Create the target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    audio_segments = []
    # Process each spectrogram image in the source folder
    for spectrogram_file in glob.glob(os.path.join(source_folder,"*.tiff")):
        # spectrogram_path = os.path.join(source_folder, spectrogram_file)
        spectrogram_path = spectrogram_file

        # Load the spectrogram image
        loaded_image = Image.open(spectrogram_path).convert('L')
        loaded_image_array = np.array(loaded_image)

        # Convert pixel values back to Mel spectrogram dB values
        loaded_mel_spectrogram_db = (loaded_image_array.astype(np.float32) / 255 * 80) - 80

        # Convert Mel spectrogram back to power spectrogram
        loaded_mel_spectrogram = librosa.db_to_power(loaded_mel_spectrogram_db)

        # Inverse Mel spectrogram to audio
        snippet_reconstructed = librosa.feature.inverse.mel_to_audio(loaded_mel_spectrogram, sr=sr, n_iter=32)

        # Brijesh: Apply post-processing.
        # Write to the bytes of a temp WAV file. Using scipy.io.wavfile.
        wav_bytes = io.BytesIO()
        wavfile.write(wav_bytes, sr, snippet_reconstructed)
        wav_bytes.seek(0)

        snippet_reconstructed_pydub = pydub.AudioSegment.from_wav(wav_bytes)
        snippet_reconstructed_pydub = apply_filters(
            snippet_reconstructed_pydub,
            compression=False,
        )

        # Save each file
        # # Save the reconstructed audio snippet
        # reconstructed_audio_file = f"{os.path.splitext(spectrogram_file)[0]}.wav"
        # # reconstructed_audio_path = os.path.join(target_folder, reconstructed_audio_file)
        # reconstructed_audio_path = os.path.join('Data/Reconstructed_Audio', os.path.split(reconstructed_audio_file)[1])
        # # Raj implementation to save audio. Use soundfile module to write audio
        # # sf.write(reconstructed_audio_path, snippet_reconstructed, sr)
        #
        # # Brijesh implementation to save audio.
        # sf.write(reconstructed_audio_path, snippet_reconstructed_pydub.get_array_of_samples(), sr)
        #
        # print(f"Saved reconstructed audio snippet to: {reconstructed_audio_path}")

        # Add to list for stitching
        audio_segments.append(snippet_reconstructed_pydub)
        print(f"Processed reconstruction of :{spectrogram_file}")

    # Stitch audio segments
    combined_segment = stitch_segments(audio_segments, crossfade_s=0)

    # Save the reconstructed audio snippet
    reconstructed_audio_file = f"{os.path.split(os.path.splitext(spectrogram_file)[0])[1][:-2]}.wav"
    # reconstructed_audio_path = os.path.join(target_folder, reconstructed_audio_file)
    reconstructed_audio_path = os.path.join('Data/Reconstructed_Audio', reconstructed_audio_file)
    sf.write(reconstructed_audio_path, combined_segment.get_array_of_samples(), sr)


def apply_filters(segment: pydub.AudioSegment, compression: bool = False) -> pydub.AudioSegment:
    """
    Apply post-processing filters to the audio segment to compress it and
    keep at a -10 dBFS level.
    """

    if compression:
        segment = pydub.effects.normalize(
            segment,
            headroom=0.1,
        )

        segment = segment.apply_gain(-10 - segment.dBFS)

        # TODO(hayk): This is quite slow, ~1.7 seconds on a beefy CPU
        segment = pydub.effects.compress_dynamic_range(
            segment,
            threshold=-20.0,
            ratio=4.0,
            attack=5.0,
            release=50.0,
        )

    desired_db = -12
    # segment = segment.apply_gain(desired_db - segment.dBFS)

    segment = pydub.effects.normalize(
        segment,
        headroom=0.1,
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
    # Example usage
    source_folder = 'Data/Spectrogram'
    target_folder = 'Data/Reconstructed_Audio'
    reconstruct_audio_from_spectrograms(source_folder, target_folder)