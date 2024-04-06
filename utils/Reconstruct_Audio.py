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
    for spectrogram_file in glob.glob(os.path.join(source_folder,"*.png")):
        # spectrogram_path = os.path.join(source_folder, spectrogram_file)
        spectrogram_path = spectrogram_file

        # Load the RGB spectrogram image
        loaded_image = Image.open(spectrogram_path)
        loaded_image_array = np.array(loaded_image)

        # Separate the RGB channels
        red_channel, green_channel, _ = loaded_image_array.transpose((2, 0, 1))

        # Convert pixel values back to Mel spectrogram dB values for each channel
        red_mel_spectrogram_db = (red_channel.astype(np.float32) / 255 * 80) - 80
        green_mel_spectrogram_db = (green_channel.astype(np.float32) / 255 * 80) - 80

        # Convert Mel spectrogram back to power spectrogram for each channel
        red_mel_spectrogram = librosa.db_to_power(red_mel_spectrogram_db)
        green_mel_spectrogram = librosa.db_to_power(green_mel_spectrogram_db)

        # Inverse Mel spectrogram to audio for each channel
        left_channel_audio = librosa.feature.inverse.mel_to_audio(red_mel_spectrogram, sr=sr, n_iter=32)
        right_channel_audio = librosa.feature.inverse.mel_to_audio(green_mel_spectrogram, sr=sr, n_iter=32)

        # Combine the two channels into a stereo audio file
        stereo_audio = np.vstack([left_channel_audio, right_channel_audio]).T

        # Brijesh: Apply post-processing.
        # Write to the bytes of a temp WAV file. Using scipy.io.wavfile.
        wav_bytes = io.BytesIO()
        wavfile.write(wav_bytes, sr, stereo_audio)
        wav_bytes.seek(0)

        snippet_reconstructed_pydub = pydub.AudioSegment.from_wav(wav_bytes)
        snippet_reconstructed_pydub = apply_filters(
            snippet_reconstructed_pydub,
            compression=False,
        )

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