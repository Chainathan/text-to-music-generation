# Function to reconstruct audio snippets from spectrogram image objects
import glob
import io
import os
import typing as T
import librosa
import numpy as np
import pydub
import soundfile as sf
from PIL import Image
from scipy.io import wavfile

# Path to your FFmpeg bin directory
ffmpeg_bin_path = r'C:\ffmpeg\bin'  # Update this path

# Add FFmpeg to PATH
os.environ["PATH"] += os.pathsep + ffmpeg_bin_path

def reconstruct_audio_from_spectrograms(
        list_imgs: T.List[np.ndarray],
        sr: int = 44100,
        n_fft: T.Optional[int] = 2048,
        hop_length: T.Optional[int] = 512,
        n_mels: T.Optional[int] = 256,
        save_results: T.Optional[bool] = False,
        target_dir: T.Optional[str] = "temp/Reconstructed_Audio"
):
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
        loaded_image = Image.open(spectrogram_path)
    else:
        loaded_image = spectrogram_img

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

    return snippet_reconstructed_pydub


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


def apply_filters(segment: pydub.AudioSegment, compression: bool = False) -> pydub.AudioSegment:
    """
    Apply post-processing filters to the audio segment to compress it and keep at a -10 dBFS level.
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
    desired_db = -5
    # segment = segment.apply_gain(desired_db - segment.dBFS)
    segment = pydub.effects.normalize(
        segment,
        headroom=0.1,
    )
    return segment


if __name__ == "__main__":
    source_folder = "temp/Results/"
    imgs = [Image.open(img_path) for img_path in glob.glob(os.path.join(source_folder, "*.png"))]
    output = reconstruct_audio_from_spectrograms(list_imgs=imgs, save_results=True)