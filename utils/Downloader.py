##%%capture
import csv
import subprocess
import os
from pathlib import Path

from datasets import load_dataset, Audio

def download_dataset(data_dir: str,aspect_data_dir: str, sampling_rate: int = 44100, limit: int = None, num_proc: int = 1, writer_batch_size: int = 1000):
    ds = load_dataset('google/MusicCaps', split='train')
    if limit is not None:
        ds = ds.select(range(limit))

    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)

    # Create the directory for aspect CSV files
    aspect_dir = Path(aspect_data_dir)
    aspect_dir.mkdir(exist_ok=True, parents=True)
    aspect_csv_path = aspect_dir / 'aspects.csv'

    # Initialize the CSV file with headers
    with open(aspect_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'aspects'])

    def process(example):
        import subprocess
        import os
        from pathlib import Path
        def download_clip(video_identifier, output_filename, start_time, end_time, tmp_dir='Data', num_attempts=5, url_base='https://www.youtube.com/watch?v='):
            import subprocess
            import os
            from pathlib import Path
            status = False
            command = f"""yt-dlp --quiet --no-warnings -x --audio-format wav -f bestaudio -o "{output_filename}" --download-sections "*{start_time}-{end_time}" {url_base}{video_identifier}""".strip()
            attempts = 0
            while True:
                try:
                    output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
                except subprocess.CalledProcessError as err:
                    attempts += 1
                    if attempts == num_attempts:
                        return status, err.output
                else:
                    break
            status = os.path.exists(output_filename)
            return status, 'Downloaded'

        outfile_path = str(data_dir / f"{example['ytid']}.wav")
        if not os.path.exists(outfile_path):
            status, log = download_clip(example['ytid'], outfile_path, example['start_s'], example['end_s'])
            example['audio'] = outfile_path
            example['download_status'] = status

        # Save the aspect information in the CSV file
        if example['download_status']:
            with open(aspect_csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Ensure aspects are correctly formatted as a single string
                aspects_str = ';'.join(eval(example['aspect_list']))
                writer.writerow([f"{example['ytid']}.wav", aspects_str])

        return example

    return ds.map(
        process,
        num_proc=num_proc,
        writer_batch_size=writer_batch_size,
        keep_in_memory=False
    ).cast_column('audio', Audio(sampling_rate=sampling_rate))

if __name__ == '__main__':
    data_dir = 'Data'
    aspect_data_dir = 'Data'  # Directory to store the aspects CSV
    ds = download_dataset(data_dir, aspect_data_dir, num_proc=2, limit=2)
