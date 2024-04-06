import os
import csv
import pandas as pd

def generate_metadata(meta_dir, source_file):
    meta_csv_path = meta_dir + '/metadata.csv'
    # Check if the file exists
    os.makedirs(os.path.dirname(meta_csv_path), exist_ok=True)
    # Initialize the CSV file with headers
    with open(meta_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['file_name', 'text'])

    # Convert metadata from source file
    with open(meta_csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        df = pd.read_csv(source_file)
        for index, row in df.iterrows():
            # Ensure aspects are correctly formatted as a single string
            meta_str = ', '.join(eval(row['aspect_list']))
            writer.writerow([f"{row['ytid']}.png", meta_str])

if __name__ == '__main__':
    source_file = "../Dataset/train/musiccaps-public.csv"
    target_folder = "../Dataset/train"
    generate_metadata(target_folder, source_file)