import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = "/workspace/SC_artifacts_eval/"
INPUT_PATH  = os.path.join(BASE_DIR, "dlrm_dataset/spotify")
input_file = os.path.join(INPUT_PATH, "spotify_session.csv")


def load_data(filename):
    data = pd.read_csv(input_file)
    return data

def create_datasets(data, val_size=0.2):
    train_data, val_data = train_test_split(data, test_size=val_size, random_state=42)
    return train_data, val_data

def write_to_files(train_data, val_data, train_file, val_file):
    train_data.to_csv(train_file, index=False, sep="\t")
    val_data.to_csv(val_file, index=False, sep="\t")

def main():
    data = load_data(INPUT_PATH)
    print("data loaded")
    train_data, val_data = create_datasets(data)

    train_file = "/workspace/SC_artifacts_eval/processed_data/spotify/train_subset.txt"
    val_file = "/workspace/SC_artifacts_eval/processed_data/spotify/val_subset.txt"

    write_to_files(train_data, val_data, train_file, val_file)

if __name__ == "__main__":
    main()
