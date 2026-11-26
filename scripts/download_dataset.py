
from datasets import load_dataset
from pathlib import Path

# Dataset identifier on Hugging Face
DATASET_NAME = "phxdev/corporate-speak-dataset"

# Local storage directory
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "huggingface"

def main():
    print(f"Downloading dataset: {DATASET_NAME}")

    dataset = load_dataset(DATASET_NAME)

    print("\nDataset downloaded successfully.")
    print(dataset)

    # Save all splits locally
    dataset.save_to_disk(str(OUTPUT_DIR))

    print(f"\nDataset saved to: {OUTPUT_DIR}")

    # Display basic information
    for split_name, split_data in dataset.items():
        print(f"\nSplit: {split_name}")
        print(f"Rows: {len(split_data)}")
        print(f"Columns: {split_data.column_names}")

        if len(split_data) > 0:
            print("\nFirst record:")
            print(split_data[0])

if __name__ == "__main__":
    main()
