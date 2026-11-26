
from datasets import load_from_disk
from collections import defaultdict

DATASET_PATH = "data/raw/huggingface"

TARGET_INSTRUCTIONS = {
    "Transform this to professional corporate communication",
}

def is_relevant(instruction):
    return (
        instruction in TARGET_INSTRUCTIONS
        or instruction.startswith("Transform multi-part message")
        or instruction == "Transform to corporate speak"
        or instruction == "Translate corporate speak to plain English"
        or instruction == "Translate this corporate speak to plain English"
    )

def main():
    dataset = load_from_disk(DATASET_PATH)

    for split_name, split in dataset.items():
        print(f"\n{'=' * 80}")
        print(f"SPLIT: {split_name}")
        print("=" * 80)

        samples = defaultdict(list)

        for row in split:
            instruction = row["instruction"]

            if is_relevant(instruction) and len(samples[instruction]) < 3:
                samples[instruction].append(row)

        for instruction, rows in sorted(samples.items()):
            print(f"\nINSTRUCTION: {instruction}")
            print("-" * 80)

            for i, row in enumerate(rows, 1):
                print(f"\nEXAMPLE {i}")
                print(f"INPUT:  {row['input']}")
                print(f"OUTPUT: {row['output']}")
                print(f"BIDIRECTIONAL: {row['bidirectional']}")

if __name__ == "__main__":
    main()
