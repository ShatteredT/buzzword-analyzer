
from datasets import load_from_disk
from collections import Counter, defaultdict

DATASET_PATH = "data/raw/huggingface"

def main():
    dataset = load_from_disk(DATASET_PATH)

    for split_name, split in dataset.items():
        print(f"\n{'=' * 70}")
        print(f"SPLIT: {split_name} | ROWS: {len(split)}")
        print("=" * 70)

        instruction_counts = Counter()
        bidirectional_counts = Counter()
        instruction_bidirectional = defaultdict(Counter)

        for row in split:
            instruction = row["instruction"].strip()
            bidirectional = row["bidirectional"]

            instruction_counts[instruction] += 1
            bidirectional_counts[str(bidirectional)] += 1
            instruction_bidirectional[instruction][str(bidirectional)] += 1

        print("\nBIDIRECTIONAL DISTRIBUTION:")
        for value, count in bidirectional_counts.items():
            print(f"{value}: {count}")

        print("\nINSTRUCTION TYPES:")
        for instruction, count in instruction_counts.most_common():
            flags = dict(instruction_bidirectional[instruction])
            print(f"{count:4} | {instruction} | bidirectional={flags}")

if __name__ == "__main__":
    main()
