#!/usr/bin/env python

# Used for generating the "labels" category of dataset.json for nnUnet.

import json
import argparse

def parse_labels_file(file_path: str):
    label_map = {"background": 0}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split('\t')
            # Assuming IDX is the first part and LABEL is the last part
            if len(parts) >= 8: # Ensure there are enough columns
                try:
                    region_id = int(parts[0])
                    label = parts[7].strip('\"\'')
                    label_map[label] = region_id
                except ValueError:
                    # Skip lines where IDX is not an integer
                    continue
    return label_map

def main():
    parser = argparse.ArgumentParser(description="Parse labels.txt file and output JSON.")
    parser.add_argument("file_path", type=str, help="Path to the labels.txt file")
    args = parser.parse_args()

    result = parse_labels_file(args.file_path)
    print(json.dumps(result, indent=4))

if __name__ == "__main__":
    main()
