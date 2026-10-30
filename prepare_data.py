import argparse
import os

from src.data import build_splits, download_bbq
from src.mmlu import MMLU_PATH, build_subset


def main():
    parser = argparse.ArgumentParser(description="Download BBQ and build the splits and the MMLU subset.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    download_bbq()
    build_splits(seed=args.seed)
    if not os.path.exists(MMLU_PATH):
        build_subset(seed=args.seed)


if __name__ == "__main__":
    main()
