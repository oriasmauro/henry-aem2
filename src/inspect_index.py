"""
Inspect an index file and print its metadata and chunk summaries.
"""

from __future__ import annotations

import argparse

from utils import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a generated RAG index JSON")
    parser.add_argument(
        "--index-path",
        default="storage/index_openai.json",
        help="Path to the index file to inspect",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=500,
        help="How many characters of each chunk to print",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_json(args.index_path)

    print(f"\nIndex path     : {args.index_path}")
    print(f"Provider       : {data.get('provider', 'unknown')}")
    print(f"Embedding model: {data.get('embedding_model', 'unknown')}")
    print(f"Embedding dim  : {data.get('embedding_dim', 'unknown')}")
    print(f"Chunk size     : {data.get('chunk_size_tokens', 'unknown')}")
    print(f"Overlap        : {data.get('overlap_tokens', 'unknown')}")
    print(f"Total chunks   : {len(data['chunks'])}\n")

    for chunk in data["chunks"]:
        print("=" * 80)
        print(f"Chunk ID: {chunk['id']}")
        print(f"Tokens  : {chunk['tokens']}")
        print("-" * 80)
        print(chunk["text"][: args.preview_chars])
        print()


if __name__ == "__main__":
    main()
