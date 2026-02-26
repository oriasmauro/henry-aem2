"""
Script to inspect the contents of the index.json file, which contains the document chunks and their embeddings. Useful for debugging and understanding how the document was processed.
"""

from __future__ import annotations

import json
from pathlib import Path

INDEX_PATH = Path("storage/index.json")


def main():
    data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))

    print(f"\nTotal chunks: {len(data['chunks'])}\n")

    for chunk in data["chunks"]:
        print("=" * 80)
        print(f"Chunk ID: {chunk['id']}")
        print(f"Tokens  : {chunk['tokens']}")
        print("-" * 80)
        print(chunk["text"][:500])  # primeros 500 chars
        print()


if __name__ == "__main__":
    main()
