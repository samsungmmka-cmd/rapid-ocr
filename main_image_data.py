"""Extract ALL data from an image (or scanned PDF) with RapidOCR.

Run:  python main_image_data.py data/invoice1.png [--accuracy fast|balanced|best]
Writes JSON, Markdown, plain text and CSV files to ./output/<name>/.
"""

import argparse
import logging
import sys
from pathlib import Path

from pdf_table_extractor import ExtractionError, ImageDataConfig, ImageDataExtractor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--accuracy", default="balanced", choices=["fast", "balanced", "best"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    extractor = ImageDataExtractor(ImageDataConfig(accuracy=args.accuracy))
    try:
        result = extractor.extract(args.path)
    except ExtractionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    print(result.text)
    print("\n=== Fields ===")
    for kv in result.key_values:
        print(f"{kv.key}: {kv.value}" + (f"  (number={kv.number})" if kv.number is not None else ""))
    for table in result.tables:
        print(f"\n=== Table {table.index + 1} ({table.method}) ===")
        for record in table.records():
            print(record)
    print(f"\nQuality: {result.quality_level}")
    for page in result.pages:
        for note in page.quality.get("notes", []):
            print(f"  note: {note}")

    out = Path("output") / Path(args.path).stem
    for path in result.save(out):
        print(f"wrote {path}")
    return 0 if result.ok else 4


if __name__ == "__main__":
    sys.exit(main())
