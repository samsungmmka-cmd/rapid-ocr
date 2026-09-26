"""Library usage example: print every table in a PDF as structured records.

Digital pages are read from the text layer; scanned pages are OCR'd with
RapidOCR automatically. Run:  python main_pdf.py [file.pdf]
"""

import logging
import sys

from pdf_table_extractor import ExtractionConfig, ExtractionError, extract_tables


def run(pdf_path: str) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        result = extract_tables(pdf_path, ExtractionConfig())
    except ExtractionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    for table in result.tables:
        pages = ", ".join(map(str, table.pages))
        print(f"\n===== {table.title or 'Untitled table'} (page {pages}, {table.method}, confidence {table.confidence:.2f}) =====")
        for record in table.records():
            print(record)
    if not result.ok:
        print(f"\nwarning: pages failed: {result.failed_pages}", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1] if len(sys.argv) > 1 else "invoice.pdf"))
