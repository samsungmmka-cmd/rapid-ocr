"""Command-line entry point: ``python main.py package.pdf -o output``
(same as ``python -m pdf_table_extractor``). The web API entrypoint is app.py."""

import sys

from pdf_table_extractor.cli import main

if __name__ == "__main__":
    sys.exit(main())
