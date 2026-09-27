"""Project entry point.

* Vercel / ASGI servers: imports ``app`` (the FastAPI instance) from here,
  e.g. ``uvicorn main:app``.
* Command line: ``python main.py package.pdf -o output`` (same as
  ``python -m pdf_table_extractor``).
"""

import sys

from pdf_table_extractor.api import app  # noqa: F401  (top-level FastAPI app for Vercel)
from pdf_table_extractor.cli import main

if __name__ == "__main__":
    sys.exit(main())
