# PDF & Image Table Extractor

Extracts tables from PDF files and image files as structured data (JSON records, CSV, Markdown).

| Input | How it is read |
|---|---|
| Digital PDF | text layer, exact |
| Scanned or mixed PDF | text layer where present, OCR for image pages |
| Encrypted PDF | with `--password-env` |
| PNG, JPEG, TIFF (multi-page), BMP, WebP, GIF, JPEG 2000, PNM | OCR with quality check and enhancement |

The file type is detected from the content, so a misnamed file is still read correctly. The PDF reader is unchanged by image support.

* **Digital PDFs** are read from the text layer with PyMuPDF. Text is exact, with no OCR errors.
* **Scanned or image-only pages** are rendered, deskewed and searched for ruled grids with OpenCV. Text is read with RapidOCR and placed into cells. Cells where one text detection crosses a column rule, or where ink was missed, are re-read one by one.
* Each table gets its **section title**, including a heading left at the bottom of the previous page.
* Tables that **continue across pages** are merged, and repeated header rows are removed.

## Full content of any PDF: `DocumentDataExtractor`

`DocumentDataExtractor` returns the full content of a PDF: text in reading order, headings, paragraphs, lists, fields and tables. It picks a method per page:

| Page type | Method |
|---|---|
| Digital (usable text layer) | exact native text. Tables come from the unchanged table extractor, with titles and cross-page merging. |
| Digital, with boxed layout read as one big table | layout analysis on the exact native text |
| Scanned or image-only | OCR ensemble (see `ImageDataExtractor` below) |
| Garbled text layer (broken font encoding) | OCR instead of the unusable text |
| Form (AcroForm) | typed field values are returned as fields |
| Blank | skipped |
| Encrypted | needs `password` |

It also accepts images, so it is the single entry point for any document. The result has the same shape as `ImageDataExtractor`, with `quality.method` telling which path each page took.

```python
from pdf_table_extractor import DocumentDataExtractor

result = DocumentDataExtractor().extract("data/invoice.pdf")   # PDF or image
print(result.text)
print(result.get("invoice number"))
```

The table-only readers (`TableExtractor`, `DocumentReader`, `--content tables`) are unchanged.

## REST API (FastAPI)

```bash
pip install -r requirements-api.txt
uvicorn pdf_table_extractor.api:app --host 0.0.0.0 --port 8000     # docs at /docs
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness and version |
| `POST /v1/extract` | multipart upload `file` (PDF or image) |

Query parameters for `/v1/extract`:

* `content`: `all` (default, full content) or `tables`.
* `accuracy`: `fast`, `balanced` or `best`.
* `output`: `json`, `markdown` or `text`.
* `pages`: for example `1-3,5`.

Send an encrypted PDF's password as the form field `password`, never in the URL.

```bash
curl -X POST "localhost:8000/v1/extract?output=text" -F "file=@data/hw.jpg"
curl -X POST "localhost:8000/v1/extract?content=tables" -F "file=@data/package.pdf"
curl -X POST "localhost:8000/v1/extract" -F "file=@locked.pdf" -F "password=secret" -H "X-API-Key: $API_KEY"
```

Configure it with environment variables:

* `API_KEY` requires the `X-API-Key` header.
* `MAX_UPLOAD_MB` sets the upload limit (default 50).
* `MAX_CONCURRENT_JOBS` sets parallel extractions (default 2).
* `JOB_WAIT_SECONDS` sets how long a request waits for a free slot before a 503 (default 120).
* `MAX_PAGES` sets pages per document (default 100).
* `MAX_PIXELS` sets pixels per decoded image or rendered page (default 60,000,000).

### Production deployment with an API key

With `APP_ENV=production`, the API refuses to start unless a key of at least 24 characters is configured. It fails closed. Every `/v1/*` call must then send `X-API-Key`. `/health` stays open for load-balancer probes.

1. Put the key in `.env`, which is git-ignored and never copied into the image:
   ```bash
   cp .env.example .env
   python -c "import secrets; print(secrets.token_urlsafe(32))"   # paste as API_KEY in .env
   ```
2. Run it with Docker. Compose loads `.env` into the container:
   ```bash
   docker compose up -d --build
   curl -H "X-API-Key: <your key>" -F file=@data/invoice.pdf localhost:8000/v1/extract
   ```
   Or run it without Docker: `set -a; . ./.env; set +a; uvicorn app:app`.
3. On Vercel, Kubernetes or a cloud host, set `API_KEY` (and `APP_ENV=production`) in the platform's environment-variable or secret settings.
4. To rotate a key, put the new and old keys in the secret, one per line or comma-separated. Move clients to the new key, then remove the old one. No downtime is needed.

**Vercel and other serverless hosts.** `main.py` exposes the FastAPI `app` that Vercel detects. Set `APP_ENV`, `API_KEY` (marked sensitive) and `OCR_MODEL_DIR=/tmp/rapidocr` in the project's environment variables. The dependency bundle is about 411 MB, under Vercel's 500 MB limit. That depends on installing only the headless OpenCV build. `rapidocr` insists on the full `opencv-python` package (116 MB, needs graphics libraries), and the local shim in `vendor/opencv-python-shim` (plus a `[tool.uv]` override in `pyproject.toml`) satisfies that requirement with the headless build instead. Vercel rejects uploads above 4.5 MB.

Serve the API behind HTTPS, through a reverse proxy or load balancer, so the key is never sent in clear text.

Hardening: oversized requests are rejected before multipart parsing. Image frames and PDF pages are validated from their headers and decoded one at a time, with pixel caps. Page ranges are bounded, and client filenames are reduced to a clean base name. When `API_KEY` is unset the API is open, so set it, or put the API behind an authenticating gateway, in production.

Errors return JSON `{"error", "detail"}` with these status codes:

| Status | Meaning |
|---|---|
| 401 | missing or wrong API key |
| 413 | file too large |
| 415 | unsupported file type |
| 422 | invalid input, encrypted PDF without a password, or bad pages |
| 503 | busy, or OCR unavailable |

OCR models load once per process and are shared by all requests.

## Extract ALL data from images: `ImageDataExtractor`

The table reader above returns tables only. `ImageDataExtractor` returns everything on the page, in reading order:

* **Text blocks:** headings, paragraphs and list items. Multi-column layouts are read column by column.
* **Fields:** key-value pairs written inline (`Date: 28/10/2024`, `PAN No. DODPS7841F`) or side by side (`INVOICE #   US-001`). Totals under a table (`Subtotal`, `Tax`, `TOTAL`) are also fields. Numeric values get a parsed `number`.
* **Tables:** ruled tables, including coloured or hand-drawn rules and rows separated only by handwriting. Borderless tables are found from their header row.
* **Uncertain fragments:** signatures, stamps and scribbles are kept but marked, so they don't pollute fields.
* **Every line:** text, confidence, the model that read it, alternative readings, and a polygon on the **original** image, even after rotation, perspective correction, upscaling or deskewing.
* **Review list and quality notes:** low-confidence lines are listed for a human to check.

```python
from pdf_table_extractor import ImageDataConfig, ImageDataExtractor

extractor = ImageDataExtractor(ImageDataConfig(accuracy="balanced"))  # reuse: models load once
result = extractor.extract("data/hw.jpg")      # image, scanned PDF, or bytes
print(result.text)                             # full text in reading order
print(result.get("invoice date"))              # field lookup, case/punctuation-insensitive
for kv in result.key_values:
    print(kv.key, kv.value, kv.number, kv.confidence)
for table in result.tables:
    print(table.records())
result.save("output/hw")                       # .data.json, .data.md, .txt, one CSV per table
```

From the command line, add `--content all`:

```bash
python -m pdf_table_extractor data/ --content all -o output             # every image/PDF in data/
python -m pdf_table_extractor photo.jpg --content all --accuracy best   # slowest, most accurate
python main_image_data.py data/invoice1.png                              # example script
```

**How accuracy is achieved.** A fast multilingual model (PP-OCRv6 small) reads the whole page first. Doubtful lines are cropped along their rotated box and re-read by two stronger RapidOCR recognisers. PP-OCRv6 *medium* is much better on handwriting. PP-OCRv5 *English* is fast and cannot output Chinese look-alike characters. The best reading wins, and guards stop a shorter or foreign-script reading from replacing a good one. In a numeric table column, a non-numeric misread such as `ost` for a handwritten `750` is re-read, and a numeric reading is preferred.

| Accuracy profile | What it does | Speed on the sample invoices |
|---|---|---|
| `fast` | base model only | about 2–5 s per page |
| `balanced` (default) | re-reads lines below 0.93 confidence | about 8 s printed, 25 s handwritten |
| `best` | re-reads lines below 0.985, adds an upscaled re-read | slower |

The medium (77 MB) and English (8 MB) models download on first use. On an offline server, run the extractor once while online, or set `accuracy="fast"`. If a model cannot load, extraction continues with the others, and the quality notes say so.

**Handwriting.** Printed text is read reliably. Handwriting is read on a best-effort basis: numbers and short words are usually right, while cursive words are often approximate. Every such line appears in the review list with its alternatives. The result's quality level is `fair` or `poor` in that case.

## Unclear images and quality notes

Photos and poor scans are checked before OCR. Only the corrections the image needs are applied:

* **Orientation:** sideways or upside-down pages are rotated upright, and EXIF rotation from phones is honoured.
* **Photographed pages:** a page shot at an angle on a darker surface is flattened.
* **Shadows and dark images:** the background is flattened.
* **Noise and grain:** the image is denoised.
* **Faint text:** contrast is stretched.
* **Small text:** the image is upscaled so glyphs are about 22 px tall.
* **Blur:** the image is sharpened.
* **Tilt:** the page is deskewed.

If OCR confidence stays low, or a degraded image yields no table, a second, stronger enhancement pass runs and the better result is kept.

Every image result carries a `quality` block per page. It holds a level (`good`, `fair` or `poor`), issue codes, plain-language notes, the enhancements applied, OCR confidence and the raw metrics. The notes also appear in the Markdown report and on the terminal. Tables from `fair` or `poor` pages get a warning. `--strict-quality` makes the CLI exit with code 5 when any image is `poor`.

Example note: *"Low resolution: characters are about 6px tall (recommended 20px or more, e.g. scan at 300 DPI). The image was upscaled, but small digits and punctuation may still be misread."*

For best results, scan at 300 DPI and photograph flat pages in even light and in focus.

## Install

```bash
python -m venv venv && . venv/bin/activate
pip install -r requirements.txt        # runtime
pip install -e ".[dev]"                # optional: CLI entry point + pytest
```

The RapidOCR models ship with the `rapidocr` wheel, so the tool runs offline.

## Command line

```bash
python -m pdf_table_extractor package.pdf -o output          # JSON + CSV per table + Markdown
python -m pdf_table_extractor photo.jpg scan.tiff -o output   # images, one sub-folder each
python -m pdf_table_extractor inbox/ -r -o output             # every PDF/image in a folder
python -m pdf_table_extractor package.pdf --stdout --pages 1-2
python -m pdf_table_extractor scans/*.pdf --mode ocr --dpi 300 --borderless
PDF_PW=secret python -m pdf_table_extractor locked.pdf --password-env PDF_PW
```

| Option | Meaning |
|---|---|
| `--mode auto\|native\|ocr` | `auto` uses the text layer when present and OCR otherwise |
| `--pages 1-3,5` | 1-based page selection |
| `--borderless` | also detect whitespace-aligned tables on OCR pages |
| `--header-rows N` | number of header rows; several are joined as `Top / Sub` |
| `--no-merge`, `--no-deskew` | disable cross-page merging or deskewing |
| `--log-format json` | structured logs on stderr |
| `-r`, `--recursive` | search input folders recursively |
| `--no-enhance` | images: skip enhancement |
| `--strict-quality` | exit code 5 when an image's quality is `poor` |

Exit codes are `0` ok, `2` usage error, `3` invalid, encrypted, unsupported or oversized input, `4` partial (some pages failed), `5` poor image quality with `--strict-quality`, and `1` unexpected error.

## Library

```python
from pdf_table_extractor import DocumentReader, ImageDocumentResult

reader = DocumentReader()                 # reuse: OCR model loads once
result = reader.extract("photo.jpg")      # PDF or image, path or bytes
for table in result.tables:
    print(table.title, table.records())
if isinstance(result, ImageDocumentResult):
    print(result.quality_level, result.quality_notes())
```

## Output (`<name>.tables.json`, schema 1.0)

Each table contains `title`, `page`, `pages`, `bbox` (PDF points), `method` (`native`, `ocr-ruled` or `ocr-borderless`), `confidence`, `header`, `rows`, `records`, `low_confidence_cells` and `warnings`. The document level adds `sha256`, per-page method and timing, `status` (`ok` or `partial`) and `failed_pages`.

Image results also contain `input_type`, `image_format`, `quality_level` and `quality`. For images, `bbox` is measured on the processed (rotated and enhanced) image.

OCR cells scoring below `review_score_threshold` (default 0.85) are listed in `low_confidence_cells` for human review.

## Operational notes

* A failing page is recorded in the result and does not stop the document.
* Limits for file size, page count and rendered pixels guard memory. The OCR engine is shared behind a lock, so one extractor can serve several threads.
* Known limits: HEIC/HEIF photos need conversion to JPEG first. Handwriting is best-effort (see above). Images with text under about 8 px, or heavy motion blur, are reported as `poor` and should be rescanned. Merged cells in scans are assigned to the cell holding their text centre. OCR can drop spaces, for example `7GB` instead of `7 GB`. There is no per-page timeout, so run untrusted input in a worker process with its own timeout.

## Tests

```bash
python -m pytest            # all tests, including real OCR on data/ (about 3 min)
python -m pytest -m "not ocr"
```
