"""Command-line interface for PDFs and image files.

Exit codes: 0 ok, 1 unexpected error, 2 usage error, 3 invalid/encrypted/too
large input, 4 partial success (some pages failed), 5 poor image quality
(only with --strict-quality).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from . import __version__
from .config import ExtractionConfig, ExtractionMode
from .errors import ExtractionError
from .exporters import FORMATS, to_json, write_outputs
from .image_reader import ImageDocumentResult
from .reader import DocumentReader, iter_inputs

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_INPUT, EXIT_PARTIAL, EXIT_POOR_QUALITY = 0, 1, 2, 3, 4, 5
# With several inputs, the most severe outcome decides the exit code.
_SEVERITY = {EXIT_OK: 0, EXIT_POOR_QUALITY: 1, EXIT_PARTIAL: 2, EXIT_INPUT: 3, EXIT_ERROR: 4}


def _worst(a: int, b: int) -> int:
    return a if _SEVERITY[a] >= _SEVERITY[b] else b

log = logging.getLogger("pdf_table_extractor")


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


MAX_PAGE_SPAN = 10_000


def parse_pages(spec: str) -> list[int]:
    """'1-3,5' -> [1, 2, 3, 5]."""
    pages: set[int] = set()
    try:
        for part in filter(None, (p.strip() for p in spec.split(","))):
            if "-" in part:
                a, b = (int(x) for x in part.split("-", 1))
                if a > b or b - a > MAX_PAGE_SPAN:
                    raise ValueError  # also guards against '1-100000000' memory blow-up
                pages.update(range(a, b + 1))
            else:
                pages.add(int(part))
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid page range: {spec!r}") from None
    if not pages or min(pages) < 1 or len(pages) > MAX_PAGE_SPAN:
        raise argparse.ArgumentTypeError("pages are 1-based and must not be empty")
    return sorted(pages)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pdf-tables",
        description=(
            "Extract structured tables from PDFs (native text layer, OCR fallback for scans) and from image files "
            "(PNG, JPEG, TIFF, BMP, WebP, GIF, JPEG 2000) with quality checks and enhancement for unclear images."
        ),
    )
    p.add_argument("inputs", nargs="+", type=Path, help="PDF or image files, or directories containing them")
    p.add_argument("-r", "--recursive", action="store_true", help="search directories recursively")
    p.add_argument("--no-enhance", action="store_true", help="images: skip quality enhancement (not recommended)")
    p.add_argument("--strict-quality", action="store_true", help="exit with code 5 when an image's quality is poor")
    p.add_argument(
        "--content",
        choices=["tables", "all"],
        default="tables",
        help="'all': extract every piece of data (text, headings, fields, tables) with the OCR ensemble "
        "(images and PDFs are OCR'd page by page)",
    )
    p.add_argument("--accuracy", choices=["fast", "balanced", "best"], default="balanced", help="with --content all: speed/accuracy trade-off")
    p.add_argument("-o", "--out-dir", type=Path, default=Path("output"), help="output directory (default: ./output)")
    p.add_argument("-f", "--format", nargs="+", choices=FORMATS, default=list(FORMATS), help="output formats")
    p.add_argument("--stdout", action="store_true", help="print JSON to stdout instead of writing files")
    p.add_argument("--mode", choices=[m.value for m in ExtractionMode], default="auto")
    p.add_argument("--pages", type=parse_pages, help="1-based pages, e.g. 1-3,5")
    p.add_argument("--dpi", type=int, default=200, help="OCR render resolution (72-600)")
    p.add_argument("--borderless", action="store_true", help="also detect tables without ruling lines on OCR pages")
    p.add_argument("--native-strategy", choices=["lines", "lines_strict", "text"], default="lines")
    p.add_argument("--header-rows", type=int, default=1)
    p.add_argument("--no-merge", action="store_true", help="do not merge tables continued across pages")
    p.add_argument("--no-deskew", action="store_true")
    p.add_argument("--password-env", metavar="VAR", help="read the PDF password from this environment variable")
    p.add_argument("--max-pages", type=int, default=2000)
    p.add_argument("--max-size-mb", type=float, default=200.0)
    p.add_argument("-v", "--verbose", action="count", default=0)
    p.add_argument("-q", "--quiet", action="store_true")
    p.add_argument("--log-format", choices=["text", "json"], default="text")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def _setup_logging(args: argparse.Namespace) -> None:
    level = logging.ERROR if args.quiet else (logging.DEBUG if args.verbose > 1 else logging.INFO if args.verbose else logging.WARNING)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter() if args.log_format == "json" else logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger("pdf_table_extractor")
    root.handlers[:] = [handler]
    root.setLevel(level)
    root.propagate = False


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args)

    password = None
    if args.password_env:
        password = os.environ.get(args.password_env)
        if password is None:
            parser.error(f"environment variable {args.password_env} is not set")

    try:
        config = ExtractionConfig(
            mode=ExtractionMode(args.mode),
            ocr_dpi=args.dpi,
            detect_borderless=args.borderless,
            native_strategy=args.native_strategy,
            header_rows=args.header_rows,
            merge_continued_tables=not args.no_merge,
            deskew=not args.no_deskew,
            password=password,
            max_pages=args.max_pages,
            max_file_size_mb=args.max_size_mb,
        )
    except ValueError as exc:
        parser.error(str(exc))

    extractor = DocumentReader(config, enhance_images=not args.no_enhance)
    data_extractor = None
    if args.content == "all":
        from .image_data import ImageDataConfig, ImageDataExtractor

        data_extractor = ImageDataExtractor(
            ImageDataConfig(
                accuracy=args.accuracy,
                enhance=not args.no_enhance,
                password=password,
                max_pages=args.max_pages,
                max_file_size_mb=args.max_size_mb,
            )
        )
    exit_code = EXIT_OK
    inputs = list(iter_inputs(args.inputs, recursive=args.recursive))
    if not inputs:
        log.error("no PDF or image files found in the given inputs")
        return EXIT_INPUT
    many = len(inputs) > 1
    for path in inputs:
        if data_extractor is not None:
            exit_code = _worst(exit_code, _run_all_content(data_extractor, path, args, many))
            continue
        try:
            result = extractor.extract(path, pages=args.pages)
        except ExtractionError as exc:
            log.error("%s: %s", path, exc)
            exit_code = _worst(exit_code, EXIT_INPUT)
            continue
        except Exception:
            log.exception("%s: unexpected failure", path)
            exit_code = _worst(exit_code, EXIT_ERROR)
            continue

        if args.stdout:
            sys.stdout.write(to_json(result) + "\n")
        else:
            out_dir = args.out_dir / (path.stem + ("" if path.suffix.lower() == ".pdf" else path.suffix.replace(".", "_"))) if many else args.out_dir
            written = write_outputs(result, out_dir, tuple(args.format))
            if not args.quiet:
                print(f"{path}: {len(result.tables)} table(s), {len(written)} file(s) in {out_dir}", file=sys.stderr)
        if isinstance(result, ImageDocumentResult) and result.quality_level != "good" and not args.quiet:
            print(f"{path}: image quality is {result.quality_level}. Review these notes:", file=sys.stderr)
            for note in result.quality_notes():
                print(f"  - {note}", file=sys.stderr)
        if args.strict_quality and isinstance(result, ImageDocumentResult) and result.quality_level == "poor":
            exit_code = _worst(exit_code, EXIT_POOR_QUALITY)
        if not result.ok:
            log.warning("%s: pages failed: %s", path, result.failed_pages)
            exit_code = _worst(exit_code, EXIT_PARTIAL)
    return exit_code


def _run_all_content(extractor, path: Path, args: argparse.Namespace, many: bool) -> int:
    try:
        result = extractor.extract(path, pages=args.pages)
    except ExtractionError as exc:
        log.error("%s: %s", path, exc)
        return EXIT_INPUT
    except Exception:
        log.exception("%s: unexpected failure", path)
        return EXIT_ERROR
    if args.stdout:
        sys.stdout.write(result.to_json() + "\n")
    else:
        out_dir = args.out_dir / (path.stem + path.suffix.replace(".", "_")) if many else args.out_dir
        written = result.save(out_dir)
        if not args.quiet:
            print(
                f"{path}: {len(result.key_values)} field(s), {len(result.tables)} table(s), "
                f"{sum(len(p.blocks) for p in result.pages)} block(s); {len(written)} file(s) in {out_dir}",
                file=sys.stderr,
            )
    if result.quality_level != "good" and not args.quiet:
        print(f"{path}: quality is {result.quality_level}. Review these notes:", file=sys.stderr)
        for p in result.pages:
            for note in p.quality.get("notes", []):
                print(f"  - page {p.page}: {note}" if len(result.pages) > 1 else f"  - {note}", file=sys.stderr)
    code = EXIT_OK if result.ok else EXIT_PARTIAL
    if args.strict_quality and result.quality_level == "poor":
        code = _worst(code, EXIT_POOR_QUALITY)
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
