"""Table extraction from image files (PNG, JPEG, TIFF, BMP, WebP, GIF, JPEG 2000).

Reuses the OCR table machinery of the PDF path (grid detection, cell filling,
assembly) without modifying it. Adds what raw images need and PDFs do not:
EXIF/orientation handling, perspective correction, quality assessment,
adaptive enhancement and a retry for poor images.
"""

from __future__ import annotations

import hashlib
import io
import logging
import time
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from .assembly import assemble
from .config import ExtractionConfig
from .errors import InvalidPDFError, PDFTooLargeError
from .grid import binarize, detect_ruled_tables
from .image_enhance import correct_perspective, enhance
from .image_quality import QualityReport, assess
from .layout import PageData, RawTable, TextLine
from .models import DocumentResult, PageResult
from .normalize import clean_grid_indexed, clean_text, fix_ocr_text
from .ocr_engine import OCRBox, OCREngine
from .ocr_tables import (
    _band_of,
    _borderless_tables,
    _fill_region,
    _group_lines,
    _join_boxes,
    _low_cells,
    _table_confidence,
)

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".jpe", ".tif", ".tiff", ".bmp", ".dib", ".webp", ".gif", ".jp2", ".j2k", ".jpx", ".pbm", ".pgm", ".ppm"}
BODY_TEXT_PT = 5.5  # typical median glyph height in points (for unit scaling)
RETRY_BELOW_CONFIDENCE = 0.88
ORIENTATION_MAX_SIDE = 1600


class InvalidImageError(InvalidPDFError):
    """The input is not a readable image. Subclasses InvalidPDFError so
    existing ``except InvalidPDFError`` handlers keep working."""


@dataclass
class ImageDocumentResult(DocumentResult):
    """DocumentResult plus per-frame image quality notes."""

    input_type: str = "image"
    image_format: str | None = None
    quality: list[dict[str, Any]] = field(default_factory=list)

    @property
    def quality_level(self) -> str:
        levels = [q["level"] for q in self.quality]
        for lvl in ("poor", "fair"):
            if lvl in levels:
                return lvl
        return "good"

    def quality_notes(self) -> list[str]:
        out = []
        for q in self.quality:
            prefix = f"Page {q['page']}: " if len(self.quality) > 1 else ""
            out.extend(prefix + n for n in q["notes"])
        return out

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["input_type"] = self.input_type
        data["image_format"] = self.image_format
        data["quality_level"] = self.quality_level
        data["quality"] = self.quality
        data["bbox_unit"] = "pt (processed image pixels / scale)"
        return data


class _OrientationOCR(OCREngine):
    """Adds a cls-free recognition call used only for orientation voting."""

    def score_upright(self, img_bgr: np.ndarray) -> tuple[float, float]:
        """Return (length-weighted mean score, share of tall boxes)."""
        with self._lock:
            out = self._get()(img_bgr, use_det=True, use_cls=False, use_rec=True)
        txts = list(out.txts) if getattr(out, "txts", None) is not None else []
        scores = list(out.scores) if getattr(out, "scores", None) is not None else []
        boxes = getattr(out, "boxes", None)
        if not txts:
            return 0.0, 0.0
        weights = [max(len(t), 1) for t in txts]
        score = sum(s * wt for s, wt in zip(scores, weights)) / sum(weights)
        tall = 0
        if boxes is not None:
            for b in boxes:
                pts = np.asarray(b, dtype=float)
                bw, bh = np.ptp(pts[:, 0]), np.ptp(pts[:, 1])
                tall += bh > 1.5 * bw
        return float(score), tall / max(len(txts), 1)


_ROTATIONS = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def _rotate(img: np.ndarray, angle: int) -> np.ndarray:
    code = _ROTATIONS[angle]
    return img if code is None else cv2.rotate(img, code)


def _downscale(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    f = min(1.0, max_side / max(h, w))
    return cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA) if f < 1 else img


def detect_orientation(img: np.ndarray, engine: _OrientationOCR) -> int:
    """Clockwise rotation (0/90/180/270) that makes the text upright."""
    small = _downscale(img, ORIENTATION_MAX_SIDE)
    score0, tall = engine.score_upright(small)
    candidates = [90, 270] if tall > 0.5 else [180]
    if tall <= 0.5 and score0 >= 0.85:
        return 0
    best_angle, best = 0, score0 if tall <= 0.5 else -1.0
    for angle in candidates:
        s, t = engine.score_upright(_rotate(small, angle))
        if t <= 0.5 and s > best + 0.05:
            best_angle, best = angle, s
    return best_angle


class LazyFrames(Sequence):
    """Frames of an image file, decoded one at a time on access.

    Security: frame count and every frame's pixel size are validated from the
    headers before anything is decoded, and only requested frames are decoded,
    so a multi-page TIFF/GIF cannot exhaust memory. Pillow's global settings
    are never modified (thread-safe under concurrent API requests).
    """

    def __init__(self, data: bytes, max_frames: int, max_pixels: int):
        try:
            with warnings.catch_warnings(record=True):  # local, no global filter change
                self._im = Image.open(io.BytesIO(data))
                self.format = self._im.format or "unknown"
                self.dpi_info = self._im.info.get("dpi")
                n = getattr(self._im, "n_frames", 1)
                if n > max_frames:
                    raise PDFTooLargeError(f"image has {n} frames; limit is {max_frames}")
                for i in range(n):
                    self._im.seek(i)
                    w, h = self._im.size
                    if w * h > max_pixels:
                        raise PDFTooLargeError(f"frame {i + 1} is {w}x{h} pixels; limit is {max_pixels}")
                self._n = n
        except PDFTooLargeError:
            raise
        except Image.DecompressionBombError as exc:
            raise PDFTooLargeError(f"image too large: {exc}") from exc
        except Exception as exc:
            raise InvalidImageError(f"cannot decode image: {exc}") from exc

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int) -> np.ndarray:
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(self._n))]
        if not -self._n <= i < self._n:
            raise IndexError(i)
        try:
            with warnings.catch_warnings(record=True):
                self._im.seek(i % self._n)
                frame = ImageOps.exif_transpose(self._im.copy())  # phone photos
                if frame.mode in ("RGBA", "LA", "P", "PA"):
                    rgba = frame.convert("RGBA")
                    frame = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
                return cv2.cvtColor(np.asarray(frame.convert("RGB")), cv2.COLOR_RGB2BGR)
        except Image.DecompressionBombError as exc:
            raise PDFTooLargeError(f"image too large: {exc}") from exc
        except Exception as exc:
            raise InvalidImageError(f"cannot decode image frame {i + 1}: {exc}") from exc

    def dpi(self) -> float | None:
        try:
            value = float(self.dpi_info[0] if isinstance(self.dpi_info, (tuple, list)) else self.dpi_info)
        except (TypeError, ValueError):
            return None
        return value if 100 <= value <= 1200 else None  # 72/96 are placeholders


def load_frames(data: bytes, max_frames: int, max_pixels: int) -> tuple[LazyFrames, str, float | None]:
    """Validate an image and return (lazily decoded frames, format, dpi or None)."""
    frames = LazyFrames(data, max_frames, max_pixels)
    frames[0]  # fail fast on undecodable data
    return frames, frames.format, frames.dpi()


MAX_GRID_CELLS = 2000


def _plausible_regions(regions, shape):
    """Drop frames that are not tables: a border around the whole image, a
    region enclosing other tables, or absurdly dense grids (photo artefacts)."""
    h, w = shape[:2]

    def encloses(a, b):
        return a.bbox[0] <= b.bbox[0] and a.bbox[1] <= b.bbox[1] and a.bbox[2] >= b.bbox[2] and a.bbox[3] >= b.bbox[3]

    kept = []
    for r in regions:
        x0, y0, x1, y1 = r.bbox
        if (x1 - x0) * (y1 - y0) > 0.9 * w * h:
            log.debug("ignoring page-sized frame %s", r.bbox)
            continue
        if any(o is not r and encloses(r, o) for o in regions):
            log.debug("ignoring frame enclosing other tables %s", r.bbox)
            continue
        if r.n_rows * r.n_cols > MAX_GRID_CELLS:
            log.warning("ignoring implausible %dx%d grid", r.n_rows, r.n_cols)
            continue
        kept.append(r)
    return kept


def extract_from_array(
    img: np.ndarray, page_no: int, scale: float, cfg: ExtractionConfig, engine: OCREngine
) -> tuple[PageData, float]:
    """OCR table extraction on an already prepared BGR image.

    ``scale`` is pixels per point; it keeps geometric thresholds consistent
    with the PDF path. Returns the page and the mean OCR score of all text.
    """
    ink = binarize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    boxes = engine.recognize(img, use_det=True)

    def to_pt(b):
        return tuple(v / scale for v in b)

    tables: list[RawTable] = []
    used: set[int] = set()
    for region in _plausible_regions(detect_ruled_tables(img, scale, min_rows=1, min_cols=cfg.min_cols), img.shape):
        rx0, ry0, rx1, ry1 = region.bbox
        inside = [i for i, b in enumerate(boxes) if rx0 <= b.cx <= rx1 and ry0 <= b.cy <= ry1]
        texts, scores, warns = _fill_region(region, [boxes[i] for i in inside], img, ink, engine)
        used.update(inside)
        grid, row_keep, col_keep = clean_grid_indexed(texts)
        if len(grid) < cfg.min_rows or not grid or len(grid[0]) < cfg.min_cols:
            continue
        tables.append(
            RawTable(
                page=page_no,
                bbox=to_pt(region.bbox),
                grid=grid,
                method="ocr-ruled",
                confidence=_table_confidence(scores),
                low_confidence_cells=_low_cells(scores, row_keep, col_keep, cfg.review_score_threshold),
                warnings=warns,
            )
        )

    if cfg.detect_borderless:
        rest = [b for i, b in enumerate(boxes) if i not in used]
        for bbox, run, bands in _borderless_tables(rest, cfg):
            cells: list[list[list[OCRBox]]] = [[[] for _ in bands] for _ in run]
            for r, line in enumerate(run):
                for b in line:
                    cells[r][_band_of(bands, b.cx)].append(b)
            texts = [[fix_ocr_text(clean_text(_join_boxes(c))) for c in row] for row in cells]
            scores = [[min((b.score for b in c), default=None) for c in row] for row in cells]
            grid, row_keep, col_keep = clean_grid_indexed(texts)
            if len(grid) < cfg.min_rows or len(grid[0]) < cfg.min_cols:
                continue
            ids = {id(b) for line in run for b in line}
            used.update(i for i, b in enumerate(boxes) if id(b) in ids)
            tables.append(
                RawTable(
                    page=page_no,
                    bbox=to_pt(bbox),
                    grid=grid,
                    method="ocr-borderless",
                    confidence=_table_confidence(scores) * 0.9,
                    low_confidence_cells=_low_cells(scores, row_keep, col_keep, cfg.review_score_threshold),
                    warnings=["borderless table: column layout inferred from text alignment"],
                )
            )

    tables.sort(key=lambda t: (t.bbox[1], t.bbox[0]))
    remaining = [b for i, b in enumerate(boxes) if i not in used]
    lines = [
        TextLine(
            bbox=to_pt((min(b.bbox[0] for b in ln), min(b.bbox[1] for b in ln), max(b.bbox[2] for b in ln), max(b.bbox[3] for b in ln))),
            text=clean_text(" ".join(b.text for b in ln)),
        )
        for ln in _group_lines(remaining)
    ]
    page = PageData(page=page_no, width=img.shape[1] / scale, height=img.shape[0] / scale, tables=tables, lines=lines)
    mean = sum(b.score for b in boxes) / len(boxes) if boxes else 0.0
    return page, mean


class ImageTableExtractor:
    """Extract tables from image files, with quality notes for unclear inputs."""

    def __init__(
        self,
        config: ExtractionConfig | None = None,
        ocr_engine: OCREngine | None = None,
        enhance_images: bool = True,
        max_pixels: int = 100_000_000,
    ):
        self.config = config or ExtractionConfig()
        self.enhance_images = enhance_images
        self.max_pixels = max_pixels
        self._ocr = ocr_engine or _OrientationOCR(min_score=self.config.ocr_min_score, max_side_len=self.config.ocr_max_side_len)

    def _process_frame(self, img: np.ndarray, page_no: int, dpi: float | None) -> tuple[PageData, QualityReport]:
        cfg = replace(self.config, deskew=False)  # deskew happens in enhance()
        steps: list[str] = []

        if self.enhance_images:
            img, flattened = correct_perspective(img)
            if flattened:
                steps.append("perspective corrected (photographed page)")
        if isinstance(self._ocr, _OrientationOCR):
            angle = detect_orientation(img, self._ocr)
            if angle:
                img = _rotate(img, angle)
                steps.append(f"rotated {angle} degrees clockwise (page was sideways/upside down)")

        quality = assess(img)
        if any(st.startswith("perspective") for st in steps):
            quality.issues.append("photographed")
            quality.notes.append("Page appears to be photographed at an angle; perspective was corrected. Check edge columns for distortion.")
        if any(st.startswith("rotated") for st in steps):
            quality.issues.append("rotated")
            quality.notes.append(f"Page was not upright; it was {steps[-1].split(' (')[0]}.")
        if not self.enhance_images:
            scale = dpi / 72.0 if dpi else self._scale(quality, 1.0)
            page, quality.ocr_confidence = extract_from_array(img, page_no, scale, cfg, self._ocr)
            quality.enhancements = steps
            return page, quality

        def content(p: PageData) -> int:
            return sum(1 for t in p.tables for row in t.grid for cell in row if cell)

        best: tuple[PageData, float, list[str]] | None = None
        for aggressive in (False, True):
            enh = enhance(img, quality, aggressive=aggressive, do_deskew=self.config.deskew)
            scale = (dpi / 72.0 * enh.upscale) if dpi else self._scale(quality, enh.upscale)
            page, conf = extract_from_array(enh.image, page_no, scale, cfg, self._ocr)
            log.info(
                "page %d: %s pass, OCR confidence %.3f, %d table(s)",
                page_no, "aggressive" if aggressive else "standard", conf, len(page.tables),
            )
            label = ["aggressive enhancement pass (image was hard to read)"] if aggressive else []
            if best is None or (content(page), conf) > (content(best[0]) + 2, best[1] + 0.01) or (
                content(page) > content(best[0])
            ):
                best = (page, conf, enh.steps + label)
            # Stop when the first pass is convincing: confident, and either the
            # image is clean or tables were found.
            if conf >= RETRY_BELOW_CONFIDENCE and (page.tables or not quality.issues):
                break
        assert best is not None
        page, conf, enh_steps = best
        quality.ocr_confidence = conf
        quality.enhancements = steps + enh_steps
        if conf < RETRY_BELOW_CONFIDENCE:
            quality.issues.append("low_ocr_confidence")
            quality.notes.append(
                f"OCR confidence is low ({conf:.2f}) even after enhancement; manually verify the extracted values, "
                "or supply a sharper scan (300 DPI, good lighting, flat page)."
            )
        if not page.tables and quality.issues:
            quality.issues.append("no_tables_found")
            quality.notes.append(
                "No table could be recovered from this degraded image. If it contains a table, rescan or re-photograph it "
                "(at least 300 DPI, in focus, even lighting)."
            )
        return page, quality

    @staticmethod
    def _scale(quality: QualityReport, upscale: float) -> float:
        if quality.char_height_px:
            return max(1.0, min(quality.char_height_px * upscale / BODY_TEXT_PT, 12.0))
        return 200 / 72 * upscale

    def extract(self, source: str | Path | bytes, frames: list[int] | None = None) -> ImageDocumentResult:
        started = time.perf_counter()
        cfg = self.config
        limit = int(cfg.max_file_size_mb * 1024 * 1024)
        if isinstance(source, (bytes, bytearray)):
            data, name = bytes(source), "<bytes>"
        else:
            path = Path(source)
            if not path.is_file():
                raise InvalidImageError(f"file not found: {path}")
            if path.stat().st_size > limit:
                raise PDFTooLargeError(f"{path} exceeds {cfg.max_file_size_mb} MB limit")
            data, name = path.read_bytes(), str(path)
        if len(data) > limit:
            raise PDFTooLargeError(f"input exceeds {cfg.max_file_size_mb} MB limit")

        images, fmt, dpi = load_frames(data, cfg.max_pages, max_pixels=self.max_pixels)
        selected = sorted(set(frames)) if frames else list(range(1, len(images) + 1))
        bad = [p for p in selected if not 1 <= p <= len(images)]
        if bad:
            raise InvalidImageError(f"page(s) out of range 1..{len(images)}: {bad}")

        pages: list[PageData] = []
        results: list[PageResult] = []
        quality: list[dict[str, Any]] = []
        for page_no in selected:
            t0 = time.perf_counter()
            res = PageResult(page=page_no, method="error")
            try:
                pdata, report = self._process_frame(images[page_no - 1], page_no, dpi)
                pages.append(pdata)
                res.method = "ocr"
                res.n_tables = len(pdata.tables)
                res.warnings = pdata.warnings
                quality.append({"page": page_no, **report.to_dict()})
                if report.level == "poor":
                    log.warning("page %d: poor image quality (%s); see quality notes", page_no, ", ".join(report.issues))
            except Exception as exc:  # isolate frame failures
                log.exception("page %d failed", page_no)
                res.error = f"{type(exc).__name__}: {exc}"
            res.duration_ms = (time.perf_counter() - t0) * 1000
            results.append(res)

        tables = assemble(pages, cfg)
        for t in tables:
            q = next((q for q in quality if q["page"] == t.page), None)
            if q and q["level"] != "good":
                t.warnings.append(f"source image quality is {q['level']}: " + ", ".join(q["issues"]))

        return ImageDocumentResult(
            source=name,
            sha256=hashlib.sha256(data).hexdigest(),
            n_pages=len(images),
            mode="ocr",
            tables=tables,
            pages=results,
            duration_ms=(time.perf_counter() - started) * 1000,
            image_format=fmt,
            quality=quality,
        )
