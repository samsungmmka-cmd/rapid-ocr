"""ImageDataExtractor: extract *all* data from document images with RapidOCR.

Output per page, in reading order: headings, paragraphs, list items,
key-value pairs, tables and uncertain fragments (signatures, stamps), plus
every recognised line with its confidence, alternatives and coordinates on
the ORIGINAL image, and quality notes for unclear inputs.

Typical use::

    from pdf_table_extractor import ImageDataExtractor

    extractor = ImageDataExtractor()                 # reuse: models load once
    result = extractor.extract("invoice.jpg")        # image, scanned PDF, or bytes
    print(result.text)                               # full text, reading order
    for kv in result.key_values: print(kv.key, kv.value, kv.number)
    for table in result.tables: print(table.records())
    result.save("output/")                           # JSON + Markdown + text + CSV
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np

from .errors import EncryptedPDFError, InvalidPDFError, PDFTooLargeError
from .exporters import _atomic_write, slugify, table_to_csv, table_to_markdown
from .image_enhance import correct_perspective_matrix, enhance
from .image_layout import Block, Item, KeyValue, inline_kv, pair_kvs, text_blocks, union_bbox, xy_cut
from .image_quality import QualityReport, assess
from .image_reader import _rotate, load_frames
from .image_tables import (
    accept_ruled,
    drop_noise_rows,
    extract_footer_kv,
    fill_grid,
    find_borderless_tables,
    find_ruled_grids,
)
from .models import Table
from .normalize import clean_text, fix_ocr_text
from .ocr_models import OCRLine, OCRModels, has_cjk

log = logging.getLogger(__name__)

SCHEMA_VERSION = "image-data-1.0"
ACCURACY_PROFILES = {
    # reread lines below this confidence; thorough adds a 2x-upscaled re-read
    "fast": {"reread_below": None, "medium": False, "english": False, "thorough": False},
    "balanced": {"reread_below": 0.93, "medium": True, "english": True, "thorough": False},
    "best": {"reread_below": 0.985, "medium": True, "english": True, "thorough": True},
}


@dataclass(frozen=True)
class ImageDataConfig:
    accuracy: str = "balanced"  # fast | balanced | best
    enhance: bool = True
    detect_tables: bool = True
    detect_orientation: bool = True
    max_rereads: int = 250
    review_threshold: float = 0.80  # lines below are listed for human review
    pdf_dpi: int = 300  # scanned PDFs are rendered at this resolution
    password: str | None = field(default=None, repr=False)
    max_pages: int = 500
    max_file_size_mb: float = 200.0
    max_pixels: int = 100_000_000

    def __post_init__(self) -> None:
        if self.accuracy not in ACCURACY_PROFILES:
            raise ValueError(f"accuracy must be one of {sorted(ACCURACY_PROFILES)}")
        if not 72 <= self.pdf_dpi <= 600:
            raise ValueError("pdf_dpi must be between 72 and 600")
        if not 0 <= self.review_threshold <= 1:
            raise ValueError("review_threshold must be between 0 and 1")


# ------------------------------------------------------------------ results
@dataclass
class PageData:
    page: int
    width: int
    height: int
    blocks: list[Block]
    key_values: list[KeyValue]
    tables: list[Table]
    lines: list[OCRLine]
    quality: dict[str, Any]
    duration_ms: float = 0.0
    error: str | None = None

    @property
    def text(self) -> str:
        return render_text(self)

    def review(self, threshold: float) -> list[OCRLine]:
        return [ln for ln in self.lines if ln.score < threshold]


@dataclass
class ImageDataResult:
    source: str
    sha256: str
    input_type: str  # image | pdf
    file_format: str
    pages: list[PageData]
    accuracy: str
    models_unavailable: dict[str, str]
    review_threshold: float
    duration_ms: float = 0.0

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text)

    @property
    def key_values(self) -> list[KeyValue]:
        return [kv for p in self.pages for kv in p.key_values]

    @property
    def tables(self) -> list[Table]:
        return [t for p in self.pages for t in p.tables]

    @property
    def failed_pages(self) -> list[int]:
        return [p.page for p in self.pages if p.error]

    @property
    def ok(self) -> bool:
        return not self.failed_pages

    @property
    def quality_level(self) -> str:
        levels = [p.quality.get("level", "good") for p in self.pages if p.quality]
        return "poor" if "poor" in levels else "fair" if "fair" in levels else "good"

    def get(self, key: str, default: str | None = None) -> str | None:
        """Value of the first key-value pair whose key matches (case-insensitive,
        ignoring punctuation), e.g. result.get("invoice date")."""
        want = _key_norm(key)
        for kv in self.key_values:
            if _key_norm(kv.key) == want:
                return kv.value
        return default

    def to_dict(self) -> dict[str, Any]:
        tables = self.tables
        return {
            "schema_version": SCHEMA_VERSION,
            "source": self.source,
            "sha256": self.sha256,
            "input_type": self.input_type,
            "file_format": self.file_format,
            "n_pages": len(self.pages),
            "status": "ok" if self.ok else "partial",
            "failed_pages": self.failed_pages,
            "accuracy": self.accuracy,
            "models_unavailable": self.models_unavailable,
            "quality_level": self.quality_level,
            "coordinates": "pixels of the original image (PDF: rendered page)",
            "duration_ms": round(self.duration_ms, 1),
            "text": self.text,
            "key_values": [_kv_dict(kv, p.page) for p in self.pages for kv in p.key_values],
            "tables": [t.to_dict() for t in tables],
            "pages": [_page_dict(p, tables, self.review_threshold) for p in self.pages],
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_markdown(self) -> str:
        return render_markdown(self)

    def save(self, out_dir: str | Path) -> list[Path]:
        """Write <name>.data.json, <name>.data.md, <name>.txt and one CSV per table."""
        out = Path(out_dir)
        stem = slugify(Path(self.source).stem) or "document"
        written = []
        for name, content in (
            (f"{stem}.data.json", self.to_json() + "\n"),
            (f"{stem}.data.md", self.to_markdown()),
            (f"{stem}.txt", self.text + "\n"),
        ):
            _atomic_write(out / name, content)
            written.append(out / name)
        for t in self.tables:
            path = out / f"{stem}_table{t.index + 1:02d}.csv"
            _atomic_write(path, table_to_csv(t))
            written.append(path)
        return written


def _key_norm(key: str) -> str:
    return "".join(ch for ch in key.lower() if ch.isalnum())


def _r(b) -> list[float]:
    return [round(float(v), 1) for v in b]


def _kv_dict(kv: KeyValue, page: int) -> dict[str, Any]:
    return {
        "page": page,
        "key": kv.key,
        "value": kv.value,
        "number": kv.number,
        "confidence": round(kv.confidence, 4),
        "source": kv.source,
        "bbox": _r(kv.bbox),
    }


def _page_dict(p: PageData, all_tables: list[Table], review_threshold: float) -> dict[str, Any]:
    return {
        "page": p.page,
        "width": p.width,
        "height": p.height,
        "error": p.error,
        "duration_ms": round(p.duration_ms, 1),
        "quality": p.quality,
        "text": p.text,
        "blocks": [
            {
                "type": b.type,
                "text": b.text,
                "level": b.level,
                "confidence": round(b.confidence, 4),
                "bbox": _r(b.bbox),
                **({"table_index": all_tables.index(p.tables[b.ref])} if b.type == "table" else {}),
                **({"key_value_index": b.ref} if b.type == "key_value" else {}),
            }
            for b in p.blocks
        ],
        "lines": [
            {
                "id": ln.id,
                "text": ln.text,
                "confidence": round(ln.score, 4),
                "model": ln.model,
                "bbox": _r(ln.bbox),
                "polygon": [_r(pt) for pt in ln.poly],
                "alternatives": [{"text": c.text, "confidence": round(c.score, 4), "model": c.model} for c in ln.alternatives],
            }
            for ln in p.lines
        ],
        "review": [
            {"line_id": ln.id, "text": ln.text, "confidence": round(ln.score, 4), "alternatives": [c.text for c in ln.alternatives]}
            for ln in p.review(review_threshold)
        ],
    }


# ---------------------------------------------------------------- rendering
def render_text(page: PageData) -> str:
    out: list[str] = []
    for b in page.blocks:
        if b.type == "table":
            t = page.tables[b.ref]
            rows = ([t.header] if t.header else []) + t.rows
            out.append("\n".join("\t".join(r) for r in rows))
        elif b.type == "key_value":
            kv = page.key_values[b.ref]
            out.append(f"{kv.key}: {kv.value}")
        elif b.type == "uncertain":
            out.append(f"[uncertain: {b.text.replace(chr(10), ' ')}]")
        else:
            out.append(b.text)
    return "\n\n".join(out)


def render_markdown(result: ImageDataResult) -> str:
    md = [f"# Data extracted from {Path(result.source).name}", ""]
    md.append(
        f"{len(result.pages)} page(s) · {len(result.key_values)} field(s) · {len(result.tables)} table(s) · "
        f"quality: {result.quality_level} · accuracy profile: {result.accuracy}"
    )
    for p in result.pages:
        if len(result.pages) > 1:
            md += ["", f"## Page {p.page}"]
        notes = p.quality.get("notes", []) if p.quality else []
        if notes:
            md += ["", "> **Quality notes**"] + [f"> - {n}" for n in notes]
        if p.error:
            md += ["", f"> **Error:** {p.error}"]
        if p.key_values:
            md += ["", "### Fields", "", "| Field | Value | Confidence |", "|---|---|---|"]
            md += [f"| {_md(kv.key)} | {_md(kv.value)} | {kv.confidence:.2f} |" for kv in p.key_values]
        md += ["", "### Content (reading order)", ""]
        for b in p.blocks:
            if b.type == "heading":
                md += ["#" * (3 + (b.level or 2) - 1) + " " + b.text.replace("\n", " "), ""]
            elif b.type == "paragraph":
                md += ["  \n".join(b.text.split("\n")), ""]
            elif b.type == "list_item":
                md += [f"- {b.text}", ""]
            elif b.type == "key_value":
                kv = p.key_values[b.ref]
                md += [f"**{_md(kv.key)}:** {_md(kv.value)}", ""]
            elif b.type == "table":
                t = p.tables[b.ref]
                md += [f"*Table {t.index + 1}{': ' + t.title if t.title else ''}* ({t.method}, confidence {t.confidence:.2f})", ""]
                md += [table_to_markdown(t), ""]
                md += [f"> {w}" for w in t.warnings] + ([""] if t.warnings else [])
            elif b.type == "uncertain":
                md += [f"> *Uncertain (signature, stamp or illegible):* {b.text.replace(chr(10), ' ')}", ""]
        review = p.review(result.review_threshold)
        if review:
            md += ["### Needs review", "", "| Text read | Confidence | Alternatives |", "|---|---|---|"]
            md += [f"| {_md(ln.text)} | {ln.score:.2f} | {_md(', '.join(c.text for c in ln.alternatives))} |" for ln in review]
            md.append("")
    return "\n".join(md).rstrip() + "\n"


def _md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


# ---------------------------------------------------------------- geometry
def _rot_matrix(angle: int, w: int, h: int) -> np.ndarray:
    """Matrix of cv2.rotate for an image of size (w, h) (clockwise angle)."""
    if angle == 90:
        return np.array([[0, -1, h - 1], [1, 0, 0], [0, 0, 1]], dtype=float)
    if angle == 180:
        return np.array([[-1, 0, w - 1], [0, -1, h - 1], [0, 0, 1]], dtype=float)
    if angle == 270:
        return np.array([[0, 1, 0], [-1, 0, w - 1], [0, 0, 1]], dtype=float)
    return np.eye(3)


def _apply(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    hom = np.hstack([pts, np.ones((len(pts), 1))]) @ m.T
    return hom[:, :2] / hom[:, 2:3]


def _map_bbox(m: np.ndarray, b) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = b
    pts = _apply(m, np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
    return (float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max()))


# ------------------------------------------------------------- PDF pages
def render_pdf_page(page, dpi: int, max_pixels: int) -> np.ndarray:
    """Render one PDF page to BGR, capping the pixel count (render-bomb guard:
    a page may be up to 200x200 inches)."""
    import pymupdf

    scale = dpi / 72
    area = max(page.rect.width * page.rect.height, 1.0)
    if area * scale * scale > max_pixels:
        scale = (max_pixels / area) ** 0.5
        log.warning("page %d rendered at %.0f dpi (pixel cap)", page.number + 1, scale * 72)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csRGB, alpha=False)
    arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.stride)[:, : pix.width * 3]
    return cv2.cvtColor(arr.reshape(pix.height, pix.width, 3), cv2.COLOR_RGB2BGR)


class PdfFrames:
    """PDF pages rendered one at a time on access (bounded memory)."""

    def __init__(self, data: bytes, name: str, cfg: ImageDataConfig):
        import pymupdf

        try:
            self._doc = pymupdf.open(stream=data, filetype="pdf")
        except Exception as exc:
            raise InvalidPDFError(f"cannot parse {name}: {exc}") from exc
        if self._doc.needs_pass and not (cfg.password and self._doc.authenticate(cfg.password)):
            self._doc.close()
            raise EncryptedPDFError(f"{name} is encrypted; supply the correct password")
        if self._doc.page_count > cfg.max_pages:
            n = self._doc.page_count
            self._doc.close()
            raise PDFTooLargeError(f"{name} has {n} pages; limit is {cfg.max_pages}")
        self._cfg = cfg

    def __len__(self) -> int:
        return self._doc.page_count

    def __getitem__(self, i: int) -> np.ndarray:
        return render_pdf_page(self._doc[i], self._cfg.pdf_dpi, self._cfg.max_pixels)

    def close(self) -> None:
        self._doc.close()


# ---------------------------------------------------------------- extractor
class ImageDataExtractor:
    """Extract all text, fields and tables from document images using RapidOCR."""

    def __init__(self, config: ImageDataConfig | None = None, models: OCRModels | None = None):
        self.config = config or ImageDataConfig()
        profile = ACCURACY_PROFILES[self.config.accuracy]
        self.profile = profile
        self.models = models or OCRModels(use_medium=profile["medium"], use_english=profile["english"])

    # ------------------------------------------------------------- input
    def _frames(self, data: bytes, name: str) -> tuple[list[np.ndarray], str, str]:
        from .reader import detect_type

        kind = detect_type(data[:1024])
        if kind == "pdf":
            return PdfFrames(data, name, self.config), "pdf", "PDF"
        frames, fmt, _dpi = load_frames(data, self.config.max_pages, self.config.max_pixels)
        return frames, "image", fmt

    def extract(self, source: str | Path | bytes, pages: list[int] | None = None) -> ImageDataResult:
        started = time.perf_counter()
        limit = int(self.config.max_file_size_mb * 1024 * 1024)
        if isinstance(source, (bytes, bytearray)):
            data, name = bytes(source), "<bytes>"
        else:
            path = Path(source)
            if not path.is_file():
                raise InvalidPDFError(f"file not found: {path}")
            if path.stat().st_size > limit:
                raise PDFTooLargeError(f"{path} exceeds {self.config.max_file_size_mb} MB limit")
            data, name = path.read_bytes(), str(path)
        if len(data) > limit:
            raise PDFTooLargeError(f"input exceeds {self.config.max_file_size_mb} MB limit")

        frames, kind, fmt = self._frames(data, name)
        selected = sorted(set(pages)) if pages else list(range(1, len(frames) + 1))
        bad = [p for p in selected if not 1 <= p <= len(frames)]
        if bad:
            raise InvalidPDFError(f"page(s) out of range 1..{len(frames)}: {bad}")

        results: list[PageData] = []
        try:
            for page_no in selected:
                t0 = time.perf_counter()
                try:
                    img = frames[page_no - 1]
                    page = self.extract_page(img, page_no)
                except Exception as exc:  # isolate page failures
                    log.exception("page %d failed", page_no)
                    page = PageData(page_no, 0, 0, [], [], [], [], {}, error=f"{type(exc).__name__}: {exc}")
                page.duration_ms = (time.perf_counter() - t0) * 1000
                log.info("page %d: %d blocks, %d fields, %d tables, %.0f ms", page_no, len(page.blocks), len(page.key_values), len(page.tables), page.duration_ms)
                results.append(page)
        finally:
            if hasattr(frames, "close"):
                frames.close()

        index = 0
        for p in results:  # document-wide table numbering
            for t in p.tables:
                t.index = index
                index += 1
        return ImageDataResult(
            source=name,
            sha256=hashlib.sha256(data).hexdigest(),
            input_type=kind,
            file_format=fmt,
            pages=results,
            accuracy=self.config.accuracy,
            models_unavailable=self.models.unavailable,
            review_threshold=self.config.review_threshold,
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    # --------------------------------------------------------------- page
    def extract_page(self, original: np.ndarray, page_no: int = 1) -> PageData:
        cfg = self.config
        img = original
        matrix = np.eye(3)  # original -> processed
        steps: list[str] = []

        if cfg.enhance:
            flat, m = correct_perspective_matrix(img)
            if m is not None:
                img, matrix = flat, m @ matrix
                steps.append("perspective corrected (photographed page)")
        if cfg.detect_orientation:
            from .image_reader import detect_orientation

            angle = detect_orientation(img, self.models)
            if angle:
                h, w = img.shape[:2]
                matrix = _rot_matrix(angle, w, h) @ matrix
                img = _rotate(img, angle)
                steps.append(f"rotated {angle} degrees clockwise")

        quality = assess(img)
        if any(s.startswith("perspective") for s in steps):
            quality.issues.append("photographed")
            quality.notes.append("Page appears to be photographed at an angle; perspective was corrected.")
        if any(s.startswith("rotated") for s in steps):
            quality.issues.append("rotated")
            quality.notes.append(f"Page was not upright; it was {steps[-1]}.")

        work, lines, enh_steps, enh_matrix, upscale = self._read(img, quality)
        matrix = enh_matrix @ matrix
        char_h = (quality.char_height_px or 12.0) * upscale

        latin = _is_latin(lines)
        self._refine(work, lines, latin)
        for ln in lines:
            ln.text = fix_ocr_text(clean_text(ln.text))
        lines = [ln for ln in lines if ln.text]

        blocks, kvs, tables = self._layout(work, lines, char_h)

        # --- map everything back to original-image coordinates
        inv = np.linalg.inv(matrix)
        for ln in lines:
            ln.poly = _apply(inv, ln.poly)
            for i, (w_text, w_poly) in enumerate(ln.words):
                ln.words[i] = (w_text, _apply(inv, w_poly))
        for b in blocks:
            b.bbox = _map_bbox(inv, b.bbox)
        for kv in kvs:
            kv.bbox = _map_bbox(inv, kv.bbox)
        for t in tables:
            t.bbox = _map_bbox(inv, t.bbox)

        for t in tables:
            t.page = t.pages[0] = page_no
        quality.enhancements = steps + enh_steps
        quality.ocr_confidence = float(np.mean([ln.score for ln in lines])) if lines else 0.0
        self._quality_notes(quality, lines)
        return PageData(
            page=page_no,
            width=int(original.shape[1]),
            height=int(original.shape[0]),
            blocks=blocks,
            key_values=kvs,
            tables=tables,
            lines=sorted(lines, key=lambda ln: ln.id),
            quality=quality.to_dict(),
        )

    def _read(self, img: np.ndarray, quality: QualityReport):
        """OCR the (enhanced) page; retry with aggressive enhancement if weak."""
        if not self.config.enhance:
            return img, self.models.read_page(img), [], np.eye(3), 1.0

        def content(lines: list[OCRLine]) -> float:
            return sum(ln.score * len(ln.text) for ln in lines)

        best = None
        for aggressive in (False, True):
            enh = enhance(img, quality, aggressive=aggressive)
            lines = self.models.read_page(enh.image)
            mean = float(np.mean([ln.score for ln in lines])) if lines else 0.0
            steps = enh.steps + (["aggressive enhancement pass"] if aggressive else [])
            if best is None or content(lines) > content(best[1]) * 1.03:
                best = (enh.image, lines, steps, enh.matrix, enh.upscale)
            if mean >= 0.85:
                break
        return best

    def _refine(self, img: np.ndarray, lines: list[OCRLine], latin: bool) -> None:
        threshold = self.profile["reread_below"]
        if threshold is None:
            return
        todo = [ln for ln in lines if ln.score < threshold or (latin and has_cjk(ln.text))]
        todo.sort(key=lambda ln: ln.score)
        if len(todo) > self.config.max_rereads:
            log.warning("re-reading only the %d weakest of %d uncertain lines", self.config.max_rereads, len(todo))
            todo = todo[: self.config.max_rereads]
        for ln in todo:
            self.models.refine(img, ln, latin_document=latin, thorough=self.profile["thorough"])

    def _layout(
        self, img: np.ndarray, lines: list[OCRLine], char_h: float, allow_reread: bool = True
    ) -> tuple[list[Block], list[KeyValue], list[Table]]:
        """``allow_reread=False`` for exact (native PDF) text: never replace it with OCR."""
        page_w = img.shape[1]
        lh = median(ln.height for ln in lines) if lines else char_h * 1.6
        taken: set[int] = set()
        table_cells = []

        if self.config.detect_tables:
            for grid in find_ruled_grids(img, char_h):
                tc = fill_grid(grid, [ln for ln in lines if ln.id not in taken], img, self.models, char_h)
                noise = drop_noise_rows(tc)
                if not accept_ruled(tc):
                    continue
                extract_footer_kv(tc)
                tc.line_ids -= {ln.id for ln in noise}
                table_cells.append(tc)
                taken |= tc.line_ids
            table_cells += find_borderless_tables([ln for ln in lines if ln.id not in taken], char_h)
            for tc in table_cells:
                taken |= tc.line_ids

        # Key-value pairs from table footers, inline "Key: value", and side-by-side pairs.
        kvs: list[KeyValue] = []
        for tc in table_cells:
            for label, value in tc.footer_kv:
                kvs.append(KeyValue(label.text.rstrip(":"), value.text, min(label.score, value.score), union_bbox([label.bbox, value.bbox]), "table_footer", [label.id, value.id]))
        free = [ln for ln in lines if ln.id not in taken]
        pair_list, paired = pair_kvs(free, page_w)
        kvs += pair_list
        for ln in free:
            if ln.id in paired:
                continue
            kv = inline_kv(ln)
            if kv:
                kvs.append(kv)
                paired.add(ln.id)

        for tc in table_cells if allow_reread else []:
            self._fix_numeric_columns(img, tc)

        # Tables as public models.
        tables: list[Table] = []
        for tc in table_cells:
            texts = [[fix_ocr_text(clean_text(t)) for t in row] for row in tc.texts()]
            header = texts[0] if tc.has_header and texts else []
            body = texts[1:] if tc.has_header else texts
            scores = [[min((ln.score for ln in c), default=None) for c in row] for row in tc.cells]
            vals = [s for r in scores for s in r if s is not None]
            low = [
                {"section": "header" if tc.has_header and r == 0 else "body", "row": r - (1 if tc.has_header and r > 0 else 0), "col": c, "score": s}
                for r, row in enumerate(scores)
                for c, s in enumerate(row)
                if s is not None and s < self.config.review_threshold
            ]
            tables.append(
                Table(page=0, index=len(tables), bbox=tc.bbox, method=tc.method, header=header, rows=body,
                      confidence=float(np.mean(vals)) if vals else 0.0, low_confidence_cells=low, warnings=tc.warnings)
            )

        # Reading order over atomic items.
        items: list[Item] = [Item(t.bbox, "table", i) for i, t in enumerate(tables)]
        items += [Item(kv.bbox, "kv", i) for i, kv in enumerate(kvs)]
        items += [Item(ln.bbox, "line", ln) for ln in free if ln.id not in paired]
        median_h = lh
        blocks: list[Block] = []
        for leaf in xy_cut(items, lh):
            leaf_blocks: list[Block] = []
            leaf_lines = [it.ref for it in leaf if it.kind == "line"]
            leaf_blocks += text_blocks(leaf_lines, lh, median_h) if leaf_lines else []
            for it in leaf:
                if it.kind == "table":
                    leaf_blocks.append(Block("table", "", it.bbox, tables[it.ref].confidence, ref=it.ref))
                elif it.kind == "kv":
                    kv = kvs[it.ref]
                    leaf_blocks.append(Block("key_value", f"{kv.key}: {kv.value}", kv.bbox, kv.confidence, kv.line_ids, ref=it.ref))
            leaf_blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
            blocks += leaf_blocks

        # Title: a heading directly above a table.
        for b in blocks:
            if b.type != "table":
                continue
            t = tables[b.ref]
            above = [h for h in blocks if h.type == "heading" and h.bbox[3] <= t.bbox[1] + 2 and t.bbox[1] - h.bbox[3] <= 3 * lh]
            if above:
                t.title = max(above, key=lambda h: h.bbox[3]).text.replace("\n", " ")
        return blocks, kvs, tables

    def _fix_numeric_columns(self, img: np.ndarray, tc) -> None:
        """In a column that is mostly numbers, a non-numeric cell ('ost' for a
        handwritten 750) is re-read and a numeric reading preferred."""
        from .image_tables import cell_text, is_value
        from .ocr_models import crop_polygon

        body = tc.cells[1:] if tc.has_header else tc.cells
        if len(body) < 3 or self.profile["reread_below"] is None:
            return
        for c in range(len(body[0])):
            texts = [cell_text(row[c]) for row in body if row[c]]
            if len(texts) < 3 or sum(is_value(t) for t in texts) < 0.6 * len(texts):
                continue
            if median(len(t) for t in texts) < 2:
                continue  # tick marks / single symbols, not a number column
            for row in body:
                for ln in row[c]:
                    if len(ln.text) > 12 or any(ch.isdigit() for ch in ln.text):
                        continue  # already numeric-ish ('1 set', '240-'): keep
                    crop = crop_polygon(img, ln.poly)
                    options = [self.models.recognize_crop(crop, m) for m in ("medium", "en", "small")]
                    numeric = [o for o in options if o and is_value(o.text) and o.score >= 0.5 and len(o.text) >= 2]
                    if numeric:
                        best = max(numeric, key=lambda o: o.score)
                        ln.alternatives.append(type(best)(ln.text, ln.score, ln.model))
                        ln.text, ln.score, ln.model = best.text, best.score, best.model + " (numeric column)"
                        tc.warnings.append(f"'{ln.alternatives[-1].text}' re-read as '{ln.text}' to match a numeric column")

    def _quality_notes(self, q: QualityReport, lines: list[OCRLine]) -> None:
        if not lines:
            q.issues.append("no_text")
            q.notes.append("No text was recognised on this page.")
            return
        low = [ln for ln in lines if ln.score < self.config.review_threshold]
        weak = sum(1 for ln in lines if ln.score < 0.9) / len(lines)
        if weak >= 0.2:
            q.issues.append("hard_to_read_text")
            q.notes.append(
                f"{weak:.0%} of text lines were hard to read (typical of handwriting, stamps or poor print); "
                "values from those lines are less reliable."
            )
        if low:
            q.issues.append("needs_review")
            q.notes.append(f"{len(low)} line(s) below {self.config.review_threshold:.2f} confidence are listed under 'review' for manual checking.")
        if self.models.unavailable:
            q.notes.append("Some accuracy models could not be loaded (" + ", ".join(self.models.unavailable) + "); results use the base model only.")


def _is_latin(lines: list[OCRLine]) -> bool:
    confident = "".join(ln.text for ln in lines if ln.score >= 0.85)
    cjk = sum(1 for ch in confident if has_cjk(ch))
    alpha = sum(1 for ch in confident if ch.isalpha())
    return alpha > 0 and cjk <= 0.05 * alpha
