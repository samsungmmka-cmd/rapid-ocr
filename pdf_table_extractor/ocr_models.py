"""RapidOCR model ensemble for high-accuracy image data extraction.

Strategy (measured on real invoices, see tests/test_image_data.py):
  1. One full-page pass with the fast multilingual PP-OCRv6 *small* model
     (detection + orientation classifier + recognition).
  2. Lines with low confidence, or with script-inconsistent characters
     (e.g. Chinese glyphs in a Latin document), are cropped along their
     rotated box and re-read by stronger recognisers: PP-OCRv6 *medium*
     (multilingual, ~15x slower, markedly better on handwriting) and
     PP-OCRv5 *English* (fast, cannot emit non-Latin glyphs).
  3. The best candidate wins, with guards against truncated re-reads.

Extra models are downloaded by RapidOCR on first use. If a download or load
fails, the ensemble degrades to the models that are available.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .errors import OCRUnavailableError

log = logging.getLogger(__name__)

_CJK = re.compile(r"[　-〿぀-ヿ㐀-䶿一-鿿가-힯＀-￯℀-⅏]")
_LATINISH = re.compile(r"[A-Za-z0-9]")


@dataclass
class Candidate:
    text: str
    score: float
    model: str


@dataclass
class OCRLine:
    """One recognised text line. Coordinates are processed-image pixels."""

    id: int
    poly: np.ndarray  # (4, 2) float, clockwise from top-left
    text: str
    score: float
    model: str = "small"
    alternatives: list[Candidate] = field(default_factory=list)
    # Word-level boxes from the first pass: [(word, (4,2) poly)]. Only valid
    # while ``text`` is unchanged; see ``word_boxes``.
    words: list[tuple[str, np.ndarray]] = field(default_factory=list)

    def word_boxes(self) -> list[tuple[str, np.ndarray]]:
        if self.words and "".join(w for w, _ in self.words).replace(" ", "") == self.text.replace(" ", ""):
            return self.words
        return []

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (
            float(self.poly[:, 0].min()),
            float(self.poly[:, 1].min()),
            float(self.poly[:, 0].max()),
            float(self.poly[:, 1].max()),
        )

    @property
    def height(self) -> float:
        # Height perpendicular to the text direction, robust to slant.
        return float((np.linalg.norm(self.poly[3] - self.poly[0]) + np.linalg.norm(self.poly[2] - self.poly[1])) / 2)

    @property
    def cx(self) -> float:
        b = self.bbox
        return (b[0] + b[2]) / 2

    @property
    def cy(self) -> float:
        b = self.bbox
        return (b[1] + b[3]) / 2


def has_cjk(text: str) -> bool:
    return bool(_CJK.search(text))


def crop_polygon(img: np.ndarray, poly: np.ndarray, pad: int = 4) -> np.ndarray:
    """Rectified crop of a (possibly rotated) text box."""
    pts = np.asarray(poly, dtype=np.float32)
    w = int(round(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[3] - pts[2]))))
    h = int(round(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2]))))
    w, h = max(w, 4), max(h, 4)
    dst = np.float32([[pad, pad], [w + pad, pad], [w + pad, h + pad], [pad, h + pad]])
    m = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img, m, (w + 2 * pad, h + 2 * pad), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


class OCRModels:
    """Lazily loaded RapidOCR instances shared behind one lock."""

    def __init__(self, min_score: float = 0.3, max_side_len: int = 4000, use_medium: bool = True, use_english: bool = True):
        self._lock = threading.RLock()
        self._engines: dict[str, Any] = {}
        self._failed: dict[str, str] = {}
        self._base_params = {"Global.text_score": min_score, "Global.max_side_len": max_side_len, "Global.log_level": "error"}
        self.use_medium = use_medium
        self.use_english = use_english

    # ------------------------------------------------------------- loading
    def _params(self, name: str) -> dict[str, Any]:
        from rapidocr import LangRec, ModelType, OCRVersion

        params = dict(self._base_params)
        import os

        if os.getenv("OCR_MODEL_DIR"):  # writable model cache, e.g. /tmp on serverless hosts
            params["Global.model_root_dir"] = os.environ["OCR_MODEL_DIR"]
        if name == "medium":
            params["Rec.model_type"] = ModelType.MEDIUM
        elif name == "en":
            params.update({"Rec.ocr_version": OCRVersion.PPOCRV5, "Rec.lang_type": LangRec.EN, "Rec.model_type": ModelType.MOBILE})
        return params

    def _engine(self, name: str):
        if name in self._engines:
            return self._engines[name]
        if name in self._failed:
            return None
        try:
            from rapidocr import RapidOCR

            self._engines[name] = RapidOCR(params=self._params(name))
            log.info("loaded RapidOCR model set '%s'", name)
        except Exception as exc:
            if name == "small":
                raise OCRUnavailableError(f"could not initialise RapidOCR: {exc}") from exc
            self._failed[name] = str(exc)
            log.warning("RapidOCR model '%s' unavailable (%s); continuing without it", name, exc)
            return None
        return self._engines[name]

    @property
    def unavailable(self) -> dict[str, str]:
        return dict(self._failed)

    # ------------------------------------------------------------- calls
    def read_page(self, img: np.ndarray) -> list[OCRLine]:
        with self._lock:
            out = self._engine("small")(img, use_det=True, use_cls=True, use_rec=True, return_word_box=True)
        boxes = getattr(out, "boxes", None)
        txts = list(out.txts) if getattr(out, "txts", None) is not None else []
        scores = list(out.scores) if getattr(out, "scores", None) is not None else []
        lines = []
        for i, text in enumerate(txts):
            text = (text or "").strip()
            if not text or boxes is None or i >= len(boxes):
                continue
            words = []
            word_results = getattr(out, "word_results", None)
            if word_results is not None and i < len(word_results) and word_results[i]:
                for item in word_results[i]:
                    try:
                        words.append((str(item[0]), np.asarray(item[2], dtype=np.float64)))
                    except (IndexError, TypeError, ValueError):
                        words = []
                        break
            lines.append(
                OCRLine(id=len(lines), poly=np.asarray(boxes[i], dtype=np.float64), text=text, score=float(scores[i]), words=words)
            )
        return lines

    def recognize_crop(self, crop: np.ndarray, name: str) -> Candidate | None:
        with self._lock:
            engine = self._engine(name)
            if engine is None:
                return None
            out = engine(crop, use_det=False, use_cls=False, use_rec=True)
        txts = list(out.txts) if getattr(out, "txts", None) is not None else []
        scores = list(out.scores) if getattr(out, "scores", None) is not None else []
        if not txts or not (txts[0] or "").strip():
            return None
        return Candidate(text=txts[0].strip(), score=float(scores[0]) if scores else 0.0, model=name)

    def score_upright(self, img: np.ndarray) -> tuple[float, float]:
        """(length-weighted mean score, share of tall boxes) without the
        orientation classifier; used to detect sideways/upside-down pages."""
        with self._lock:
            out = self._engine("small")(img, use_det=True, use_cls=False, use_rec=True)
        txts = list(out.txts) if getattr(out, "txts", None) is not None else []
        scores = list(out.scores) if getattr(out, "scores", None) is not None else []
        boxes = getattr(out, "boxes", None)
        if not txts:
            return 0.0, 0.0
        weights = [max(len(t), 1) for t in txts]
        score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        tall = 0
        if boxes is not None:
            for b in boxes:
                pts = np.asarray(b, dtype=float)
                tall += np.ptp(pts[:, 1]) > 1.5 * np.ptp(pts[:, 0])
        return float(score), tall / max(len(txts), 1)

    # --------------------------------------------------------- ensemble
    def refine(self, img: np.ndarray, line: OCRLine, latin_document: bool, thorough: bool = False) -> OCRLine:
        """Re-read one line with the stronger models and keep the best reading."""
        crop = crop_polygon(img, line.poly)
        models = []
        if self.use_medium:
            models.append("medium")
        if self.use_english and latin_document:
            models.append("en")
        candidates = [Candidate(line.text, line.score, line.model)]
        for name in models:
            cand = self.recognize_crop(crop, name)
            if cand:
                candidates.append(cand)
        if thorough and self.use_medium and crop.shape[0] < 40:
            big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            cand = self.recognize_crop(big, "medium")
            if cand:
                cand.model = "medium-2x"
                candidates.append(cand)

        best = max(candidates, key=lambda c: _adjusted(c, candidates, line.text, latin_document))
        line.alternatives = [c for c in candidates if c is not best]
        if best.model != line.model or best.text != line.text:
            line.text, line.score, line.model = best.text, best.score, best.model
        return line


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _adjusted(c: Candidate, all_c: list[Candidate], original: str, latin: bool) -> float:
    score = c.score
    if latin and has_cjk(c.text):
        score *= 0.4  # foreign-script glyphs in a Latin document are misreads
    if len(original) >= 12 and len(c.text) < 0.6 * len(original) and not has_cjk(original):
        score *= 0.5  # rec-only re-reads can truncate very long lines
    if not _LATINISH.search(c.text) and _LATINISH.search(original):
        score *= 0.7
    if sum(1 for o in all_c if _norm(o.text) == _norm(c.text)) >= 2:
        score += 0.05  # independent models agree
    return score
