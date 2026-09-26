"""Thin, thread-safe wrapper around RapidOCR 3.x with lazy model loading."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

from .errors import OCRUnavailableError

log = logging.getLogger(__name__)


@dataclass
class OCRBox:
    bbox: tuple[float, float, float, float]  # axis-aligned, image pixels
    text: str
    score: float

    @property
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


class OCREngine:
    """Lazily builds one RapidOCR instance and serialises calls to it.

    RapidOCR mutates internal parameters on every call, so concurrent use of
    one instance is unsafe; the lock makes a shared engine safe across threads.
    """

    def __init__(self, min_score: float = 0.5, max_side_len: int = 4000, log_level: str = "error"):
        self._params = {
            "Global.text_score": min_score,
            "Global.max_side_len": max_side_len,
            "Global.log_level": log_level,
        }
        self._engine = None
        self._init_error: OCRUnavailableError | None = None
        self._lock = threading.Lock()

    def _get(self):
        if self._init_error is not None:  # fail fast; do not retry per page
            raise self._init_error
        if self._engine is None:
            try:
                from rapidocr import RapidOCR

                self._engine = RapidOCR(params=self._params)
            except ImportError as exc:  # pragma: no cover - depends on env
                self._init_error = OCRUnavailableError("rapidocr is not installed; install it or use --mode native")
                raise self._init_error from exc
            except Exception as exc:  # model download/load failures
                self._init_error = OCRUnavailableError(f"could not initialise RapidOCR: {exc}")
                raise self._init_error from exc
            log.info("RapidOCR engine initialised")
        return self._engine

    def recognize(self, img_bgr: np.ndarray, use_det: bool = True) -> list[OCRBox]:
        """Run OCR on a BGR image and return axis-aligned boxes."""
        with self._lock:
            engine = self._get()
            if use_det:
                out = engine(img_bgr, use_det=True, use_cls=True, use_rec=True)
            else:
                out = engine(img_bgr, use_det=False, use_cls=False, use_rec=True)

        # Fields may be None, tuples or numpy arrays depending on the version.
        txts = list(getattr(out, "txts", None) if getattr(out, "txts", None) is not None else ())
        scores = list(getattr(out, "scores", None) if getattr(out, "scores", None) is not None else ())
        boxes = getattr(out, "boxes", None)
        result: list[OCRBox] = []
        h, w = img_bgr.shape[:2]
        for i, text in enumerate(txts):
            text = (text or "").strip()
            if not text:
                continue
            if use_det and boxes is not None and i < len(boxes):
                pts = np.asarray(boxes[i], dtype=float)
                bbox = (float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max()))
            else:
                bbox = (0.0, 0.0, float(w), float(h))
            score = float(scores[i]) if i < len(scores) else 0.0
            result.append(OCRBox(bbox=bbox, text=text, score=score))
        return result
