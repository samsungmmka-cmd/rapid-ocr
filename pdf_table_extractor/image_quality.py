"""Image quality assessment for photos and scans of documents.

Produces numeric metrics plus plain-language notes that travel with the
extraction result, so reviewers know when the source was hard to read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .grid import binarize, estimate_skew

# Thresholds (tuned for document images; see tests/test_image_reader.py).
MIN_CHAR_HEIGHT_PX = 12  # below this, glyphs are too small for reliable OCR
BLUR_VARIANCE = 2000.0  # contrast-normalised Laplacian variance on text; below = blurry
LOW_CONTRAST = 90.0  # paper minus ink intensity (0..255)
DARK_MEAN = 110.0
NOISE_SIGMA = 6.0
UNEVEN_LIGHTING = 0.10  # background brightness spread / mean
SKEW_NOTE_DEG = 0.5


@dataclass
class QualityReport:
    width: int
    height: int
    char_height_px: float | None
    sharpness: float
    contrast: float
    brightness: float
    noise_sigma: float
    lighting_unevenness: float
    skew_deg: float
    ink_ratio: float = 0.0
    issues: list[str] = field(default_factory=list)  # machine-readable codes
    notes: list[str] = field(default_factory=list)  # human-readable explanations
    ocr_confidence: float | None = None
    enhancements: list[str] = field(default_factory=list)

    @property
    def level(self) -> str:
        """good | fair | poor, combining image issues and OCR confidence."""
        severe = {"low_resolution", "blurry", "very_low_contrast"}
        n_severe = len(severe.intersection(self.issues))
        if self.char_height_px is not None and self.char_height_px < 8:
            n_severe += 1  # tiny glyphs are unreliable on their own
        if "no_tables_found" in self.issues and self.issues != ["no_tables_found"]:
            n_severe += 1
        conf = self.ocr_confidence
        if n_severe >= 2 or (conf is not None and conf < 0.75):
            return "poor"
        moderate = {"needs_review", "hard_to_read_text", "low_ocr_confidence"}
        if n_severe == 1 or len(self.issues) >= 3 or (conf is not None and conf < 0.9) or moderate & set(self.issues):
            return "fair"
        return "good"

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "issues": self.issues,
            "notes": self.notes,
            "enhancements": self.enhancements,
            "ocr_confidence": None if self.ocr_confidence is None else round(self.ocr_confidence, 4),
            "metrics": {
                "width": self.width,
                "height": self.height,
                "char_height_px": None if self.char_height_px is None else round(self.char_height_px, 1),
                "sharpness": round(self.sharpness, 1),
                "contrast": round(self.contrast, 1),
                "brightness": round(self.brightness, 1),
                "noise_sigma": round(self.noise_sigma, 2),
                "lighting_unevenness": round(self.lighting_unevenness, 3),
                "skew_deg": round(self.skew_deg, 2),
            },
        }


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def estimate_char_height(gray: np.ndarray) -> float | None:
    """Median glyph height in pixels from connected components, or None."""
    ink = binarize(gray)
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n <= 1:
        return None
    w, h, area = stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT], stats[1:, cv2.CC_STAT_AREA]
    max_h = max(gray.shape[0] // 10, 8)
    keep = (h >= 4) & (h <= max_h) & (w <= 3 * h) & (area >= 8) & (area <= 0.9 * w * h + 1)
    heights = h[keep]
    if heights.size < 15:
        return None
    return float(np.median(heights))


def estimate_noise(gray: np.ndarray) -> float:
    """Immerkaer's fast noise sigma estimate."""
    g = gray.astype(np.float64)
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    conv = cv2.filter2D(g, -1, kernel)[1:-1, 1:-1]
    h, w = g.shape
    if h < 3 or w < 3:
        return 0.0
    return float(np.sqrt(np.pi / 2) * np.abs(conv).sum() / (6 * (w - 2) * (h - 2)))


def _background(gray: np.ndarray) -> np.ndarray:
    """Paper brightness map: text removed by a large closing, then blurred."""
    k = max(15, (min(gray.shape) // 30) | 1)
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    return cv2.GaussianBlur(closed, (0, 0), k / 2)


def assess(img: np.ndarray) -> QualityReport:
    gray = to_gray(img)
    h, w = gray.shape[:2]
    char_h = estimate_char_height(gray)

    ink = binarize(gray)
    text_mask = cv2.dilate(ink, np.ones((5, 5), np.uint8)) > 0
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharp = float(lap[text_mask].var()) if text_mask.any() else float(lap.var())

    paper = float(np.percentile(gray, 95))
    inkv = float(np.percentile(gray[ink > 0], 10)) if (ink > 0).any() else paper
    contrast = paper - inkv
    # Normalise by contrast so faint-but-sharp text is not reported as blurry.
    sharp = sharp / max((contrast / 255.0) ** 2, 0.01)
    bg = _background(gray)
    unevenness = float((np.percentile(bg, 95) - np.percentile(bg, 5)) / max(bg.mean(), 1.0))

    report = QualityReport(
        width=w,
        height=h,
        char_height_px=char_h,
        sharpness=sharp,
        contrast=contrast,
        brightness=float(gray.mean()),
        noise_sigma=estimate_noise(gray),
        lighting_unevenness=unevenness,
        skew_deg=-estimate_skew(ink) if ink.any() else 0.0,
        ink_ratio=float((ink > 0).mean()),
    )
    _add_notes(report)
    return report


def _add_notes(r: QualityReport) -> None:
    def add(code: str, note: str) -> None:
        r.issues.append(code)
        r.notes.append(note)

    if r.char_height_px is None and r.ink_ratio > 0.005:
        add(
            "low_resolution",
            "Low resolution: the text is too small or too blurred for individual characters to be measured. "
            "Results are unreliable; rescan at 300 DPI or photograph the page closer and in focus.",
        )
    elif r.char_height_px is None:
        add("no_text_detected", "Very little text-like content was found; the image may be blank or not a document.")
    elif r.char_height_px < MIN_CHAR_HEIGHT_PX:
        add(
            "low_resolution",
            f"Low resolution: characters are about {r.char_height_px:.0f}px tall (recommended 20px or more, e.g. scan at 300 DPI). "
            "The image was upscaled, but small digits and punctuation may still be misread.",
        )
    if r.sharpness < BLUR_VARIANCE:
        add("blurry", f"Image is blurry or out of focus (sharpness {r.sharpness:.0f}, expected above {BLUR_VARIANCE:.0f}). Verify numbers and codes.")
    if r.contrast < LOW_CONTRAST / 2:
        add("very_low_contrast", f"Very low contrast between text and background ({r.contrast:.0f}/255); faint text may be missed.")
    elif r.contrast < LOW_CONTRAST:
        add("low_contrast", f"Low contrast between text and background ({r.contrast:.0f}/255); contrast was enhanced.")
    if r.brightness < DARK_MEAN:
        add("dark", f"Image is dark (mean brightness {r.brightness:.0f}/255); brightness was normalised.")
    if r.noise_sigma > NOISE_SIGMA:
        add("noisy", f"Image is noisy or grainy (noise level {r.noise_sigma:.1f}); denoising was applied.")
    if r.lighting_unevenness > UNEVEN_LIGHTING:
        add("uneven_lighting", "Uneven lighting or shadows detected; the background was flattened before OCR.")
    if abs(r.skew_deg) >= SKEW_NOTE_DEG:
        add("skewed", f"Page is tilted by about {abs(r.skew_deg):.1f} degrees; it was straightened.")
