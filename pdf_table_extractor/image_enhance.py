"""Preprocessing that makes unclear document images readable for OCR.

Every step is conditional on measured quality and is recorded, so the result
states exactly what was done to the image.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from .grid import binarize, deskew, estimate_skew
from .image_quality import (
    BLUR_VARIANCE,
    LOW_CONTRAST,
    NOISE_SIGMA,
    UNEVEN_LIGHTING,
    DARK_MEAN,
    QualityReport,
    _background,
    to_gray,
)

log = logging.getLogger(__name__)

MAX_PIXELS = 40_000_000
TARGET_CHAR_PX = 22.0  # glyph height RapidOCR reads most reliably
AGGRESSIVE_CHAR_PX = 28.0


@dataclass
class Enhanced:
    image: np.ndarray  # BGR, ready for OCR
    upscale: float = 1.0
    steps: list[str] = field(default_factory=list)
    # 3x3 matrix mapping input-image points to enhanced-image points.
    matrix: np.ndarray = field(default_factory=lambda: np.eye(3))


def _cap_factor(shape: tuple[int, ...], factor: float) -> float:
    h, w = shape[:2]
    if h * w * factor * factor > MAX_PIXELS:
        factor = (MAX_PIXELS / (h * w)) ** 0.5
    return max(factor, 1.0)


def correct_perspective(img: np.ndarray) -> tuple[np.ndarray, bool]:
    """Backward-compatible wrapper: (image, whether it was corrected)."""
    out, matrix = correct_perspective_matrix(img)
    return out, matrix is not None


def correct_perspective_matrix(img: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Flatten a photographed page lying on a darker background.

    Conservative: only acts when a large four-cornered page outline is found
    and the area outside it is clearly darker than the page itself (so a
    table border on white paper never triggers cropping).
    """
    gray = to_gray(img)
    h, w = gray.shape
    small_f = min(1.0, 1000 / max(h, w))
    small = cv2.resize(gray, (int(w * small_f), int(h * small_f))) if small_f < 1 else gray
    edges = cv2.Canny(cv2.GaussianBlur(small, (5, 5), 0), 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img, None
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    sh, sw = small.shape
    if area < 0.3 * sh * sw:
        return img, None
    quad = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
    if len(quad) != 4 or not cv2.isContourConvex(quad):
        return img, None
    pts = quad.reshape(4, 2).astype(np.float32) / small_f

    mask = np.zeros_like(gray)
    cv2.fillConvexPoly(mask, pts.astype(np.int32), 255)
    inside, outside = gray[mask > 0], gray[mask == 0]
    if outside.size < 0.03 * gray.size or float(np.median(inside)) - float(np.median(outside)) < 40:
        return img, None

    # Order corners: top-left, top-right, bottom-right, bottom-left.
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    tl, br, tr, bl = pts[np.argmin(s)], pts[np.argmax(s)], pts[np.argmin(d)], pts[np.argmax(d)]
    out_w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    out_h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if out_w < 100 or out_h < 100:
        return img, None
    m = cv2.getPerspectiveTransform(
        np.array([tl, tr, br, bl], np.float32),
        np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], np.float32),
    )
    warped = cv2.warpPerspective(img, m, (out_w, out_h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    # Trim the page edge: remnants of the darker background along the border
    # would otherwise be read as a page-sized ruled frame.
    mx, my = max(3, int(out_w * 0.01)), max(3, int(out_h * 0.01))
    crop = np.array([[1, 0, -mx], [0, 1, -my], [0, 0, 1]], dtype=np.float64)
    return warped[my : out_h - my, mx : out_w - mx].copy(), crop @ m.astype(np.float64)


def enhance(img: np.ndarray, quality: QualityReport, aggressive: bool = False, do_deskew: bool = True) -> Enhanced:
    """Apply only the corrections the quality report calls for.

    ``aggressive`` is the second attempt for poor images: it always denoises,
    boosts contrast, upscales further and sharpens.
    """
    steps: list[str] = []
    gray = to_gray(img)

    if quality.lighting_unevenness > UNEVEN_LIGHTING or quality.brightness < DARK_MEAN or aggressive:
        bg = _background(gray).astype(np.float32)
        gray = np.clip(gray.astype(np.float32) / np.maximum(bg, 1.0) * 255.0, 0, 255).astype(np.uint8)
        steps.append("background flattened (shadows / uneven lighting removed)")

    # Upscale first: denoising and contrast work better on enlarged glyphs,
    # and enhancing contrast before upscaling amplifies JPEG block artefacts.
    upscale = 1.0
    matrix = np.eye(3)
    target = AGGRESSIVE_CHAR_PX if aggressive else TARGET_CHAR_PX
    if quality.char_height_px and quality.char_height_px < target * 0.8:
        upscale = _cap_factor(gray.shape, min(target / quality.char_height_px, 4.0))
        if upscale > 1.05:
            gray = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
            matrix = np.diag([upscale, upscale, 1.0])
            steps.append(f"upscaled x{upscale:.2f} (small text)")
        else:
            upscale = 1.0

    small_text = bool(quality.char_height_px and quality.char_height_px < 10)
    if quality.noise_sigma > NOISE_SIGMA or aggressive or small_text:
        strength = float(min(max(quality.noise_sigma * 1.5, 7.0), 20.0))
        gray = cv2.fastNlMeansDenoising(gray, None, h=strength, templateWindowSize=7, searchWindowSize=21)
        steps.append(f"denoised (strength {strength:.0f})")

    if quality.contrast < LOW_CONTRAST or aggressive:
        # Global stretch between ink and paper levels; noise-safe.
        lo, hi = np.percentile(gray, [1, 97])
        if hi - lo > 5:
            gray = np.clip((gray.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
        steps.append("contrast stretched")
        if aggressive:
            gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
            steps.append("local contrast enhanced (CLAHE)")

    if quality.sharpness < BLUR_VARIANCE or aggressive:
        blurred = cv2.GaussianBlur(gray, (0, 0), 1.2 * upscale)
        gray = cv2.addWeighted(gray, 1.6, blurred, -0.6, 0)
        steps.append("sharpened (unsharp mask)")

    if do_deskew:
        angle = estimate_skew(binarize(gray))
        if abs(angle) >= 0.05:
            h, w = gray.shape[:2]
            rot = np.vstack([cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0), [0, 0, 1]])
            gray = deskew(gray, angle)
            matrix = rot @ matrix
            steps.append(f"deskewed by {angle:.2f} degrees")

    return Enhanced(image=cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), upscale=upscale, steps=steps, matrix=matrix)
