import cv2
import numpy as np

from pdf_table_extractor.grid import binarize, deskew, detect_ruled_tables, estimate_skew

SCALE = 200 / 72


def grid_image(rows=4, cols=3, cell_w=150, cell_h=45, origin=(100, 150), text=True):
    img = np.full((900, 800, 3), 255, np.uint8)
    x0, y0 = origin
    for r in range(rows + 1):
        cv2.line(img, (x0, y0 + r * cell_h), (x0 + cols * cell_w, y0 + r * cell_h), (90, 90, 90), 2)
    for c in range(cols + 1):
        cv2.line(img, (x0 + c * cell_w, y0), (x0 + c * cell_w, y0 + rows * cell_h), (90, 90, 90), 2)
    if text:
        for r in range(rows):
            for c in range(cols):
                cv2.putText(img, "Bold Text", (x0 + c * cell_w + 10, y0 + r * cell_h + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
    return img


def test_detects_exact_grid_and_ignores_bold_text():
    [region] = detect_ruled_tables(grid_image(), SCALE)
    assert (region.n_rows, region.n_cols) == (4, 3)


def test_no_tables_on_text_only_page():
    img = np.full((900, 800, 3), 255, np.uint8)
    cv2.putText(img, "Just a heading", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    assert detect_ruled_tables(img, SCALE) == []


def test_broken_outer_border_still_yields_full_grid():
    img = grid_image(text=False)
    # erase most of the bottom rule, as happens on poor scans
    cv2.line(img, (110, 150 + 4 * 45), (500, 150 + 4 * 45), (255, 255, 255), 3)
    [region] = detect_ruled_tables(img, SCALE)
    assert (region.n_rows, region.n_cols) == (4, 3)


def test_skew_estimation_and_correction():
    img = grid_image()
    h, w = img.shape[:2]
    rotated = cv2.warpAffine(img, cv2.getRotationMatrix2D((w / 2, h / 2), 1.5, 1.0), (w, h), borderValue=(255, 255, 255))
    angle = estimate_skew(binarize(cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)))
    assert abs(angle + 1.5) <= 0.2
    [region] = detect_ruled_tables(deskew(rotated, angle), SCALE)
    assert (region.n_rows, region.n_cols) == (4, 3)
