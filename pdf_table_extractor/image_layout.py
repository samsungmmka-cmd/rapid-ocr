"""Layout analysis for OCR'd document images: key-value pairs, reading order
(recursive XY-cut) and block classification (heading, paragraph, list item)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from statistics import median

from .image_tables import group_rows, is_label, is_value
from .ocr_models import OCRLine

_INLINE_KV = re.compile(r"^\s*(?P<key>[^:：]{1,40}?)\s*[:：]\s*(?P<value>\S.*)$")
_NO_KV = re.compile(
    r"^\s*(?P<key>.{1,30}?(?:\b(?:No|Nr|Number)\.?|\s#))\s*(?P<value>[A-Z0-9][A-Z0-9/\-.]{1,}(?:\s[0-9][0-9/\-.]*){0,3})\s*$",
    re.IGNORECASE,
)
_BULLET = re.compile(r"^\s*(?:[•●▪■◦\-–*·]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)])\s+\S")
_NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


@dataclass
class KeyValue:
    key: str
    value: str
    confidence: float
    bbox: tuple[float, float, float, float]
    source: str  # "inline" | "pair" | "table_footer"
    line_ids: list[int] = field(default_factory=list)

    @property
    def number(self) -> float | None:
        return parse_number(self.value)


@dataclass
class Block:
    type: str  # heading | paragraph | list_item | key_value | table | uncertain
    text: str
    bbox: tuple[float, float, float, float]
    confidence: float
    line_ids: list[int] = field(default_factory=list)
    level: int | None = None  # heading level
    ref: int | None = None  # index into key_values / tables for those types


def parse_number(value: str) -> float | None:
    """'$1,234.50' -> 1234.5, '7226-' -> 7226.0, '6.25%' -> 6.25; else None.

    Only for values that are essentially numeric (not dates or IDs)."""
    v = value.strip()
    if re.search(r"\d[/|]\d", v) or re.search(r"[A-Za-z]{2,}", v.replace("Rs", "").replace("INR", "").replace("USD", "")):
        return None
    nums = _NUMBER.findall(v)
    if len(nums) != 1:
        return None
    try:
        return float(nums[0].replace(",", ""))
    except ValueError:
        return None


def normalize_value(value: str) -> str:
    """Fix common OCR confusions inside dates: 28|10|2024 -> 28/10/2024."""
    return re.sub(r"(?<=\d)\s*[|lI]\s*(?=\d)", "/", value) if re.fullmatch(r"[\d\s|lI/.-]+", value.strip()) else value


def union_bbox(boxes) -> tuple[float, float, float, float]:
    boxes = list(boxes)
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


# ---------------------------------------------------------------- key-values
def inline_kv(line: OCRLine) -> KeyValue | None:
    if _BULLET.match(line.text):
        return None  # "• Label: link" is a list item, not a form field
    m = _INLINE_KV.match(line.text)
    if m and len(m.group("key").split()) <= 4 and is_label(m.group("key")):
        return KeyValue(m.group("key").strip(), normalize_value(m.group("value").strip()), line.score, line.bbox, "inline", [line.id])
    m = _NO_KV.match(line.text)
    if m and any(ch.isdigit() for ch in m.group("value")):
        return KeyValue(m.group("key").strip(), m.group("value").strip(), line.score, line.bbox, "inline", [line.id])
    return None


def pair_kvs(lines: list[OCRLine], page_width: float) -> tuple[list[KeyValue], set[int]]:
    """Label and value side by side ('INVOICE #   US-001'): each label is paired
    with its nearest right-hand neighbour that shares its baseline."""

    def same_line(a: OCRLine, b: OCRLine) -> bool:
        ov = min(a.bbox[3], b.bbox[3]) - max(a.bbox[1], b.bbox[1])
        return ov >= 0.5 * min(a.bbox[3] - a.bbox[1], b.bbox[3] - b.bbox[1])

    kvs: list[KeyValue] = []
    used: set[int] = set()
    for left in sorted(lines, key=lambda ln: (ln.bbox[1], ln.bbox[0])):
        if left.id in used or left.score < 0.75:
            continue
        lt = left.text.strip()
        if not is_label(lt) or is_value(lt) or ":" in lt[:-1]:
            continue
        right_side = [
            r for r in lines
            if r.id != left.id and r.id not in used and same_line(left, r)
            # handwriting may start slightly over a printed label
            and r.bbox[0] >= left.bbox[0] + 0.6 * (left.bbox[2] - left.bbox[0])
        ]
        if not right_side:
            continue
        right = min(right_side, key=lambda r: r.bbox[0])
        rt = right.text.strip()
        if right.score < 0.75 or right.bbox[0] - left.bbox[2] > 0.25 * page_width:
            continue  # scribbles/stamps, or too far apart to be one field
        ratio = max(left.height, right.height) / max(min(left.height, right.height), 1)
        if ratio > 1.8:
            continue  # a big heading next to small print is not a field
        short_abbrev = len(lt) <= 5 and lt.endswith(".")
        if not (lt.endswith(":") or is_value(rt) or short_abbrev):
            continue
        kvs.append(
            KeyValue(
                key=lt.rstrip(":").strip(),
                value=normalize_value(rt),
                confidence=min(left.score, right.score),
                bbox=union_bbox([left.bbox, right.bbox]),
                source="pair",
                line_ids=[left.id, right.id],
            )
        )
        used |= {left.id, right.id}
    return kvs, used


# -------------------------------------------------------------- reading order
@dataclass
class Item:
    bbox: tuple[float, float, float, float]
    kind: str  # "line" | "kv" | "table"
    ref: object


def _gaps(intervals: list[tuple[float, float]], min_gap: float) -> list[tuple[float, float]]:
    """Whitespace gaps (>= min_gap) between the union of intervals."""
    intervals = sorted(intervals)
    gaps = []
    end = intervals[0][1]
    for a, b in intervals[1:]:
        if a - end >= min_gap:
            gaps.append((end, a))
        end = max(end, b)
    return gaps


def xy_cut(items: list[Item], lh: float) -> list[list[Item]]:
    """Recursive XY-cut: split at the widest whitespace gap (relative to its
    threshold) until no gap remains. Leaves come out in reading order."""
    if len(items) <= 1:
        return [items]
    min_y, min_x = 1.0 * lh, 1.0 * lh
    ygaps = _gaps([(it.bbox[1], it.bbox[3]) for it in items], min_y)
    xgaps = _gaps([(it.bbox[0], it.bbox[2]) for it in items], min_x)
    best_y = max(ygaps, key=lambda g: g[1] - g[0], default=None)
    best_x = max(xgaps, key=lambda g: g[1] - g[0], default=None)
    if best_y is None and best_x is None:
        return [items]
    use_y = best_x is None or (best_y is not None and (best_y[1] - best_y[0]) / min_y >= (best_x[1] - best_x[0]) / min_x)
    if use_y:
        cuts = [(a + b) / 2 for a, b in ygaps]
        key = lambda it: (it.bbox[1] + it.bbox[3]) / 2
    else:
        cuts = [(a + b) / 2 for a, b in xgaps]
        key = lambda it: (it.bbox[0] + it.bbox[2]) / 2
    groups: list[list[Item]] = [[] for _ in range(len(cuts) + 1)]
    for it in items:
        groups[sum(1 for c in cuts if key(it) > c)].append(it)
    out = []
    for g in groups:
        if g:
            out.extend(xy_cut(g, lh))
    return out


# ------------------------------------------------------------------- blocks
def _uppercase_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    return sum(ch.isupper() for ch in letters) / len(letters) if letters else 0.0


def is_uncertain(lines: list[OCRLine], threshold: float = 0.75) -> bool:
    """Fragments that are most likely a signature, stamp or scribble."""
    alnum = sum(ch.isalnum() for ln in lines for ch in ln.text)
    return all(ln.score < threshold for ln in lines) or (alnum <= 2 and all(ln.score < 0.9 for ln in lines))


def _merge_row_neighbours(lines: list[OCRLine], lh: float) -> tuple[list[OCRLine], dict[int, list[int]]]:
    """Join detections on the same text row that are close together (e.g. a
    script-font 'Thank' + 'you' detected as two boxes). Returns the new lines
    and, for merged ones, the ids of their members."""
    import numpy as np

    out: list[OCRLine] = []
    members: dict[int, list[int]] = {}
    for row in group_rows(lines):
        current = [row[0]]
        for ln in row[1:]:
            prev = current[-1]
            similar = 0.7 <= ln.height / max(prev.height, 1) <= 1.4
            if ln.bbox[0] - prev.bbox[2] <= 0.8 * max(prev.height, ln.height) and similar:
                current.append(ln)
            else:
                out.append(_combine(current, members, np))
                current = [ln]
        out.append(_combine(current, members, np))
    return out, members


def _combine(parts: list[OCRLine], members: dict[int, list[int]], np) -> OCRLine:
    if len(parts) == 1:
        return parts[0]
    b = union_bbox(p.bbox for p in parts)
    merged = OCRLine(
        id=parts[0].id,
        poly=np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]], dtype=float),
        text=" ".join(p.text for p in parts),
        score=min(p.score for p in parts),
        model=parts[0].model,
    )
    members[id(merged)] = [p.id for p in parts]
    return merged


def text_blocks(leaf_lines: list[OCRLine], lh: float, median_h: float) -> list[Block]:
    """Group a leaf's lines into blocks: stacked, overlapping lines with small
    vertical gaps form one block; then classify each block."""
    lines, members = _merge_row_neighbours(leaf_lines, lh)
    lines = sorted(lines, key=lambda ln: (ln.bbox[1], ln.bbox[0]))
    groups: list[list[OCRLine]] = []
    for ln in lines:
        target = None
        for g in groups:
            last = g[-1]
            vgap = ln.bbox[1] - last.bbox[3]
            hover = min(ln.bbox[2], last.bbox[2]) - max(ln.bbox[0], last.bbox[0])
            aligned = abs(ln.bbox[0] - last.bbox[0]) <= 1.5 * lh
            size_ok = 0.6 <= ln.height / max(last.height, 1) <= 1.6
            if -0.5 * lh <= vgap <= 0.9 * lh and (hover > 0 or aligned) and size_ok:
                target = g
        (target.append(ln) if target else groups.append([ln]))

    blocks: list[Block] = []
    for g in groups:
        bbox = union_bbox(ln.bbox for ln in g)
        conf = min(ln.score for ln in g)
        text = "\n".join(ln.text for ln in g)
        ids = [i for ln in g for i in members.get(id(ln), [ln.id])]
        if is_uncertain(g):
            blocks.append(Block("uncertain", text, bbox, conf, ids))
            continue
        h = median(ln.height for ln in g)
        words = len(text.split())
        if len(g) == 1 and (h >= 1.45 * median_h or (_uppercase_ratio(text) >= 0.8 and words <= 6 and len(text) >= 3 and not text.endswith("."))):
            level = 1 if h >= 2.0 * median_h else 2
            blocks.append(Block("heading", text, bbox, conf, ids, level=level))
            continue
        if _BULLET.match(g[0].text):
            current: list[OCRLine] = []
            for ln in g:
                if _BULLET.match(ln.text) and current:
                    blocks.append(Block("list_item", " ".join(x.text for x in current), union_bbox(x.bbox for x in current), min(x.score for x in current), [i for x in current for i in members.get(id(x), [x.id])]))
                    current = []
                current.append(ln)
            if current:
                blocks.append(Block("list_item", " ".join(x.text for x in current), union_bbox(x.bbox for x in current), min(x.score for x in current), [i for x in current for i in members.get(id(x), [x.id])]))
            continue
        blocks.append(Block("paragraph", text, bbox, conf, ids))
    return blocks
