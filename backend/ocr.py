"""Local English document OCR with explicit uncertainty reporting.

The primary backend is PaddleOCR. EasyOCR is retained as an offline fallback
and as a second opinion for uncertain PaddleOCR crops. No document data is
sent to a remote service.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
import logging
from typing import Iterable

import cv2
import numpy as np
from PIL import Image


logger = logging.getLogger("docdigitize.ocr")


@dataclass
class TextRegion:
    box: tuple[int, int, int, int]
    polygon: list[list[int]]
    text: str
    recognition_confidence: float
    detection_confidence: float | None = None
    engine: str = "unknown"
    alternatives: list[str] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        scores = [self.recognition_confidence]
        if self.detection_confidence is not None:
            scores.append(self.detection_confidence)
        return max(0.0, min(1.0, min(scores)))


def _rect_from_polygon(points: Iterable[Iterable[float]], width: int, height: int) -> tuple[int, int, int, int]:
    points = list(points)
    x0 = max(0, int(min(point[0] for point in points)))
    y0 = max(0, int(min(point[1] for point in points)))
    x1 = min(width, int(max(point[0] for point in points)))
    y1 = min(height, int(max(point[1] for point in points)))
    return x0, y0, x1, y1


def _polygon(points: Iterable[Iterable[float]]) -> list[list[int]]:
    return [[int(round(point[0])), int(round(point[1]))] for point in points]


def _payload(result) -> dict:
    data = getattr(result, "json", result)
    if callable(data):
        data = data()
    if isinstance(data, str):
        import json
        data = json.loads(data)
    if isinstance(data, dict) and isinstance(data.get("res"), dict):
        return data["res"]
    return data if isinstance(data, dict) else {}


class PaddleOCREngine:
    """PaddleOCR v3/v4 wrapper using local model inference only."""

    name = "paddleocr"

    def __init__(self) -> None:
        from paddleocr import PaddleOCR

        options = dict(
            lang="en",
            device="cpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            # PaddlePaddle 3.3's oneDNN executor has a Windows inference bug
            # with the OCRv6 models. Plain CPU mode is slower but reliable.
            enable_mkldnn=False,
        )
        try:
            self.reader = PaddleOCR(ocr_version="PP-OCRv6", **options)
        except (TypeError, ValueError):
            self.reader = PaddleOCR(ocr_version="PP-OCRv5", **options)

    def recognize(self, image: Image.Image) -> list[TextRegion]:
        array = np.asarray(image.convert("RGB"))
        results = self.reader.predict(array, text_rec_score_thresh=0.0)
        regions: list[TextRegion] = []
        for result in results:
            data = _payload(result)
            texts = list(data.get("rec_texts", []))
            scores = list(data.get("rec_scores", []))
            polygons = list(data.get("rec_polys", data.get("dt_polys", [])))
            detection_scores = list(data.get("dt_scores", []))
            for index, (text, score, points) in enumerate(zip(texts, scores, polygons)):
                text = str(text).strip()
                box = _rect_from_polygon(points, image.width, image.height)
                if not text or box[2] - box[0] < 3 or box[3] - box[1] < 3:
                    continue
                detection = float(detection_scores[index]) if index < len(detection_scores) else None
                regions.append(TextRegion(
                    box=box,
                    polygon=_polygon(points),
                    text=text,
                    recognition_confidence=float(score),
                    detection_confidence=detection,
                    engine=self.name,
                ))
        return regions


class EasyOCREngine:
    """CPU fallback that keeps the application usable without PaddlePaddle."""

    name = "easyocr"

    def __init__(self) -> None:
        import easyocr
        self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    def recognize(self, image: Image.Image) -> list[TextRegion]:
        rgb = image.convert("RGB")
        detections = self.reader.readtext(np.asarray(rgb), detail=1, paragraph=False)
        words: list[TextRegion] = []
        for points, text, score in detections:
            text = str(text).strip()
            box = _rect_from_polygon(points, rgb.width, rgb.height)
            if not text or box[2] - box[0] < 3 or box[3] - box[1] < 3:
                continue
            words.append(TextRegion(
                box=box,
                polygon=_polygon(points),
                text=text,
                recognition_confidence=float(score),
                engine=self.name,
            ))
        return _merge_words_into_lines(words)

    def recognize_crop(self, image: Image.Image) -> str:
        results = self.reader.readtext(np.asarray(image.convert("RGB")), detail=1, paragraph=True)
        return " ".join(str(item[1]).strip() for item in results if str(item[1]).strip())


def _merge_words_into_lines(words: list[TextRegion]) -> list[TextRegion]:
    """Turn word detections into readable line blocks without merging columns."""
    rows: list[list[TextRegion]] = []
    for word in sorted(words, key=lambda item: ((item.box[1] + item.box[3]) / 2, item.box[0])):
        middle = (word.box[1] + word.box[3]) / 2
        height = max(1, word.box[3] - word.box[1])
        for row in rows:
            row_middle = sum((item.box[1] + item.box[3]) / 2 for item in row) / len(row)
            row_height = sum(item.box[3] - item.box[1] for item in row) / len(row)
            if abs(middle - row_middle) <= min(height, row_height) * 0.55:
                row.append(word)
                break
        else:
            rows.append([word])

    lines: list[TextRegion] = []
    for row in rows:
        row.sort(key=lambda item: item.box[0])
        current: list[TextRegion] = []
        for word in row:
            if current:
                average_height = sum(item.box[3] - item.box[1] for item in current) / len(current)
                if word.box[0] - current[-1].box[2] > max(35, average_height * 2.2):
                    lines.append(_join_regions(current))
                    current = []
            current.append(word)
        if current:
            lines.append(_join_regions(current))
    return reading_order(lines)


def _join_regions(regions: list[TextRegion]) -> TextRegion:
    x0 = min(item.box[0] for item in regions)
    y0 = min(item.box[1] for item in regions)
    x1 = max(item.box[2] for item in regions)
    y1 = max(item.box[3] for item in regions)
    return TextRegion(
        box=(x0, y0, x1, y1),
        polygon=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        text=" ".join(item.text for item in regions),
        recognition_confidence=sum(item.recognition_confidence for item in regions) / len(regions),
        engine=regions[0].engine,
    )


def reading_order(regions: list[TextRegion]) -> list[TextRegion]:
    """Stable top-to-bottom order with left-to-right ordering on shared rows."""
    return sorted(regions, key=lambda item: (
        (item.box[1] + item.box[3]) / 2,
        item.box[0],
    ))


class LocalOCREngine:
    def __init__(self) -> None:
        self.primary = None
        self.secondary: EasyOCREngine | None = None
        self.initialization_warning: str | None = None

    @property
    def loaded_backend(self) -> str | None:
        return getattr(self.primary, "name", None)

    def _load(self):
        if self.primary is not None:
            return self.primary
        try:
            self.primary = PaddleOCREngine()
        except Exception as exc:
            logger.warning("PaddleOCR unavailable; using EasyOCR: %s", exc)
            self.initialization_warning = "PaddleOCR could not load; local EasyOCR fallback was used."
            self.primary = EasyOCREngine()
        return self.primary

    def recognize(self, image: Image.Image) -> list[TextRegion]:
        primary = self._load()
        try:
            regions = reading_order(primary.recognize(image))
        except Exception as exc:
            if primary.name != "paddleocr":
                raise
            logger.exception("PaddleOCR failed while processing a page; using EasyOCR")
            self.initialization_warning = (
                "PaddleOCR failed on this computer; local EasyOCR fallback was used. "
                f"Technical detail: {type(exc).__name__}."
            )
            self.primary = EasyOCREngine()
            primary = self.primary
            regions = reading_order(primary.recognize(image))
        if primary.name != "paddleocr":
            return regions

        uncertain = [item for item in regions if item.confidence < 0.82][:20]
        if not uncertain:
            return regions
        try:
            if self.secondary is None:
                self.secondary = EasyOCREngine()
            for item in uncertain:
                x0, y0, x1, y1 = item.box
                pad = max(3, round((y1 - y0) * 0.15))
                crop = image.crop((max(0, x0 - pad), max(0, y0 - pad),
                                   min(image.width, x1 + pad), min(image.height, y1 + pad)))
                alternate = self.secondary.recognize_crop(crop)
                if alternate and alternate.casefold() != item.text.casefold():
                    similarity = SequenceMatcher(None, item.text.casefold(), alternate.casefold()).ratio()
                    if similarity < 0.97:
                        item.alternatives.append(alternate)
        except Exception as exc:
            logger.warning("Secondary OCR check failed: %s", exc)
        return regions


def _coverage(box: tuple[int, int, int, int], other: tuple[int, int, int, int]) -> float:
    x0, y0 = max(box[0], other[0]), max(box[1], other[1])
    x1, y1 = min(box[2], other[2]), min(box[3], other[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
    return intersection / area


def find_possible_missed_text(image: Image.Image, regions: list[TextRegion], limit: int = 20) -> list[tuple[int, int, int, int]]:
    """Find text-like ink not covered by OCR and return review prompts."""
    gray = np.asarray(image.convert("L"))
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 35, 13)
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, image.width // 16), 1)))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, image.height // 16))))
    ink = cv2.subtract(cv2.subtract(binary, horizontal), vertical)
    connected = cv2.morphologyEx(
        ink, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, image.width // 180), 3)),
        iterations=1,
    )
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[int, int, int, int]] = []
    image_area = image.width * image.height
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        box = (x, y, x + width, y + height)
        area = width * height
        if area < 100 or area > image_area * 0.08:
            continue
        if width < 14 or height < 8 or height > image.height * 0.12:
            continue
        # Scanner borders, ruled-line ends, and resampling fringes are common
        # at page edges and should never be presented as missed writing.
        edge_x = max(3, round(image.width * 0.008))
        edge_y = max(3, round(image.height * 0.008))
        if x <= edge_x or y <= edge_y or x + width >= image.width - edge_x or y + height >= image.height - edge_y:
            continue
        if width / max(1, height) > 45:
            continue
        ink_density = cv2.countNonZero(ink[y:y + height, x:x + width]) / max(1, area)
        if not 0.015 <= ink_density <= 0.75:
            continue
        # Detection polygons commonly begin a few pixels inside handwriting.
        # A modest overlap is therefore enough to treat this ink as covered.
        if any(_coverage(box, item.box) >= 0.20 for item in regions):
            continue
        candidates.append(box)
    candidates.sort(key=lambda box: ((box[1] + box[3]) / 2, box[0]))
    return candidates[:limit]
