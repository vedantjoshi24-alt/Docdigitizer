"""Non-destructive document preparation and input quality checks."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps


@dataclass
class PreparedPage:
    original: Image.Image
    image: Image.Image
    scale_x: float
    scale_y: float
    quality: dict


def _quality(image: Image.Image) -> dict:
    gray = np.asarray(image.convert("L"))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    glare = float(np.mean(gray >= 250))
    darkness = float(np.mean(gray <= 35))
    findings: list[dict] = []
    if min(image.width, image.height) < 900:
        findings.append({"code": "low_resolution", "severity": "check",
                         "message": "The page is small; a closer photo may recover more characters."})
    if blur_score < 55:
        findings.append({"code": "blur", "severity": "check",
                         "message": "The page appears blurred. Fine print and handwriting may be unreliable."})
    if glare > 0.12:
        findings.append({"code": "glare", "severity": "check",
                         "message": "Bright glare may hide part of the page."})
    if darkness > 0.20:
        findings.append({"code": "dark", "severity": "check",
                         "message": "The image is very dark; use more even lighting if possible."})
    return {
        "blur_score": round(blur_score, 1),
        "glare_fraction": round(glare, 3),
        "dark_fraction": round(darkness, 3),
        "findings": findings,
    }


def prepare_page(image: Image.Image, max_dimension: int = 2800) -> PreparedPage:
    """Apply EXIF orientation and resize for OCR while preserving coordinates.

    Perspective warping and aggressive thresholding are deliberately excluded:
    both can erase thin pen strokes and would break the overlay-to-original map.
    """
    original = ImageOps.exif_transpose(image).convert("RGB")
    analysis = original.copy()
    if max(analysis.size) > max_dimension:
        ratio = max_dimension / max(analysis.size)
        analysis = analysis.resize((round(analysis.width * ratio), round(analysis.height * ratio)), Image.Resampling.LANCZOS)
    elif max(analysis.size) < 1600:
        # OCR benefits from larger character strokes on screenshots and camera
        # crops. Limit enlargement so a tiny unusable image cannot consume
        # excessive memory; its low-resolution warning remains visible.
        ratio = min(3.0, 1600 / max(analysis.size))
        analysis = analysis.resize((round(analysis.width * ratio), round(analysis.height * ratio)), Image.Resampling.LANCZOS)
    return PreparedPage(
        original=original,
        image=analysis,
        scale_x=original.width / analysis.width,
        scale_y=original.height / analysis.height,
        quality=_quality(original),
    )
