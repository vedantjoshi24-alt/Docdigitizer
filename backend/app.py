"""DocDigitize local API: page-faithful OCR, review, and export."""
from __future__ import annotations

from datetime import datetime, timezone
import io
import logging
import os
from pathlib import Path
import re
from threading import Lock

import pypdfium2 as pdfium
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from exports import json_bytes, review_html, searchable_pdf, transcript_text
from ocr import LocalOCREngine, TextRegion, find_possible_missed_text
from preprocessing import PreparedPage, prepare_page
from storage import JobStore


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("docdigitize")
PROJECT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_PDF_PAGES = 12

app = FastAPI(title="DocDigitize", version="2.0.0")
store = JobStore()
ocr_engine = LocalOCREngine()
processing_lock = Lock()


@app.middleware("http")
async def prevent_stale_frontend(request, call_next):
    response = await call_next(request)
    if request.url.path in {"/", "/index.html"}:
        response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "version": app.version,
        "processing": "local",
        "ocr_backend_loaded": ocr_engine.loaded_backend,
        "data_directory": str(store.root),
    }


def _pages(raw: bytes, content_type: str) -> list[Image.Image]:
    if content_type == "application/pdf" or raw[:5] == b"%PDF-":
        try:
            document = pdfium.PdfDocument(raw)
            if len(document) > MAX_PDF_PAGES:
                raise HTTPException(status_code=413, detail=f"PDFs are limited to {MAX_PDF_PAGES} pages per upload.")
            images: list[Image.Image] = []
            for index in range(len(document)):
                page = document[index]
                bitmap = page.render(scale=2.4)
                images.append(bitmap.to_pil().convert("RGB").copy())
                bitmap.close()
                page.close()
            document.close()
            return images
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Could not open this PDF.") from exc
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload a PNG, JPG, WEBP, or PDF document.")
    try:
        with Image.open(io.BytesIO(raw)) as source:
            return [source.copy()]
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Could not read this image file.") from exc


def _scale_box(box: tuple[int, int, int, int], prepared: PreparedPage) -> list[int]:
    return [
        round(box[0] * prepared.scale_x), round(box[1] * prepared.scale_y),
        round(box[2] * prepared.scale_x), round(box[3] * prepared.scale_y),
    ]


def _scale_polygon(polygon: list[list[int]], prepared: PreparedPage) -> list[list[int]]:
    return [[round(point[0] * prepared.scale_x), round(point[1] * prepared.scale_y)] for point in polygon]


def _block(region: TextRegion, prepared: PreparedPage, page_number: int, index: int) -> dict:
    confidence = round(region.confidence, 3)
    reasons: list[str] = []
    if region.recognition_confidence < 0.88:
        reasons.append("The recognizer reported uncertain characters.")
    if region.detection_confidence is not None and region.detection_confidence < 0.70:
        reasons.append("The text region itself was weakly detected.")
    if region.alternatives:
        reasons.append("A second local recognizer produced a different reading.")
    if min(prepared.original.size) < 500:
        reasons.append("The source is very small, so this line needs visual confirmation.")
    status = "likely" if confidence >= 0.88 and not reasons else "check"
    return {
        "id": f"p{page_number}-b{index}",
        "text": region.text,
        "original_text": region.text,
        "box": _scale_box(region.box, prepared),
        "polygon": _scale_polygon(region.polygon, prepared),
        "confidence": confidence,
        "recognition_confidence": round(region.recognition_confidence, 3),
        "detection_confidence": round(region.detection_confidence, 3) if region.detection_confidence is not None else None,
        "status": status,
        "reasons": reasons,
        "alternatives": region.alternatives,
        "engine": region.engine,
        "review_state": "unreviewed",
        "manual": False,
    }


def _missed_block(box: tuple[int, int, int, int], prepared: PreparedPage, page_number: int, index: int) -> dict:
    scaled = _scale_box(box, prepared)
    x0, y0, x1, y1 = scaled
    return {
        "id": f"p{page_number}-m{index}",
        "text": "",
        "original_text": "",
        "box": scaled,
        "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        "confidence": 0.0,
        "recognition_confidence": 0.0,
        "detection_confidence": None,
        "status": "missed",
        "reasons": ["This area contains text-like ink that was not covered by an OCR result."],
        "alternatives": [],
        "engine": "coverage-check",
        "review_state": "unreviewed",
        "manual": False,
    }


def _process_page(source: Image.Image, page_number: int) -> tuple[Image.Image, dict]:
    prepared = prepare_page(source)
    regions = ocr_engine.recognize(prepared.image)
    missed = find_possible_missed_text(prepared.image, regions)
    blocks = [_block(region, prepared, page_number, index) for index, region in enumerate(regions, 1)]
    blocks.extend(_missed_block(box, prepared, page_number, index) for index, box in enumerate(missed, 1))
    warnings = [item["message"] for item in prepared.quality["findings"]]
    if not regions:
        warnings.append("No text was recognized on this page. Review every highlighted possible-text area.")
    return prepared.original, {
        "page_number": page_number,
        "width": prepared.original.width,
        "height": prepared.original.height,
        "quality": prepared.quality,
        "blocks": blocks,
        "warnings": warnings,
    }


def _summary(pages: list[dict]) -> dict:
    blocks = [block for page in pages for block in page["blocks"]]
    return {
        "pages": len(pages),
        "blocks": len(blocks),
        "likely": sum(block["status"] == "likely" for block in blocks),
        "needs_review": sum(block["status"] in {"check", "missed"} and block.get("review_state") not in {"verified", "corrected"} for block in blocks),
        "possible_missed": sum(block["status"] == "missed" and block.get("review_state") not in {"verified", "corrected"} for block in blocks),
        "verified": sum(block.get("review_state") in {"verified", "corrected"} for block in blocks),
    }


def _process_document(images: list[Image.Image], filename: str) -> dict:
    originals: list[Image.Image] = []
    pages: list[dict] = []
    with processing_lock:
        for page_number, image in enumerate(images, 1):
            original, page = _process_page(image, page_number)
            originals.append(original)
            pages.append(page)
    result = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "filename": filename,
        "processing": "local",
        "ocr_backend": ocr_engine.loaded_backend,
        "backend_warning": ocr_engine.initialization_warning,
        "pages": pages,
        "summary": _summary(pages),
    }
    return store.create(originals, result)


@app.post("/api/digitize")
async def digitize(file: UploadFile = File(...)):
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded document is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The upload limit is 20 MB.")
    images = _pages(raw, file.content_type or "")
    try:
        result = await run_in_threadpool(_process_document, images, file.filename or "document")
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Document processing failed")
        raise HTTPException(status_code=500, detail="Local document processing failed. See the launcher window for details.") from exc


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    try:
        return store.load(job_id)
    except (FileNotFoundError, OSError):
        raise HTTPException(status_code=404, detail="Document job not found.")


def _validated_update(job_id: str, payload: dict) -> dict:
    existing = store.load(job_id)
    incoming_pages = payload.get("pages")
    if not isinstance(incoming_pages, list) or len(incoming_pages) != len(existing["pages"]):
        raise HTTPException(status_code=422, detail="Page count cannot be changed.")
    for source, incoming in zip(existing["pages"], incoming_pages):
        blocks = incoming.get("blocks")
        if not isinstance(blocks, list) or len(blocks) > 5000:
            raise HTTPException(status_code=422, detail="Invalid text block list.")
        cleaned = []
        for index, block in enumerate(blocks):
            text = str(block.get("text", ""))[:10000]
            box = block.get("box")
            if not isinstance(box, list) or len(box) != 4:
                raise HTTPException(status_code=422, detail="Every text block needs a four-value box.")
            x0, y0, x1, y1 = [max(0, int(value)) for value in box]
            x0, x1 = min(x0, source["width"]), min(x1, source["width"])
            y0, y1 = min(y0, source["height"]), min(y1, source["height"])
            if x1 <= x0 or y1 <= y0:
                raise HTTPException(status_code=422, detail="A text block has an empty box.")
            prior = next((item for item in source["blocks"] if item.get("id") == block.get("id")), {})
            cleaned.append({
                **prior,
                "id": str(block.get("id") or f"p{source['page_number']}-manual-{index}"),
                "text": text,
                "original_text": str(block.get("original_text", prior.get("original_text", "")))[:10000],
                "box": [x0, y0, x1, y1],
                "polygon": block.get("polygon", [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]),
                "confidence": block.get("confidence", prior.get("confidence")),
                "recognition_confidence": block.get("recognition_confidence", prior.get("recognition_confidence")),
                "detection_confidence": block.get("detection_confidence", prior.get("detection_confidence")),
                "status": str(block.get("status", prior.get("status", "check"))),
                "reasons": list(block.get("reasons", prior.get("reasons", [])))[:20],
                "alternatives": list(block.get("alternatives", prior.get("alternatives", [])))[:10],
                "engine": str(block.get("engine", prior.get("engine", "manual"))),
                "review_state": str(block.get("review_state", "unreviewed")),
                "manual": bool(block.get("manual", prior.get("manual", False))),
            })
        source["blocks"] = cleaned
    existing["summary"] = _summary(existing["pages"])
    store.save(job_id, existing)
    return existing


@app.put("/api/jobs/{job_id}")
def update_job(job_id: str, payload: dict = Body(...)):
    try:
        return _validated_update(job_id, payload)
    except (FileNotFoundError, OSError):
        raise HTTPException(status_code=404, detail="Document job not found.")


@app.get("/api/jobs/{job_id}/pages/{page_number}.png")
def page_image(job_id: str, page_number: int):
    try:
        return FileResponse(store.page_path(job_id, page_number), media_type="image/png")
    except (FileNotFoundError, OSError):
        raise HTTPException(status_code=404, detail="Page image not found.")


@app.get("/api/jobs/{job_id}/export/{kind}")
def export_job(job_id: str, kind: str):
    try:
        job = store.load(job_id)
        source_stem = Path(job.get("filename", "document")).stem or "document"
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", source_stem).strip("-.") or "document"
        if kind == "pdf":
            paths = [store.page_path(job_id, page["page_number"]) for page in job["pages"]]
            content, media, suffix = searchable_pdf(job, paths), "application/pdf", "searchable.pdf"
        elif kind == "txt":
            content, media, suffix = transcript_text(job).encode("utf-8"), "text/plain; charset=utf-8", "transcript.txt"
        elif kind == "json":
            content, media, suffix = json_bytes(job), "application/json", "result.json"
        elif kind == "html":
            content, media, suffix = review_html(job), "text/html; charset=utf-8", "transcript.html"
        else:
            raise HTTPException(status_code=404, detail="Unknown export format.")
        headers = {"Content-Disposition": f'attachment; filename="{safe_stem}-{suffix}"'}
        return Response(content, media_type=media, headers=headers)
    except HTTPException:
        raise
    except (FileNotFoundError, OSError):
        raise HTTPException(status_code=404, detail="Document job not found.")


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
