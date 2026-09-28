"""Exports created from the user's corrected local transcript."""
from __future__ import annotations

import html
import io
import json
from pathlib import Path

from PIL import Image
from reportlab.pdfgen import canvas


def transcript_text(job: dict) -> str:
    output: list[str] = []
    for page in job.get("pages", []):
        if len(job.get("pages", [])) > 1:
            output.append(f"--- Page {page['page_number']} ---")
        output.extend(block.get("text", "") for block in page.get("blocks", []) if block.get("text", "").strip())
        output.append("")
    return "\n".join(output).rstrip() + "\n"


def json_bytes(job: dict) -> bytes:
    return json.dumps(job, ensure_ascii=False, indent=2).encode("utf-8")


def searchable_pdf(job: dict, page_paths: list[Path]) -> bytes:
    stream = io.BytesIO()
    pdf = canvas.Canvas(stream, pageCompression=1)
    for page, path in zip(job.get("pages", []), page_paths):
        with Image.open(path) as source:
            width, height = source.size
        pdf.setPageSize((width, height))
        pdf.drawImage(str(path), 0, 0, width=width, height=height, preserveAspectRatio=True)
        for block in page.get("blocks", []):
            text = str(block.get("text", "")).strip()
            box = block.get("box", [])
            if not text or len(box) != 4:
                continue
            x0, y0, x1, y1 = [float(value) for value in box]
            font_size = max(4.0, (y1 - y0) * 0.78)
            text_object = pdf.beginText()
            text_object.setTextOrigin(x0, max(0, height - y1))
            text_object.setFont("Helvetica", font_size)
            text_object.setTextRenderMode(3)
            measured = pdf.stringWidth(text, "Helvetica", font_size)
            if measured > 0:
                text_object.setHorizScale(max(25, min(400, (x1 - x0) / measured * 100)))
            text_object.textLine(text)
            pdf.drawText(text_object)
        pdf.showPage()
    pdf.save()
    return stream.getvalue()


def review_html(job: dict) -> bytes:
    sections = []
    for page in job.get("pages", []):
        lines = "\n".join(
            f'<li class="{html.escape(block.get("review_state", "unreviewed") if block.get("review_state") in {"verified", "corrected"} else block.get("status", "check"))}">'
            f'{html.escape(block.get("text", "") or "[possible missed text]")}</li>'
            for block in page.get("blocks", [])
        )
        sections.append(f"<h2>Page {page['page_number']}</h2><ol>{lines}</ol>")
    body = "".join(sections)
    document = f"""<!doctype html><meta charset=\"utf-8\"><title>DocDigitize transcript</title>
<style>body{{font:16px/1.5 system-ui;max-width:850px;margin:40px auto;padding:0 20px}}li{{margin:.45em 0}}.check,.missed{{background:#fff2bf}}.corrected,.verified{{background:#dff7e8}}</style>
<h1>DocDigitize transcript</h1>{body}"""
    return document.encode("utf-8")
