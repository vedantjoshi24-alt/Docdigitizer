from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import pypdfium2 as pdfium
from PIL import Image


BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))

from exports import searchable_pdf, transcript_text  # noqa: E402
from ocr import TextRegion, reading_order  # noqa: E402
from storage import JobStore  # noqa: E402


def sample_job() -> dict:
    return {
        "filename": "sample.png",
        "pages": [{
            "page_number": 1,
            "width": 400,
            "height": 220,
            "blocks": [{
                "id": "p1-b1",
                "text": "Name: Vedant Joshi",
                "original_text": "Name: Vedant Joshi",
                "box": [30, 40, 280, 70],
                "status": "likely",
                "review_state": "verified",
            }],
        }],
    }


class ExportTests(unittest.TestCase):
    def test_searchable_pdf_keeps_corrected_text(self):
        with tempfile.TemporaryDirectory() as folder:
            page = Path(folder) / "page.png"
            Image.new("RGB", (400, 220), "white").save(page)
            data = searchable_pdf(sample_job(), [page])
            self.assertTrue(data.startswith(b"%PDF"))
            document = pdfium.PdfDocument(data)
            text = document[0].get_textpage().get_text_range()
            self.assertIn("Vedant Joshi", text)

    def test_plain_transcript_has_reading_order(self):
        job = sample_job()
        job["pages"][0]["blocks"].append({"text": "Roll No: 13"})
        self.assertEqual(transcript_text(job), "Name: Vedant Joshi\nRoll No: 13\n")


class StorageTests(unittest.TestCase):
    def test_job_roundtrip_keeps_page_and_json(self):
        with tempfile.TemporaryDirectory() as folder:
            store = JobStore(Path(folder))
            job = store.create([Image.new("RGB", (20, 10), "white")], sample_job())
            loaded = store.load(job["job_id"])
            self.assertEqual(loaded["filename"], "sample.png")
            self.assertTrue(store.page_path(job["job_id"], 1).is_file())
            json.dumps(loaded)


class OCRPostprocessingTests(unittest.TestCase):
    def test_reading_order_does_not_swap_different_height_lines(self):
        lower_wide = TextRegion((7, 87, 546, 119), [], "College", 0.98)
        upper_short = TextRegion((10, 66, 192, 92), [], "Branch", 0.93)
        self.assertEqual(
            [item.text for item in reading_order([lower_wide, upper_short])],
            ["Branch", "College"],
        )


if __name__ == "__main__":
    unittest.main()
