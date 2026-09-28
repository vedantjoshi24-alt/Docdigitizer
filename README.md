# DocDigitize 2

DocDigitize turns English document photos and PDFs into a page-faithful,
reviewable digital copy. Processing runs on the local computer. It does not
send documents to a paid or cloud OCR service.

## Run on Windows

Open PowerShell in this folder and run:

```powershell
./run_windows.ps1
```

Then open <http://127.0.0.1:8000>. Keep the PowerShell window open while the
website is in use.

The launcher creates a private Python 3.12 environment under
`%LOCALAPPDATA%\DocDigitize`. On first use it installs the free OCR packages.
The first document also downloads the selected open model files. Those files
are cached locally for later runs.

## What the result contains

- The original page image, unchanged except for EXIF orientation correction.
- An ordered transcript with a box linking every reading to the source page.
- Model scores, disagreement flags, and possible missed-text regions.
- An editor for corrections, verification, reading order, and manually added
  regions.
- Searchable PDF, TXT, HTML, and full JSON downloads.

Jobs and corrections are saved under `%LOCALAPPDATA%\DocDigitize\data`. Delete
that folder when you want to remove locally stored documents.

## Recognition pipeline

1. Validate the upload and render each PDF page to an image.
2. Measure resolution, blur, glare, and darkness without destructively
   thresholding the source.
3. Run PaddleOCR locally for text detection and recognition. Its mixed English
   model handles printed and handwritten regions in one page.
4. Cross-check uncertain PaddleOCR crops with local EasyOCR. If PaddleOCR is
   unavailable, EasyOCR becomes the offline fallback.
5. Search for text-like ink not covered by any OCR box. These areas are shown
   as possible omissions, never silently discarded.
6. Preserve coordinates and corrections in a local job record. Exports use the
   corrected transcript.

The old FUNSD/LayoutLM checkpoint is retained in `training/` for research, but
it is no longer in the primary pipeline. FUNSD classifies recognized tokens as
questions, answers, or headers; it does not recognize missing handwriting.

## Confidence and review

OCR scores are model signals, not guarantees of correctness. A line is shown
as **Check** when its reading score is low, its detection is weak, or a second
recognizer disagrees. Ink outside OCR coverage is shown as **Possible missed
text**. A user can transcribe it or mark it as not text.

No system can infer characters that are invisible in the photograph. For that
reason, the input-quality checks recommend recapturing severely blurred, dark,
small, or glare-covered pages. A document becomes a dependable final copy when
all flagged areas have been reviewed.

## API

- `GET /api/health` — runtime status.
- `POST /api/digitize` — multipart upload in the `file` field.
- `GET /api/jobs/{job_id}` — saved transcript and page metadata.
- `PUT /api/jobs/{job_id}` — save edited page blocks.
- `GET /api/jobs/{job_id}/pages/{page}.png` — original page image.
- `GET /api/jobs/{job_id}/export/{pdf|txt|html|json}` — corrected export.

## Current validation requirement

Accuracy thresholds must be measured using original, full-resolution examples
of the documents this project is expected to read. Screenshots of the webpage
are useful for diagnosing the old failure but are not suitable ground-truth OCR
inputs. Keep a private validation set with printed, handwritten, and mixed
pages plus exact transcripts; do not commit personal documents to source
control.
