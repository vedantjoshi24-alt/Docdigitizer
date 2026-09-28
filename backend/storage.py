"""Small local job store. Documents never leave the machine."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from threading import Lock
import uuid

from PIL import Image


def default_data_dir() -> Path:
    configured = os.getenv("DOCDIGITIZE_DATA_DIR")
    if configured:
        return Path(configured)
    local = os.getenv("LOCALAPPDATA")
    if local:
        return Path(local) / "DocDigitize" / "data"
    return Path.home() / ".docdigitize"


class JobStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or default_data_dir()).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    @staticmethod
    def _valid_job_id(job_id: str) -> str:
        try:
            return str(uuid.UUID(job_id))
        except ValueError as exc:
            raise FileNotFoundError(job_id) from exc

    def create(self, pages: list[Image.Image], result: dict) -> dict:
        job_id = str(uuid.uuid4())
        folder = self.root / job_id
        folder.mkdir(parents=True)
        for index, page in enumerate(pages, 1):
            page.save(folder / f"page-{index}.png", format="PNG", optimize=True)
        result["job_id"] = job_id
        self.save(job_id, result)
        return result

    def load(self, job_id: str) -> dict:
        folder = self.root / self._valid_job_id(job_id)
        with (folder / "result.json").open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def save(self, job_id: str, result: dict) -> None:
        folder = self.root / self._valid_job_id(job_id)
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / "result.json"
        with self._lock:
            descriptor, temporary = tempfile.mkstemp(prefix="result-", suffix=".json", dir=folder)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    json.dump(result, handle, ensure_ascii=False, indent=2)
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)

    def page_path(self, job_id: str, page_number: int) -> Path:
        folder = self.root / self._valid_job_id(job_id)
        path = folder / f"page-{int(page_number)}.png"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
