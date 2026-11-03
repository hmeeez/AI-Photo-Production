from __future__ import annotations

import queue
import threading
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from ai.super_resolution import prepare_inference
from core.pipeline import process_image


class BatchWorker(QThread):
    progress = Signal(str)
    item_done = Signal(str, str)
    batch_done = Signal()
    waiting = Signal()

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        options: dict[str, object],
        job_store,
        monitor_mode: bool,
    ) -> None:
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.options = options
        self.job_store = job_store
        self.monitor_mode = monitor_mode
        self.jobs: queue.Queue[tuple[int, Path]] = queue.Queue()
        self._cancel_requested = False
        self._paused = threading.Event()
        self._paused.set()

    def enqueue(self, job_id: int, path: Path) -> None:
        self.jobs.put((job_id, path))

    def pause(self) -> None:
        self._paused.clear()

    def resume(self) -> None:
        self._paused.set()

    def cancel(self) -> None:
        self._cancel_requested = True
        self._paused.set()

    def run(self) -> None:
        try:
            runtime = None
            runtime_error = None
            if bool(self.options.get("enable_ai_upscale")):
                try:
                    runtime = prepare_inference(
                        Path(str(self.options.get("model_path") or "")),
                        str(self.options.get("ai_backend", "auto")),
                    )
                    self.progress.emit(
                        f"Loaded {runtime.backend} model once for this sequential batch."
                    )
                except Exception as exc:
                    runtime_error = f"{type(exc).__name__}: {exc}"
            for job_id, path in self.job_store.pending(self.input_dir):
                self.enqueue(job_id, path)
            processed = 0
            announced_wait = False
            while not self._cancel_requested:
                self._paused.wait()
                if self._cancel_requested:
                    break
                try:
                    job_id, path = self.jobs.get(timeout=0.5)
                except queue.Empty:
                    if self.monitor_mode and not announced_wait:
                        self.waiting.emit()
                        announced_wait = True
                    if self.monitor_mode:
                        continue
                    break
                announced_wait = False
                if self._cancel_requested:
                    break
                if not path.is_file():
                    self.job_store.set_status(job_id, "failed", error="Source file disappeared.")
                    self.item_done.emit(path.name, "failed: source file disappeared")
                    continue
                self.job_store.set_status(job_id, "in_progress")
                if runtime_error:
                    self.job_store.set_status(job_id, "failed", error=runtime_error)
                    self.item_done.emit(path.name, f"failed: AI model setup error: {runtime_error}")
                    continue
                report = process_image(
                    path,
                    self.output_dir,
                    self.options,
                    self.progress.emit,
                    runtime,
                )
                status = str(report["status"])
                error = "; ".join(str(item) for item in report.get("errors", [])) or None
                self.job_store.set_status(
                    job_id,
                    status if status in {"completed", "partial"} else "failed",
                    str(report["report_path"]),
                    error,
                )
                processed += 1
                self.item_done.emit(path.name, status)
            if not self.monitor_mode and processed == 0:
                self.item_done.emit("", "No pending photographs were found in INPUT.")
        except Exception as exc:
            self.item_done.emit("", f"Batch failed: {type(exc).__name__}: {exc}")
        finally:
            self.batch_done.emit()


class ModelTestWorker(QThread):
    completed = Signal(str)

    def __init__(self, model_path: Path, backend: str) -> None:
        super().__init__()
        self.model_path = model_path
        self.backend = backend

    def run(self) -> None:
        try:
            runtime = prepare_inference(self.model_path, self.backend)
            message = (
                f"{runtime.backend} model test passed — {runtime.scale}x, "
                f"{runtime.input_name} → {runtime.output_name}."
            )
            if runtime.fallback_reason:
                message += f" CPU fallback: {runtime.fallback_reason}"
            self.completed.emit(message)
        except Exception as exc:
            self.completed.emit(f"Model test failed — {type(exc).__name__}: {exc}")
