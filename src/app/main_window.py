from __future__ import annotations

import json
import queue
import sqlite3
from pathlib import Path

import psutil
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.settings import load_config
from app.worker import BatchWorker, ModelTestWorker
from database.jobs import JobStore
from hardware.detect import detect_hardware
from image_processing.analysis import SUPPORTED_EXTENSIONS
from monitoring.folder_watcher import FolderWatcher


class MainWindow(QMainWindow):
    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root
        self.config_path = root / "config.json"
        self.config = load_config(root)
        for directory in (
            "INPUT",
            "OUTPUT",
            r"OUTPUT\01_ENHANCED",
            r"OUTPUT\02_UPSCALED",
            r"OUTPUT\03_PRINT_READY",
            r"OUTPUT\04_ILLUSTRATOR",
            r"OUTPUT\05_PDF",
            r"OUTPUT\06_REPORTS",
            r"OUTPUT\07_PREVIEWS",
            "MODELS",
            "ICC_PROFILES",
            "TEMP",
            "LOGS",
            "CONFIG",
            "SCRIPTS",
            "TESTS",
        ):
            (root / directory).mkdir(parents=True, exist_ok=True)
        self.worker: BatchWorker | None = None
        self.model_test_worker: ModelTestWorker | None = None
        self.tested_model_signature: tuple[str, int, int] | None = None
        self.tested_model_backend: str | None = None
        self.job_store = JobStore(root / "OUTPUT" / "jobs.sqlite3")
        self.watcher: FolderWatcher | None = None
        self.watch_candidates: dict[Path, tuple[int, int, int]] = {}
        self.setWindowTitle("AI Photo Production")
        self.resize(900, 760)
        self._build_ui()
        self._load_config()
        self._refresh_hardware()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_timer)
        self.timer.start(1000)
        if self.ai_enabled.isChecked() and Path(self.model_edit.text()).is_file():
            self._test_model()

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        folders = QGroupBox("Folders")
        folder_form = QFormLayout(folders)
        self.input_edit = QLineEdit()
        self.output_edit = QLineEdit()
        folder_form.addRow("Input", self._path_control(self.input_edit, True))
        folder_form.addRow("Output", self._path_control(self.output_edit, True))
        layout.addWidget(folders)

        settings = QGroupBox("Print preparation (optional)")
        form = QFormLayout(settings)
        self.ai_enabled = QPushButton("Enable ONNX super-resolution")
        self.ai_enabled.setCheckable(True)
        self.ai_enabled.setToolTip(
            "Requires a compatible local ONNX x2/x4 model with dynamic input dimensions."
        )
        self.model_edit = QLineEdit()
        self.ai_tile_size = QComboBox()
        self.ai_tile_size.addItem("Automatic (based on available RAM)", "auto")
        for tile_size in (128, 256, 384, 512):
            self.ai_tile_size.addItem(str(tile_size), tile_size)
        self.ai_tile_size.setToolTip(
            "Automatic mode selects a conservative tile size using currently available RAM "
            "for each image. Select a size to override it."
        )
        self.ai_tile_overlap = QSpinBox()
        self.ai_tile_overlap.setValue(16)
        self.ai_tile_size.currentTextChanged.connect(self._update_ai_overlap_limit)
        self._update_ai_overlap_limit(self.ai_tile_size.currentText())
        self.ai_backend = QComboBox()
        self.ai_backend.addItem("Automatic (Intel GPU, Vulkan if available, then CPU)", "auto")
        self.ai_backend.addItem("ONNX Runtime CPU", "onnx_cpu")
        self.ai_backend.addItem("OpenVINO Intel GPU (required)", "openvino_gpu")
        self.ai_backend.addItem("ONNX Runtime Vulkan (custom build required)", "onnx_vulkan")
        self.model_test_button = QPushButton("Test selected backend")
        self.model_test_status = QLabel("Model not tested")
        self.ppi = QSpinBox()
        self.ppi.setRange(72, 1200)
        self.ppi.setValue(700)
        self.icc_edit = QLineEdit()
        self.paper = QComboBox()
        self.paper.addItems(["A1 portrait", "A1 landscape", "Custom"])
        self.paper.currentTextChanged.connect(self._paper_changed)
        self.paper_width = QDoubleSpinBox()
        self.paper_width.setRange(50, 2000)
        self.paper_width.setValue(594)
        self.paper_width.setSuffix(" mm")
        self.paper_height = QDoubleSpinBox()
        self.paper_height.setRange(50, 2000)
        self.paper_height.setValue(841)
        self.paper_height.setSuffix(" mm")
        self.margin = QDoubleSpinBox()
        self.margin.setRange(0, 100)
        self.margin.setSuffix(" mm")
        self.bleed = QDoubleSpinBox()
        self.bleed.setRange(0, 50)
        self.bleed.setSuffix(" mm")
        self.crop_marks = QCheckBox("Add crop marks (requires at least 4 mm bleed)")
        self.aspect_mode = QComboBox()
        self.aspect_mode.addItems(["fit", "crop"])
        self.print_enabled = QPushButton("Enable print-ready CMYK TIFF")
        self.print_enabled.setCheckable(True)
        self.print_enabled.setToolTip("Requires a valid printer CMYK ICC profile and sufficient RAM/disk.")
        self.illustrator_enabled = QPushButton("Also create AI and PDF/X-4 in Illustrator")
        self.illustrator_enabled.setCheckable(True)
        self.illustrator_enabled.setToolTip(
            "Requires working Illustrator COM scripting and a PDF/X-4 preset for the PDF."
        )
        form.addRow("", self.ai_enabled)
        form.addRow("ONNX super-resolution model", self._path_control(self.model_edit, False, model=True))
        form.addRow("", self.model_test_button)
        form.addRow("Model test", self.model_test_status)
        form.addRow("AI tile size", self.ai_tile_size)
        form.addRow("Tile overlap", self.ai_tile_overlap)
        form.addRow("Inference backend", self.ai_backend)
        form.addRow("Target PPI", self.ppi)
        form.addRow("Printer ICC profile", self._path_control(self.icc_edit, False, icc=True))
        form.addRow("Paper", self.paper)
        form.addRow("Custom width", self.paper_width)
        form.addRow("Custom height", self.paper_height)
        form.addRow("Outer margin", self.margin)
        form.addRow("Bleed", self.bleed)
        form.addRow("", self.crop_marks)
        form.addRow("Aspect handling", self.aspect_mode)
        form.addRow("", self.print_enabled)
        form.addRow("", self.illustrator_enabled)
        layout.addWidget(settings)

        self.hardware_label = QLabel("Detecting hardware…")
        self.resource_label = QLabel()
        layout.addWidget(self.hardware_label)
        layout.addWidget(self.resource_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        layout.addWidget(self.progress)
        controls = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.cancel_button = QPushButton("Cancel")
        self.pause_button = QPushButton("Pause")
        self.resume_button = QPushButton("Resume")
        self.retry_button = QPushButton("Retry failed")
        self.monitor_checkbox = QCheckBox("Watch INPUT for new images")
        self.pause_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        controls.addWidget(self.monitor_checkbox)
        controls.addWidget(self.start_button)
        controls.addWidget(self.pause_button)
        controls.addWidget(self.resume_button)
        controls.addWidget(self.retry_button)
        controls.addWidget(self.cancel_button)
        layout.addLayout(controls)
        self.job_counts_label = QLabel()
        layout.addWidget(self.job_counts_label)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)
        self.setCentralWidget(central)
        self.start_button.clicked.connect(self._start)
        self.cancel_button.clicked.connect(self._cancel)
        self.pause_button.clicked.connect(self._pause)
        self.resume_button.clicked.connect(self._resume)
        self.retry_button.clicked.connect(self._retry_failed)
        self.model_test_button.clicked.connect(self._test_model)
        self.model_edit.textChanged.connect(self._model_changed)
        self.ai_backend.currentIndexChanged.connect(self._model_changed)

    def _path_control(
        self,
        edit: QLineEdit,
        directory: bool,
        icc: bool = False,
        model: bool = False,
    ) -> QWidget:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        button = QPushButton("Browse…")
        row.addWidget(edit, 1)
        row.addWidget(button)
        if icc:
            button.clicked.connect(
                lambda: self._choose_file(
                    edit, "ICC profiles (*.icc *.icm)", "Choose printer ICC profile"
                )
            )
        elif model:
            button.clicked.connect(
                lambda: self._choose_file(edit, "ONNX models (*.onnx)", "Choose ONNX model")
            )
        elif directory:
            button.clicked.connect(lambda: self._choose_directory(edit))
        return container

    @staticmethod
    def _choose_directory(edit: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(None, "Choose folder", edit.text())
        if path:
            edit.setText(path)

    @staticmethod
    def _choose_file(edit: QLineEdit, file_filter: str, title: str) -> None:
        path, _ = QFileDialog.getOpenFileName(None, title, edit.text(), file_filter)
        if path:
            edit.setText(path)

    def _load_config(self) -> None:
        input_path = Path(str(self.config.get("input_dir", "INPUT")))
        output_path = Path(str(self.config.get("output_dir", "OUTPUT")))
        self.input_edit.setText(str(input_path if input_path.is_absolute() else self.root / input_path))
        self.output_edit.setText(str(output_path if output_path.is_absolute() else self.root / output_path))
        self.ppi.setValue(int(self.config.get("target_ppi", 700)))
        self.ai_enabled.setChecked(bool(self.config.get("enable_ai_upscale", False)))
        model_path = Path(str(self.config.get("model_path", "")))
        self.model_edit.setText(
            str(model_path if model_path.is_absolute() else self.root / model_path)
        )
        saved_tile_size = self.config.get("ai_tile_size", "auto")
        if saved_tile_size != "auto":
            try:
                saved_tile_size = int(saved_tile_size)
            except (TypeError, ValueError):
                saved_tile_size = "auto"
        tile_index = self.ai_tile_size.findData(saved_tile_size)
        self.ai_tile_size.setCurrentIndex(tile_index if tile_index >= 0 else 0)
        self.ai_tile_overlap.setValue(int(self.config.get("ai_tile_overlap", 16)))
        saved_backend = str(self.config.get("ai_backend", "auto"))
        backend_index = self.ai_backend.findData(saved_backend)
        if backend_index >= 0:
            self.ai_backend.setCurrentIndex(backend_index)
        self.icc_edit.setText(str(self.config.get("icc_profile", "")))
        self.print_enabled.setChecked(bool(self.config.get("enable_print_output", False)))
        self.illustrator_enabled.setChecked(bool(self.config.get("enable_illustrator", False)))
        self.paper.setCurrentText(str(self.config.get("paper", "A1 portrait")))
        self.aspect_mode.setCurrentText(str(self.config.get("fit_mode", "fit")))
        self.paper_width.setValue(float(self.config.get("paper_width_mm", 594)))
        self.paper_height.setValue(float(self.config.get("paper_height_mm", 841)))
        self.margin.setValue(float(self.config.get("margin_mm", 0)))
        self.bleed.setValue(float(self.config.get("bleed_mm", 0)))
        self.crop_marks.setChecked(bool(self.config.get("crop_marks", False)))
        self._paper_changed(self.paper.currentText())
        self.monitor_checkbox.setChecked(bool(self.config.get("monitor_input", False)))

    def _paper_changed(self, paper: str) -> None:
        custom = paper == "Custom"
        self.paper_width.setEnabled(custom)
        self.paper_height.setEnabled(custom)

    def _refresh_hardware(self) -> None:
        info = detect_hardware(self.root)
        self.hardware = info
        self.hardware_label.setText(
            f"CPU: {info['cpu']} | GPU: {info['gpu']} | RAM: {info['ram_gb']} GB | "
            f"Illustrator: {info['illustrator']}"
        )
        self._refresh_resources()

    def _update_ai_overlap_limit(self, _tile_size: str) -> None:
        selected_tile_size = self.ai_tile_size.currentData()
        tile_size = 128 if selected_tile_size == "auto" else int(selected_tile_size)
        maximum = min(128, tile_size // 2 - 1)
        self.ai_tile_overlap.setRange(8, maximum)

    def _test_model(self) -> None:
        if self.model_test_worker and self.model_test_worker.isRunning():
            return
        model_path = Path(self.model_edit.text())
        if not model_path.is_file():
            self.model_test_status.setText("Choose an existing ONNX model.")
            return
        self.model_test_button.setEnabled(False)
        backend = str(self.ai_backend.currentData())
        self.model_test_status.setText(f"Loading model and testing {self.ai_backend.currentText()}…")
        self.model_test_worker = ModelTestWorker(model_path, backend)
        self.model_test_worker.completed.connect(self._model_test_done)
        self.model_test_worker.start()

    def _model_test_done(self, result: str) -> None:
        self.model_test_status.setText(result)
        self.model_test_button.setEnabled(True)
        self.tested_model_signature = None
        self.tested_model_backend = None
        worker = self.model_test_worker
        if "model test passed" in result.lower() and worker:
            try:
                stat = worker.model_path.stat()
                signature = (
                    str(worker.model_path.resolve()),
                    stat.st_size,
                    stat.st_mtime_ns,
                )
                if signature[0] == str(Path(self.model_edit.text()).resolve()):
                    self.tested_model_signature = signature
                    self.tested_model_backend = worker.backend
            except OSError:
                self.tested_model_signature = None
                self.tested_model_backend = None
        self.log.append(result)

    def _model_changed(self) -> None:
        self.tested_model_signature = None
        self.tested_model_backend = None
        self.model_test_status.setText("Model changed; test required.")

    def _model_signature(self, model_path: Path) -> tuple[str, int, int] | None:
        try:
            stat = model_path.stat()
        except OSError:
            return None
        return str(model_path.resolve()), stat.st_size, stat.st_mtime_ns

    def _refresh_resources(self) -> None:
        memory = psutil.virtual_memory()
        disk = self.root.drive or str(self.root)
        try:
            import shutil
            free_gb = shutil.disk_usage(disk).free / (1024**3)
        except OSError:
            free_gb = 0.0
        self.resource_label.setText(
            f"Available RAM: {memory.available / (1024**3):.1f} GB | "
            f"CPU: {psutil.cpu_percent():.0f}% | Free disk: {free_gb:.1f} GB"
        )

    def _on_timer(self) -> None:
        self._refresh_resources()
        self._poll_watcher()
        self._refresh_job_counts()

    def _start(self) -> None:
        input_dir = Path(self.input_edit.text())
        output_dir = Path(self.output_edit.text())
        if not input_dir.is_dir():
            QMessageBox.warning(self, "Input folder", "Choose an existing input folder.")
            return
        if self.worker and self.worker.isRunning():
            return
        options: dict[str, object] = {
            "enable_ai_upscale": self.ai_enabled.isChecked(),
            "model_path": self.model_edit.text(),
            "ai_tile_size": self.ai_tile_size.currentData(),
            "ai_tile_overlap": self.ai_tile_overlap.value(),
            "ai_backend": str(self.ai_backend.currentData()),
            "enable_print_output": self.print_enabled.isChecked(),
            "enable_illustrator": self.illustrator_enabled.isChecked(),
            "target_ppi": self.ppi.value(),
            "icc_profile": self.icc_edit.text(),
            "fit_mode": self.aspect_mode.currentText(),
            "paper": self.paper.currentText(),
            "paper_width_mm": self.paper_width.value(),
            "paper_height_mm": self.paper_height.value(),
            "margin_mm": self.margin.value(),
            "bleed_mm": self.bleed.value(),
            "crop_marks": self.crop_marks.isChecked(),
        }
        if options["enable_illustrator"] and not options["enable_print_output"]:
            QMessageBox.warning(
                self,
                "Print preparation required",
                "Enable CMYK TIFF output before requesting Illustrator documents.",
            )
            return
        if options["enable_ai_upscale"] and not Path(str(options["model_path"])).is_file():
            QMessageBox.warning(
                self,
                "ONNX model required",
                "Choose a local, compatible ONNX super-resolution model before enabling AI upscale.",
            )
            return
        model_signature = self._model_signature(Path(str(options["model_path"])))
        if options["enable_ai_upscale"] and (
            model_signature != self.tested_model_signature
            or str(self.ai_backend.currentData()) != self.tested_model_backend
        ):
            QMessageBox.warning(
                self,
                "Test the selected inference backend first",
                "Run the model test successfully for the selected model and backend before starting.",
            )
            return
        options["illustrator_path"] = self.hardware["illustrator"]
        self._save_config(input_dir, output_dir, options)
        try:
            monitor_mode = self.monitor_checkbox.isChecked()
            if monitor_mode:
                self.watcher = FolderWatcher(input_dir)
                self.watcher.start()
            for path in input_dir.iterdir():
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    if monitor_mode:
                        self.watch_candidates.setdefault(path, (-1, -1, 0))
                    else:
                        self.job_store.register(path)
        except Exception as exc:
            QMessageBox.critical(self, "Could not start", f"{type(exc).__name__}: {exc}")
            if self.watcher:
                try:
                    self.watcher.stop()
                except Exception as stop_exc:
                    self.log.append(
                        f"Watcher shutdown error: {type(stop_exc).__name__}: {stop_exc}"
                    )
            self.watcher = None
            self.watch_candidates.clear()
            return

        self.worker = BatchWorker(
            input_dir, output_dir, options, self.job_store, monitor_mode
        )
        self.worker.progress.connect(self.log.append)
        self.worker.item_done.connect(self._item_done)
        self.worker.batch_done.connect(self._batch_done)
        self.worker.waiting.connect(self._worker_waiting)
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.resume_button.setEnabled(False)
        self.retry_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setRange(0, 0)
        self.log.append(
            "Starting sequential batch and folder monitor…"
            if monitor_mode else "Starting sequential batch…"
        )
        self.worker.start()

    def _pause(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.pause()
            self.pause_button.setEnabled(False)
            self.resume_button.setEnabled(True)
            self.log.append("Pause requested; processing pauses after the current image.")

    def _resume(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.resume()
            self.pause_button.setEnabled(True)
            self.resume_button.setEnabled(False)
            self.log.append("Processing resumed.")

    def _cancel(self) -> None:
        if self.worker:
            self.worker.cancel()
            self.cancel_button.setEnabled(False)
            self.pause_button.setEnabled(False)
            self.resume_button.setEnabled(False)
            self.log.append("Cancellation requested; the current image will finish first.")

    def _item_done(self, filename: str, result: str) -> None:
        self.log.append(f"{filename or 'Batch'}: {result}")
        self._refresh_job_counts()

    def _worker_waiting(self) -> None:
        if self.monitor_checkbox.isChecked():
            self.log.append("Watching INPUT; waiting for a new stable image.")

    def _batch_done(self) -> None:
        if self.watcher:
            try:
                self.watcher.stop()
            except Exception as exc:
                self.log.append(f"Watcher shutdown error: {type(exc).__name__}: {exc}")
            self.watcher = None
        self.watch_candidates.clear()
        self.progress.setRange(0, 1)
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.retry_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.log.append("Batch finished.")
        self._refresh_job_counts()

    def _poll_watcher(self) -> None:
        if not self.watcher:
            return
        while True:
            try:
                path = self.watcher.events.get_nowait()
            except queue.Empty:
                break
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                self.watch_candidates.setdefault(path, (-1, -1, 0))

        for path, previous in list(self.watch_candidates.items()):
            try:
                stat = path.stat()
            except OSError:
                self.watch_candidates.pop(path, None)
                continue
            current = (stat.st_size, stat.st_mtime_ns)
            stable_count = previous[2] + 1 if previous[:2] == current else 0
            if stable_count >= 2:
                try:
                    job_id, created = self.job_store.register(path)
                    if created and self.worker and self.worker.isRunning():
                        self.worker.enqueue(job_id, path)
                        self.log.append(f"Queued new image: {path.name}")
                except (OSError, sqlite3.Error) as exc:
                    self.log.append(f"Could not queue {path.name}: {type(exc).__name__}: {exc}")
                self.watch_candidates.pop(path, None)
            else:
                self.watch_candidates[path] = (*current, stable_count)

    def _retry_failed(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        input_dir = Path(self.input_edit.text())
        try:
            retried = self.job_store.retry_failed(input_dir)
        except Exception as exc:
            QMessageBox.critical(self, "Retry failed", f"{type(exc).__name__}: {exc}")
            return
        self.log.append(f"Requeued {retried} failed or partial job(s).")
        self._refresh_job_counts()
        if retried:
            self._start()

    def _refresh_job_counts(self) -> None:
        try:
            counts = self.job_store.counts(Path(self.input_edit.text()))
        except (OSError, ValueError):
            return
        self.job_counts_label.setText(
            "Jobs — "
            + " | ".join(
                f"{name}: {counts.get(name, 0)}"
                for name in ("pending", "in_progress", "completed", "partial", "failed")
            )
        )

    def _save_config(self, input_dir: Path, output_dir: Path, options: dict[str, object]) -> None:
        self.config.update({
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "target_ppi": options["target_ppi"],
            "icc_profile": options["icc_profile"],
            "enable_print_output": options["enable_print_output"],
            "enable_illustrator": options["enable_illustrator"],
            "enable_ai_upscale": options["enable_ai_upscale"],
            "model_path": options["model_path"],
            "ai_tile_size": options["ai_tile_size"],
            "ai_tile_overlap": options["ai_tile_overlap"],
            "ai_backend": options["ai_backend"],
            "paper": options["paper"],
            "paper_width_mm": options["paper_width_mm"],
            "paper_height_mm": options["paper_height_mm"],
            "margin_mm": options["margin_mm"],
            "bleed_mm": options["bleed_mm"],
            "crop_marks": options["crop_marks"],
            "fit_mode": options["fit_mode"],
            "monitor_input": self.monitor_checkbox.isChecked(),
        })
        self.config_path.write_text(json.dumps(self.config, indent=2), encoding="utf-8")

    def closeEvent(self, event) -> None:
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Processing", "Wait for the current batch to finish or cancel it.")
            event.ignore()
            return
        if self.model_test_worker and self.model_test_worker.isRunning():
            QMessageBox.warning(self, "Model test", "Wait for the CPU model test to finish.")
            event.ignore()
            return
        if self.watcher:
            self.watcher.stop()
            self.watcher = None
        event.accept()
