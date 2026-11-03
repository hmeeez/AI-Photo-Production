from __future__ import annotations

import queue
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer


class _ImageEventHandler(FileSystemEventHandler):
    def __init__(self, events: queue.Queue[Path]) -> None:
        self.events = events

    def on_created(self, event: FileSystemEvent) -> None:
        self._add_event(event)

    def on_modified(self, event: FileSystemEvent) -> None:
        self._add_event(event)

    def on_moved(self, event: FileSystemEvent) -> None:
        destination = getattr(event, "dest_path", None)
        if destination and not event.is_directory:
            self.events.put(Path(destination))

    def _add_event(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self.events.put(Path(event.src_path))


class FolderWatcher:
    def __init__(self, input_dir: Path) -> None:
        self.events: queue.Queue[Path] = queue.Queue()
        self.observer = Observer()
        self.observer.schedule(_ImageEventHandler(self.events), str(input_dir), recursive=False)

    def start(self) -> None:
        self.observer.start()

    def stop(self) -> None:
        self.observer.stop()
        self.observer.join(timeout=5)
        if self.observer.is_alive():
            raise RuntimeError("Input-folder watcher did not stop within 5 seconds.")
