from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow


def main() -> int:
    application = QApplication(sys.argv)
    application.setApplicationName("AI Photo Production")
    window = MainWindow(Path(__file__).resolve().parents[1])
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
