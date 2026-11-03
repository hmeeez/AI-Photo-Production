from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path

import psutil


def _powershell_value(expression: str) -> str:
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", expression],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "Unavailable"
    value = result.stdout.strip()
    return value or "Unavailable"


def detect_hardware(root: Path) -> dict[str, object]:
    cpu_name = _powershell_value(
        "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)"
    )
    gpu_name = _powershell_value(
        "(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join ', '"
    )
    illustrator_candidates = [
        Path(r"C:\Program Files\Adobe\Adobe Illustrator 2026\Support Files\Contents\Windows\Illustrator.exe"),
        Path(r"C:\Program Files\Adobe\Adobe Illustrator 2025\Support Files\Contents\Windows\Illustrator.exe"),
    ]
    illustrator = next((str(path) for path in illustrator_candidates if path.is_file()), "Not detected")
    disk = shutil.disk_usage(root)
    memory = psutil.virtual_memory()
    return {
        "cpu": cpu_name if cpu_name != "Unavailable" else platform.processor(),
        "logical_cores": psutil.cpu_count(logical=True) or 1,
        "physical_cores": psutil.cpu_count(logical=False) or 1,
        "gpu": gpu_name,
        "ram_gb": round(memory.total / (1024**3), 1),
        "available_ram_gb": round(memory.available / (1024**3), 1),
        "free_disk_gb": round(disk.free / (1024**3), 1),
        "illustrator": illustrator,
    }
