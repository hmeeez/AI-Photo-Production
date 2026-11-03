from __future__ import annotations

import argparse
import io
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageCms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.pipeline import process_image  # noqa: E402
from illustrator.automation import _validate_pdf_x4  # noqa: E402
from print_preparation.dimensions import A1_MM, target_pixels  # noqa: E402


def _resolution(tag: tifffile.TiffTag) -> int:
    value = tag.value
    return round(value[0] / value[1]) if isinstance(value, tuple) else round(value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a real-model A1 CMYK pipeline smoke test."
    )
    parser.add_argument("--icc-profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "TEMP" / "integration-smoke")
    parser.add_argument("--model-path", type=Path, default=ROOT / "MODELS" / "real_esrgan_x4.onnx")
    parser.add_argument("--illustrator-path", type=Path)
    parser.add_argument("--illustrator", action="store_true")
    parser.add_argument("--backend", choices=("auto", "onnx_cpu", "openvino_gpu"), default="auto")
    parser.add_argument("--ppi", type=int, default=72)
    args = parser.parse_args()

    if not args.icc_profile.is_file():
        parser.error(f"ICC profile does not exist: {args.icc_profile}")
    if not args.model_path.is_file():
        parser.error(f"ONNX model does not exist: {args.model_path}")
    illustrator_path = args.illustrator_path
    if args.illustrator and illustrator_path is None:
        candidates = (
            Path(r"C:\Program Files\Adobe\Adobe Illustrator 2026\Support Files\Contents\Windows\Illustrator.exe"),
            Path(r"C:\Program Files\Adobe\Adobe Illustrator 2025\Support Files\Contents\Windows\Illustrator.exe"),
        )
        illustrator_path = next((path for path in candidates if path.is_file()), None)
        if illustrator_path is None:
            parser.error("Illustrator was requested, but no supported installation was found.")
    if illustrator_path is not None and not illustrator_path.is_file():
        parser.error(f"Illustrator executable does not exist: {illustrator_path}")
    if not 72 <= args.ppi <= 1200:
        parser.error("PPI must be between 72 and 1200.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_root = args.output_dir.resolve() / run_id
    input_dir = run_root / "INPUT"
    output_dir = run_root / "OUTPUT"
    input_dir.mkdir(parents=True)
    source_path = input_dir / "smoke_source.png"
    pixels = np.random.default_rng(42).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(source_path)

    options: dict[str, object] = {
        "enable_ai_upscale": True,
        "model_path": str(args.model_path.resolve()),
        "ai_backend": args.backend,
        "ai_tile_size": 256,
        "ai_tile_overlap": 16,
        "enable_print_output": True,
        "target_ppi": args.ppi,
        "icc_profile": str(args.icc_profile.resolve()),
        "paper": "A1 portrait",
        "fit_mode": "fit",
        "enable_illustrator": illustrator_path is not None,
    }
    if illustrator_path is not None:
        options["illustrator_path"] = str(illustrator_path.resolve())

    report = process_image(source_path, output_dir, options)
    if report["status"] != "completed":
        raise RuntimeError(
            f"Smoke test failed: {json.dumps(report.get('errors', []), ensure_ascii=False)}"
        )
    outputs = report["outputs"]
    for key in ("enhanced", "upscaled", "preview", "print_ready"):
        path = Path(str(outputs[key]))
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty smoke-test output: {key}")

    with Image.open(outputs["upscaled"]) as image:
        if image.size != (256, 192):
            raise RuntimeError(f"Unexpected AI output dimensions: {image.size}")
    expected_size = target_pixels(*A1_MM, args.ppi)
    expected_profile = ImageCms.ImageCmsProfile(
        io.BytesIO(args.icc_profile.read_bytes())
    ).tobytes()
    with tifffile.TiffFile(outputs["print_ready"]) as tiff:
        page = tiff.pages[0]
        if (
            page.imagewidth != expected_size[0]
            or page.imagelength != expected_size[1]
            or page.samplesperpixel != 4
            or page.photometric != tifffile.PHOTOMETRIC.SEPARATED
        ):
            raise RuntimeError("The A1 print TIFF has unexpected dimensions or color mode.")
        if (
            _resolution(page.tags["XResolution"]) != args.ppi
            or _resolution(page.tags["YResolution"]) != args.ppi
            or int(page.tags["ResolutionUnit"].value) != 2
        ):
            raise RuntimeError("The A1 print TIFF has unexpected resolution metadata.")
        profile_tag = page.tags.get(34675)
        if profile_tag is None or profile_tag.value != expected_profile:
            raise RuntimeError("The A1 print TIFF is missing the selected printer ICC profile.")

    if illustrator_path is not None:
        for key in ("illustrator", "pdf_x4"):
            path = Path(str(outputs.get(key, "")))
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Illustrator smoke-test output is missing: {key}")
        valid, message = _validate_pdf_x4(
            Path(str(outputs["pdf_x4"])),
            A1_MM[0] / 25.4 * 72,
            A1_MM[1] / 25.4 * 72,
        )
        if not valid:
            raise RuntimeError(f"PDF/X-4 validation failed: {message}")

    print(json.dumps({
        "status": report["status"],
        "backend": report["stages"]["ai_upscaling"]["backend"],
        "ai_dimensions": report["stages"]["ai_upscaling"]["dimensions"],
        "print_dimensions": report["print"]["pixel_dimensions"],
        "ppi": args.ppi,
        "illustrator": illustrator_path is not None,
        "output_directory": str(run_root),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
