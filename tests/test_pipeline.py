import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.pipeline import _resolve_print_layout, _validate_print_tiff, process_image


class PipelineTests(unittest.TestCase):
    def test_resolves_custom_print_layout(self) -> None:
        layout = _resolve_print_layout({
            "paper": "Custom",
            "paper_width_mm": 210,
            "paper_height_mm": 297,
            "margin_mm": 5,
            "bleed_mm": 4,
            "crop_marks": True,
        })

        self.assertEqual(layout, ("Custom 210x297 mm", (210.0, 297.0), 5, 4, True))

    def test_crop_marks_require_minimum_bleed(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 4 mm of bleed"):
            _resolve_print_layout({"paper": "A1 portrait", "crop_marks": True})

    def test_validates_large_print_tiff_without_loading_image_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "print.tif"
            profile = b"test ICC profile"
            tifffile.imwrite(
                path,
                data=np.zeros((24, 32, 4), dtype=np.uint8),
                photometric="separated",
                iccprofile=profile,
                resolution=(700, 700),
                resolutionunit="INCH",
            )

            _validate_print_tiff(path, (32, 24), profile, 700)

    def test_creates_enhanced_image_preview_and_report_without_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "portrait.png"
            Image.new("RGBA", (96, 128), (90, 120, 160, 255)).save(source)

            first = process_image(source, root / "output", {"enable_print_output": False})
            second = process_image(source, root / "output", {"enable_print_output": False})

            self.assertEqual(first["status"], "completed")
            self.assertEqual(second["status"], "completed")
            self.assertTrue(Path(first["outputs"]["enhanced"]).is_file())
            self.assertTrue(Path(first["outputs"]["preview"]).is_file())
            self.assertNotEqual(first["outputs"]["enhanced"], second["outputs"]["enhanced"])
            report = json.loads(Path(first["report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(report["analysis"]["width"], 96)


if __name__ == "__main__":
    unittest.main()
