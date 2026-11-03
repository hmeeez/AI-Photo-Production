import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from image_processing.analysis import analyze_image


class ImageAnalysisTests(unittest.TestCase):
    def test_analysis_reports_color_space_profile_and_conservative_category(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "photo.jpg"
            profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
            Image.new("RGB", (480, 640), (110, 90, 70)).save(
                path, icc_profile=profile.tobytes()
            )
            result = analyze_image(path)

            self.assertEqual(result["color_space"], "RGB")
            self.assertTrue(result["icc_profile_embedded"])
            self.assertIn("sRGB", result["icc_profile_description"])
            self.assertEqual(result["category"], "unknown")
            self.assertEqual(result["face_count"], 0)
            self.assertLessEqual(result["category_confidence"], 0.5)

    def test_noise_estimate_uses_signed_residuals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "flat.png"
            Image.fromarray(np.full((128, 128, 3), 128, dtype=np.uint8)).save(path)
            result = analyze_image(path)
            self.assertEqual(result["noise_estimate"], 0)


if __name__ == "__main__":
    unittest.main()
