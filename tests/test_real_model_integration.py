import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai.super_resolution import (
    create_cpu_session,
    model_sha256,
    prepare_inference,
    upscale_with_session,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "MODELS" / "real_esrgan_x4.onnx"
EXPECTED_SHA256 = "5c586662929cbc686c1a5c38d9c060dbdb4ea5863a1f7672b8c0761e6b89c033"


@unittest.skipUnless(MODEL_PATH.is_file(), "Download MODELS/real_esrgan_x4.onnx to run integration tests.")
class RealModelIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = Image.fromarray(
            np.random.default_rng(42).integers(0, 256, (48, 64, 3), dtype=np.uint8)
        )

    def test_bundled_model_checksum_and_cpu_inference(self) -> None:
        self.assertEqual(model_sha256(MODEL_PATH), EXPECTED_SHA256)
        session = create_cpu_session(MODEL_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "cpu_upscaled.tif"
            dimensions = upscale_with_session(
                self.image,
                session,
                output_path,
                tile_size=64,
                overlap=8,
            )
            self.assertEqual(dimensions, (256, 192, 4))
            with Image.open(output_path) as output:
                self.assertEqual(output.size, (256, 192))
                self.assertTrue(output.info.get("icc_profile"))

    def test_automatic_backend_inference(self) -> None:
        runtime = prepare_inference(MODEL_PATH, "auto")
        self.assertIn(runtime.backend, {"OpenVINO Intel GPU", "ONNX Runtime CPU"})
        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "automatic_upscaled.tif"
            dimensions = upscale_with_session(
                self.image,
                runtime.session,
                output_path,
                tile_size=64,
                overlap=8,
                model_info=(runtime.input_name, runtime.output_name, runtime.scale),
            )
            self.assertEqual(dimensions, (256, 192, 4))
            self.assertTrue(output_path.is_file())


if __name__ == "__main__":
    unittest.main()
