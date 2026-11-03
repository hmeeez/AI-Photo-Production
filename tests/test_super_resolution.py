import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai.super_resolution import (
    InferenceRuntime,
    _upscale_to_tiff,
    _validate_upscaled_tiff,
    create_vulkan_session,
    validate_model,
)


class _TensorInfo:
    def __init__(self, name: str, shape: list[object]) -> None:
        self.name = name
        self.shape = shape


class _NearestNeighborSession:
    def __init__(self, scale: int = 2) -> None:
        self.scale = scale
        self.inputs = [_TensorInfo("input", [1, 3, None, None])]
        self.outputs = [_TensorInfo("output", [1, 3, None, None])]

    def get_inputs(self) -> list[_TensorInfo]:
        return self.inputs

    def get_outputs(self) -> list[_TensorInfo]:
        return self.outputs

    def run(self, output_names: list[str], input_feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        batch = input_feed["input"]
        output = np.repeat(np.repeat(batch, self.scale, axis=2), self.scale, axis=3)
        return [output]


class _SymbolicDimensionSession(_NearestNeighborSession):
    def __init__(self) -> None:
        super().__init__()
        self.inputs[0].shape = [1, 3, "height", "width"]
        self.outputs[0].shape = [1, 3, "out_height", "out_width"]


class SuperResolutionTests(unittest.TestCase):
    def test_upscaled_tiff_validation_checks_metadata_without_decoding_pixels(self) -> None:
        import tifffile
        from PIL import ImageCms

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "large.tif"
            profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
            tifffile.imwrite(
                path,
                data=np.zeros((8, 8, 3), dtype=np.uint8),
                photometric="rgb",
                iccprofile=profile,
            )

            with patch("ai.super_resolution.Image.open", side_effect=AssertionError):
                _validate_upscaled_tiff(path, (8, 8), profile)

    def test_dynamic_model_probe_detects_scale(self) -> None:
        self.assertEqual(validate_model(_NearestNeighborSession()), ("input", "output", 2))

    def test_rejects_fixed_spatial_model(self) -> None:
        session = _NearestNeighborSession()
        session.inputs[0].shape = [1, 3, 256, 256]
        with self.assertRaisesRegex(ValueError, "dynamic-spatial"):
            validate_model(session)

    def test_accepts_symbolic_dynamic_dimensions(self) -> None:
        self.assertEqual(validate_model(_SymbolicDimensionSession()), ("input", "output", 2))

    def test_vulkan_selection_reports_missing_execution_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "model.onnx"
            model_path.write_bytes(b"test model fixture")
            with patch(
                "ai.super_resolution.ort.get_available_providers",
                return_value=["CPUExecutionProvider"],
            ):
                with self.assertRaisesRegex(RuntimeError, "does not include VulkanExecutionProvider"):
                    create_vulkan_session(model_path)

    def test_tiled_upscale_writes_valid_tiff_without_seams(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "upscaled.tif"
            source_array = np.zeros((141, 171, 3), dtype=np.uint8)
            source_array[:, :, 0] = np.arange(171, dtype=np.uint8)[None, :]
            source_array[:, :, 1] = np.arange(141, dtype=np.uint8)[:, None]
            source = Image.fromarray(source_array)

            width, height, scale = _upscale_to_tiff(
                source,
                destination,
                _NearestNeighborSession(2),
                tile_size=64,
                overlap=8,
            )

            self.assertEqual((width, height, scale), (342, 282, 2))
            self.assertTrue(destination.is_file())
            self.assertFalse((destination.parent / "upscaled.pixels.npy").exists())
            with Image.open(destination) as result:
                self.assertEqual(result.size, (342, 282))
                self.assertEqual(result.mode, "RGB")
                output_array = np.asarray(result)
                np.testing.assert_array_equal(output_array[::2, ::2], source_array)

    def test_pipeline_runs_cpu_upscale_and_records_model_provenance(self) -> None:
        from core.pipeline import process_image

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "photo.png"
            model_path = root / "model.onnx"
            model_path.write_bytes(b"test model fixture")
            Image.new("RGB", (95, 73), (30, 80, 140)).save(source_path)
            with patch(
                "core.pipeline.prepare_inference",
                return_value=InferenceRuntime(
                    _NearestNeighborSession(),
                    "ONNX Runtime CPU",
                    None,
                    "input",
                    "output",
                    2,
                ),
            ):
                report = process_image(
                    source_path,
                    root / "OUTPUT",
                    {
                        "enable_ai_upscale": True,
                        "model_path": str(model_path),
                        "ai_tile_size": 64,
                        "ai_tile_overlap": 8,
                        "enable_print_output": False,
                    },
                )
            self.assertEqual(report["status"], "completed", report["errors"])
            self.assertEqual(report["stages"]["ai_upscaling"]["backend"], "ONNX Runtime CPU")
            self.assertEqual(report["stages"]["ai_upscaling"]["scale"], 2)
            self.assertTrue(Path(report["outputs"]["upscaled"]).is_file())
            with Image.open(report["outputs"]["upscaled"]) as output:
                self.assertEqual(output.size, (190, 146))


if __name__ == "__main__":
    unittest.main()
