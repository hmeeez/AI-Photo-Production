import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.settings import load_config


class SettingsTests(unittest.TestCase):
    def test_loads_portable_example_when_local_config_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.example.json").write_text(
                json.dumps({"input_dir": "INPUT", "model_path": "MODELS/model.onnx"}),
                encoding="utf-8",
            )

            config = load_config(root)

            self.assertEqual(config["input_dir"], "INPUT")
            self.assertEqual(config["model_path"], "MODELS/model.onnx")

    def test_prefers_local_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(
                json.dumps({"input_dir": "my-photos"}),
                encoding="utf-8",
            )

            self.assertEqual(load_config(root), {"input_dir": "my-photos"})

    def test_rejects_non_object_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text("[]", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "JSON object"):
                load_config(root)


if __name__ == "__main__":
    unittest.main()
