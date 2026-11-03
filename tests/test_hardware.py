import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hardware.detect import recommended_ai_tile_size


class HardwareTuningTests(unittest.TestCase):
    def test_ai_tile_size_scales_with_available_memory(self) -> None:
        self.assertEqual(recommended_ai_tile_size(2), 128)
        self.assertEqual(recommended_ai_tile_size(4), 256)
        self.assertEqual(recommended_ai_tile_size(8), 384)
        self.assertEqual(recommended_ai_tile_size(16), 512)


if __name__ == "__main__":
    unittest.main()
