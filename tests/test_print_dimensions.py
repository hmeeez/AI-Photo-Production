import unittest

from src.print_preparation.dimensions import target_pixels


class TargetPixelsTests(unittest.TestCase):
    def test_a1_at_700_ppi(self) -> None:
        self.assertEqual(target_pixels(594, 841, 700), (16370, 23177))

    def test_rejects_non_positive_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            target_pixels(0, 841, 300)

    def test_rejects_invalid_ppi(self) -> None:
        with self.assertRaises(ValueError):
            target_pixels(594, 841, 0)


if __name__ == "__main__":
    unittest.main()
