import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from illustrator.automation import _jsx_string, _validate_pdf_x4, create_illustrator_files


class IllustratorAutomationTests(unittest.TestCase):
    def test_jsx_string_escapes_paths(self) -> None:
        self.assertEqual(_jsx_string(r"C:\Photos\it's.jpg"), '"C:\\\\Photos\\\\it\'s.jpg"')

    def test_rejects_invalid_fit_mode_before_com_automation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            illustrator = root / "Illustrator.exe"
            tiff = root / "photo.tif"
            illustrator.touch()
            tiff.touch()
            with patch("illustrator.automation.pythoncom.CoInitialize") as initialize:
                with self.assertRaisesRegex(ValueError, "fit mode"):
                    create_illustrator_files(
                        tiff,
                        root / "photo.ai",
                        root / "photo.pdf",
                        illustrator,
                        (594, 841),
                        "stretch",
                    )
                initialize.assert_not_called()

    def test_pdf_validation_checks_page_box_and_x4_intent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "print.pdf"
            path.write_bytes(
                b"%PDF-1.6\n"
                b"1 0 obj << /Type /Page /MediaBox [0 0 1684.7 2383.9] >> endobj\n"
                b"/GTS_PDFXVersion (PDF/X-4)\n/GTS_PDFX\n"
            )
            valid, message = _validate_pdf_x4(path, 1684.7, 2383.9)
            self.assertTrue(valid, message)

            path.write_bytes(b"%PDF-1.6\n/MediaBox [0 0 600 800]\n")
            valid, message = _validate_pdf_x4(path, 1684.7, 2383.9)
            self.assertFalse(valid)
            self.assertIn("dimensions", message)


if __name__ == "__main__":
    unittest.main()
