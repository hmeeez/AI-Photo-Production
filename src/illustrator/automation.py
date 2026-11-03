from __future__ import annotations

import json
import re
from pathlib import Path

import pythoncom
import win32com.client


def _jsx_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def _validate_pdf_x4(
    pdf_path: Path,
    width_points: float,
    height_points: float,
) -> tuple[bool, str]:
    if not pdf_path.is_file() or pdf_path.stat().st_size < 5:
        return False, "Illustrator reported PDF export, but the output file failed validation."
    with pdf_path.open("rb") as pdf_file:
        header = pdf_file.read(8)
        if not header.startswith(b"%PDF-"):
            return False, "Illustrator output is not a valid PDF file."
        pdf_file.seek(0)
        pdf_sample = pdf_file.read(min(pdf_path.stat().st_size, 16 * 1024 * 1024))

    media_box = re.search(
        rb"/MediaBox\s*\[\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)"
        rb"\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*\]",
        pdf_sample,
    )
    if not media_box:
        return False, "PDF page dimensions could not be verified."
    x0, y0, x1, y1 = (float(value) for value in media_box.groups())
    if abs((x1 - x0) - width_points) > 2 or abs((y1 - y0) - height_points) > 2:
        return False, "PDF page dimensions do not match the requested artboard."
    if b"GTS_PDFXVersion" not in pdf_sample or b"PDF/X-4" not in pdf_sample:
        return False, "PDF does not contain Illustrator's PDF/X-4 conformance declaration."
    if b"/GTS_PDFX" not in pdf_sample:
        return False, "PDF is missing the PDF/X output intent."
    return True, "PDF/X-4 markers and page dimensions verified."


def create_illustrator_files(
    tiff_path: Path,
    ai_path: Path,
    pdf_path: Path,
    illustrator_path: Path,
    paper_mm: tuple[float, float],
    fit_mode: str = "fit",
) -> dict[str, str | None]:
    if not illustrator_path.is_file():
        raise FileNotFoundError(f"Adobe Illustrator was not found at {illustrator_path}.")
    if not tiff_path.is_file():
        raise FileNotFoundError(f"CMYK TIFF does not exist: {tiff_path}")
    if fit_mode not in {"fit", "crop"}:
        raise ValueError("Illustrator fit mode must be 'fit' or 'crop'.")

    width_points = paper_mm[0] / 25.4 * 72
    height_points = paper_mm[1] / 25.4 * 72
    scale_expression = "Math.min" if fit_mode == "fit" else "Math.max"
    crop_script = ""
    if fit_mode == "crop":
        crop_script = f"""
var clipGroup = doc.groupItems.add();
embedded.move(clipGroup, ElementPlacement.PLACEATEND);
var clipPath = doc.pathItems.rectangle({height_points:.5f}, 0, {width_points:.5f}, {height_points:.5f});
clipPath.move(clipGroup, ElementPlacement.PLACEATBEGINNING);
clipPath.clipping = true;
clipGroup.clipped = true;
"""
    script = f"""
var inputFile = new File({_jsx_string(str(tiff_path))});
var aiFile = new File({_jsx_string(str(ai_path))});
var pdfFile = new File({_jsx_string(str(pdf_path))});
var doc = app.documents.add(DocumentColorSpace.CMYK, {width_points:.5f}, {height_points:.5f});
var placed = doc.placedItems.add();
placed.file = inputFile;
var scale = {scale_expression}({width_points:.5f} / placed.width, {height_points:.5f} / placed.height);
placed.width = placed.width * scale;
placed.height = placed.height * scale;
placed.position = [({width_points:.5f} - placed.width) / 2, ({height_points:.5f} + placed.height) / 2];
placed.embed();
if (doc.rasterItems.length === 0) throw new Error("Embedded image validation failed.");
var embedded = doc.rasterItems[doc.rasterItems.length - 1];
{crop_script}
var aiOptions = new IllustratorSaveOptions();
aiOptions.pdfCompatible = true;
doc.saveAs(aiFile, aiOptions);
var aiBytes = aiFile.length;
var artboard = doc.artboards[0].artboardRect;
var presetName = "";
var presets = app.PDFPresetsList;
for (var i = 0; i < presets.length; i++) {{
    if (presets[i].toLowerCase().indexOf("pdf/x-4") >= 0) {{
        presetName = presets[i];
        break;
    }}
}}
var result = "AI_SAVED|" + aiBytes + "|" + doc.rasterItems.length + "|" +
    Math.round(artboard[2] - artboard[0]) + "|" + Math.round(artboard[1] - artboard[3]);
if (presetName) {{
    var pdfOptions = new PDFSaveOptions();
    pdfOptions.pDFPreset = presetName;
    doc.saveAs(pdfFile, pdfOptions);
    result += "|PDF_SAVED|" + presetName;
}} else {{
    result += "|PDF_X4_PRESET_NOT_FOUND";
}}
doc.close(SaveOptions.DONOTSAVECHANGES);
result;
"""
    pythoncom.CoInitialize()
    application = None
    try:
        application = win32com.client.Dispatch("Illustrator.Application")
        try:
            result = str(application.DoJavaScript(script))
        except Exception as exc:
            if ai_path.is_file() and ai_path.stat().st_size > 0:
                return {
                    "ai": str(ai_path),
                    "pdf": None,
                    "pdf_preset": None,
                    "warning": f"Illustrator saved the AI file but PDF/X-4 export failed: {exc}",
                }
            raise RuntimeError(f"Illustrator scripting failed: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Could not automate Adobe Illustrator: {exc}") from exc
    finally:
        if application is not None:
            application = None
        pythoncom.CoUninitialize()

    if not result.startswith("AI_SAVED|") or not ai_path.is_file() or ai_path.stat().st_size == 0:
        raise RuntimeError(f"Illustrator did not validate the native AI document (response: {result!r}).")
    result_parts = result.split("|")
    try:
        reported_bytes, embedded_count, reported_width, reported_height = (
            int(value) for value in result_parts[1:5]
        )
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"Illustrator returned invalid document validation data: {result!r}") from exc
    if reported_bytes <= 0 or embedded_count < 1:
        raise RuntimeError(f"Illustrator document is empty or image was not embedded: {result!r}")
    expected_width, expected_height = round(width_points), round(height_points)
    if abs(reported_width - expected_width) > 2 or abs(reported_height - expected_height) > 2:
        raise RuntimeError(
            "Illustrator artboard dimensions do not match the requested page size: "
            f"{reported_width}x{reported_height} pt vs {expected_width}x{expected_height} pt."
        )
    if "PDF_SAVED" not in result:
        return {
            "ai": str(ai_path),
            "pdf": None,
            "pdf_preset": None,
            "warning": "AI document was saved, but no PDF/X-4 preset is installed in Illustrator.",
        }
    preset_marker = next(
        (part for part in result_parts[5:] if part.startswith("[PDF/X-4")),
        None,
    )
    if not preset_marker:
        return {
            "ai": str(ai_path),
            "pdf": None,
            "pdf_preset": None,
            "warning": "The PDF/X-4 preset name could not be verified in Illustrator's response.",
        }
    valid_pdf, validation_message = _validate_pdf_x4(
        pdf_path, width_points, height_points
    )
    if not valid_pdf:
        return {
            "ai": str(ai_path),
            "pdf": None,
            "pdf_preset": preset_marker,
            "warning": validation_message,
        }
    return {
        "ai": str(ai_path),
        "pdf": str(pdf_path),
        "pdf_preset": preset_marker,
        "warning": None,
    }
