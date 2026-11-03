from __future__ import annotations

import io
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import psutil
import tifffile
from PIL import Image, ImageCms, ImageEnhance, ImageOps

from ai.super_resolution import (
    InferenceRuntime,
    model_sha256,
    prepare_inference,
    upscale_with_session,
)
from color_management.icc import validate_cmyk_profile
from illustrator.automation import create_illustrator_files
from image_processing.analysis import SUPPORTED_EXTENSIONS, analyze_image
from print_preparation.dimensions import A1_LANDSCAPE_MM, A1_MM, target_pixels

MIN_FREE_DISK_GB = 1.0
SYSTEM_RAM_RESERVE_GB = 3.0


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 10000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not find an unused output name for {path.name}.")


def _resolve_print_layout(
    options: dict[str, object],
) -> tuple[str, tuple[float, float], float, float, bool]:
    paper = str(options.get("paper", "A1 portrait"))
    if paper == "A1 portrait":
        paper_mm = A1_MM
    elif paper == "A1 landscape":
        paper_mm = A1_LANDSCAPE_MM
    elif paper == "Custom":
        paper_mm = (
            float(options.get("paper_width_mm", 594)),
            float(options.get("paper_height_mm", 841)),
        )
    else:
        raise ValueError(f"Unsupported paper size: {paper}")
    if not all(math.isfinite(value) and value > 0 for value in paper_mm):
        raise ValueError("Custom paper dimensions must be finite positive millimeter values.")
    margin_mm = float(options.get("margin_mm", 0))
    bleed_mm = float(options.get("bleed_mm", 0))
    if not math.isfinite(margin_mm) or margin_mm < 0:
        raise ValueError("Print margin must be a finite non-negative value.")
    if not math.isfinite(bleed_mm) or bleed_mm < 0:
        raise ValueError("Print bleed must be a finite non-negative value.")
    if margin_mm * 2 >= min(paper_mm):
        raise ValueError("Print margin leaves no printable area.")
    crop_marks = bool(options.get("crop_marks", False))
    if crop_marks and bleed_mm < 4:
        raise ValueError("Crop marks require at least 4 mm of bleed.")
    paper_label = (
        f"Custom {paper_mm[0]:g}x{paper_mm[1]:g} mm"
        if paper == "Custom"
        else paper
    )
    return paper_label, paper_mm, margin_mm, bleed_mm, crop_marks


def _resource_preflight(
    source_path: Path,
    output_dir: Path,
    target_size: tuple[int, int] | None = None,
    upscale_scale: int = 1,
    ai_tile_size: int = 256,
    ai_tile_overlap: int = 16,
) -> None:
    with Image.open(source_path) as image:
        source_pixels = image.width * image.height
    upscale_pixels = source_pixels * upscale_scale * upscale_scale
    estimated_ram = source_pixels * 20 / (1024**3)
    if upscale_scale > 1:
        tile_extent = ai_tile_size + ai_tile_overlap * 2
        tile_working_bytes = (
            tile_extent * tile_extent * upscale_scale * upscale_scale * 4 * 6
        )
        estimated_ram += (0.25 + tile_working_bytes / (1024**3))
    available_ram = psutil.virtual_memory().available / (1024**3)
    reserve = min(SYSTEM_RAM_RESERVE_GB, available_ram * 0.25)
    usable_ram = max(0.0, available_ram - reserve)
    if estimated_ram > usable_ram:
        raise MemoryError(
            f"Estimated working memory is {estimated_ram:.1f} GB; only "
            f"{usable_ram:.1f} GB is available after the system reserve."
        )
    free_disk = shutil.disk_usage(output_dir).free / (1024**3)
    print_pixels = target_size[0] * target_size[1] if target_size else source_pixels
    estimated_output_gb = (
        print_pixels * 4
        + (upscale_pixels * 3 if upscale_scale > 1 else 0)
    ) / (1024**3) + 0.1
    if free_disk < estimated_output_gb + MIN_FREE_DISK_GB:
        raise OSError(
            f"Print output needs about {estimated_output_gb:.1f} GB plus a "
            f"{MIN_FREE_DISK_GB:.1f} GB safety reserve; {free_disk:.1f} GB is free."
        )


def _enhance(
    source: Image.Image,
    analysis: dict[str, object],
) -> tuple[Image.Image, list[str]]:
    image = ImageOps.exif_transpose(source)
    has_alpha = "A" in image.getbands() or "transparency" in source.info
    alpha = image.convert("RGBA").getchannel("A") if has_alpha else None
    if has_alpha:
        image = image.convert("RGBA")
    color_image = image.convert("RGB") if alpha is not None else image
    embedded_profile = source.info.get("icc_profile")
    if embedded_profile:
        try:
            input_profile = ImageCms.ImageCmsProfile(io.BytesIO(embedded_profile))
            srgb_profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
            image = ImageCms.profileToProfile(
                color_image, input_profile, srgb_profile, outputMode="RGB"
            )
        except (OSError, ImageCms.PyCMSError) as exc:
            raise ValueError(f"Could not convert the embedded source ICC profile: {exc}") from exc
    elif color_image.mode == "CMYK":
        raise ValueError("CMYK source has no embedded ICC profile; color conversion is unsafe.")
    else:
        image = color_image.convert("RGB")
    actions: list[str] = []
    brightness = float(analysis["exposure_mean"])
    if brightness < 0.2 or brightness > 0.82:
        pixels = np.asarray(image)
        lab = cv2.cvtColor(pixels, cv2.COLOR_RGB2LAB)
        lightness, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.35, tileGridSize=(8, 8))
        adjusted = cv2.cvtColor(
            cv2.merge((clahe.apply(lightness), a_channel, b_channel)),
            cv2.COLOR_LAB2RGB,
        )
        del pixels, lab, lightness, a_channel, b_channel
        image = Image.fromarray(adjusted)
        del adjusted
        actions.append("Applied mild local contrast correction for exposure.")
    result = image
    blur_score = float(analysis["blur_score"])
    noise = float(analysis["noise_estimate"])
    if 35 <= blur_score <= 300 and noise < 18:
        result = ImageEnhance.Sharpness(result).enhance(1.05)
        actions.append("Applied mild sharpening to a moderately soft, low-noise image.")
    if alpha is not None:
        result.putalpha(alpha)
    srgb_profile = (
        ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
    )
    result.info["icc_profile"] = srgb_profile.tobytes()
    return result, actions


def _write_print_tiff(
    image: Image.Image,
    destination: Path,
    profile_path: Path,
    ppi: int,
    fit_mode: str,
    paper_mm: tuple[float, float],
    original_size: tuple[int, int],
    margin_mm: float = 0,
    bleed_mm: float = 0,
    crop_marks: bool = False,
) -> tuple[tuple[int, int], float]:
    profile = validate_cmyk_profile(profile_path)
    trim_width, trim_height = target_pixels(*paper_mm, ppi)
    target_width, target_height = target_pixels(
        paper_mm[0] + bleed_mm * 2,
        paper_mm[1] + bleed_mm * 2,
        ppi,
    )
    trim_left = (target_width - trim_width) // 2
    trim_top = (target_height - trim_height) // 2
    margin_px = round(margin_mm / 25.4 * ppi)
    safe_x = margin_px
    safe_y = margin_px
    safe_width = target_width - 2 * margin_px
    safe_height = target_height - 2 * margin_px
    if fit_mode not in {"fit", "crop"}:
        raise ValueError("Fit mode must be 'fit' or 'crop'.")
    if safe_width <= 0 or safe_height <= 0:
        raise ValueError("Print margins leave no printable area at the selected PPI.")
    if "A" in image.getbands():
        flattened = Image.new("RGB", image.size, "white")
        flattened.paste(image, mask=image.getchannel("A"))
        source = flattened
    elif image.mode == "RGB":
        source = image
    else:
        source = image.convert("RGB")
    target_ratio = safe_width / safe_height
    source_ratio = source.width / source.height
    if fit_mode == "crop":
        if source_ratio > target_ratio:
            crop_width = source.height * target_ratio
            crop_box = ((source.width - crop_width) / 2, 0, (source.width + crop_width) / 2, source.height)
        else:
            crop_height = source.width / target_ratio
            crop_box = (0, (source.height - crop_height) / 2, source.width, (source.height + crop_height) / 2)

    if fit_mode == "fit":
        scale = min(safe_width / source.width, safe_height / source.height)
        content_width = max(1, round(source.width * scale))
        content_height = max(1, round(source.height * scale))
        content_x = safe_x + (safe_width - content_width) // 2
        content_y = safe_y + (safe_height - content_height) // 2
        scale_x = content_width / source.width
        scale_y = content_height / source.height
        source_box = (0.0, 0.0, float(source.width), float(source.height))
    else:
        content_width, content_height = safe_width, safe_height
        content_x, content_y = safe_x, safe_y
        crop_x0, crop_y0, crop_x1, crop_y1 = crop_box
        scale_x = safe_width / (crop_x1 - crop_x0)
        scale_y = safe_height / (crop_y1 - crop_y0)
        scale = max(scale_x, scale_y)
        source_box = crop_box

    transform = ImageCms.buildTransformFromOpenProfiles(
        ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")),
        profile,
        "RGB",
        "CMYK",
        renderingIntent=1,
    )
    tile_size = 256
    mark_rectangles: list[tuple[int, int, int, int]] = []
    if crop_marks:
        mark_length = round(3 * ppi / 25.4)
        mark_gap = round(0.75 * ppi / 25.4)
        mark_width = max(1, round(0.25 * ppi / 25.4))
        trim_right = trim_left + trim_width
        trim_bottom = trim_top + trim_height
        mark_rectangles = [
            (trim_left - mark_length, trim_top - mark_width, trim_left - mark_gap, trim_top),
            (trim_left - mark_width, trim_top - mark_length, trim_left, trim_top - mark_gap),
            (trim_right + mark_gap, trim_top - mark_width, trim_right + mark_length, trim_top),
            (trim_right, trim_top - mark_length, trim_right + mark_width, trim_top - mark_gap),
            (trim_left - mark_length, trim_bottom, trim_left - mark_gap, trim_bottom + mark_width),
            (trim_left - mark_width, trim_bottom + mark_gap, trim_left, trim_bottom + mark_length),
            (trim_right + mark_gap, trim_bottom, trim_right + mark_length, trim_bottom + mark_width),
            (trim_right, trim_bottom + mark_gap, trim_right + mark_width, trim_bottom + mark_length),
        ]

    def cmyk_tiles():
        for tile_y in range(0, target_height, tile_size):
            for tile_x in range(0, target_width, tile_size):
                tile = Image.new("RGB", (tile_size, tile_size), "white")
                left = max(tile_x, content_x)
                top = max(tile_y, content_y)
                right = min(tile_x + tile_size, content_x + content_width)
                bottom = min(tile_y + tile_size, content_y + content_height)
                if left < right and top < bottom:
                    if fit_mode == "fit":
                        extent = (
                            (left - content_x) / scale_x,
                            (top - content_y) / scale_y,
                            (right - content_x) / scale_x,
                            (bottom - content_y) / scale_y,
                        )
                    else:
                        extent = (
                            source_box[0] + (left - content_x) / scale_x,
                            source_box[1] + (top - content_y) / scale_y,
                            source_box[0] + (right - content_x) / scale_x,
                            source_box[1] + (bottom - content_y) / scale_y,
                        )
                    patch_size = (right - left, bottom - top)
                    patch = source.transform(
                        patch_size,
                        Image.Transform.EXTENT,
                        extent,
                        resample=Image.Resampling.BICUBIC,
                    )
                    tile.paste(patch, (left - tile_x, top - tile_y))
                converted = ImageCms.applyTransform(tile, transform)
                output_tile = np.zeros((tile_size, tile_size, 4), dtype=np.uint8)
                converted_width = min(tile_size, target_width - tile_x)
                converted_height = min(tile_size, target_height - tile_y)
                output_tile[:converted_height, :converted_width] = np.asarray(converted)[:converted_height, :converted_width]
                for x0, y0, x1, y1 in mark_rectangles:
                    left = max(tile_x, x0, 0)
                    top = max(tile_y, y0, 0)
                    right = min(tile_x + converted_width, x1, target_width)
                    bottom = min(tile_y + converted_height, y1, target_height)
                    if left < right and top < bottom:
                        mark = output_tile[
                            top - tile_y : bottom - tile_y,
                            left - tile_x : right - tile_x,
                        ]
                        mark[:] = (0, 0, 0, 255)
                yield output_tile

    partial_path = destination.with_name(f"{destination.stem}.partial{destination.suffix}")
    try:
        with tifffile.TiffWriter(partial_path, bigtiff=target_width * target_height * 4 >= 4_000_000_000) as writer:
            writer.write(
                data=cmyk_tiles(),
                shape=(target_height, target_width, 4),
                dtype=np.uint8,
                photometric="separated",
                planarconfig="contig",
                tile=(tile_size, tile_size),
                compression="deflate",
                iccprofile=profile.tobytes(),
                resolution=(ppi, ppi),
                resolutionunit="INCH",
                metadata=None,
                maxworkers=1,
                buffersize=1,
            )
        _validate_print_tiff(
            partial_path,
            (target_width, target_height),
            profile.tobytes(),
            ppi,
        )
        partial_path.rename(destination)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise

    original_width, original_height = original_size
    if fit_mode == "fit":
        effective_ppi = min(
            original_width * ppi / content_width,
            original_height * ppi / content_height,
        )
    else:
        visible_source_width = original_width * (source_box[2] - source_box[0]) / source.width
        visible_source_height = original_height * (source_box[3] - source_box[1]) / source.height
        effective_ppi = min(
            visible_source_width * ppi / target_width,
            visible_source_height * ppi / target_height,
        )
    return (target_width, target_height), round(effective_ppi, 1)


def _validate_print_tiff(
    path: Path,
    expected_size: tuple[int, int],
    expected_profile: bytes,
    expected_ppi: int,
) -> None:
    with tifffile.TiffFile(path) as check:
        page = check.pages[0]
        if (
            page.imagewidth != expected_size[0]
            or page.imagelength != expected_size[1]
            or page.samplesperpixel != 4
            or page.photometric != tifffile.PHOTOMETRIC.SEPARATED
        ):
            raise OSError("Print TIFF failed its dimensions or color-mode validation.")
        profile_tag = page.tags.get(34675)
        if profile_tag is None or profile_tag.value != expected_profile:
            raise OSError("Print TIFF is missing its embedded printer ICC profile.")
        x_resolution = page.tags.get("XResolution")
        y_resolution = page.tags.get("YResolution")
        resolution_unit = page.tags.get("ResolutionUnit")
        if x_resolution is None or y_resolution is None or resolution_unit is None:
            raise OSError("Print TIFF is missing its resolution metadata.")
        resolutions = tuple(
            round(value[0] / value[1]) if isinstance(value, tuple) else round(value)
            for value in (x_resolution.value, y_resolution.value)
        )
        if resolutions != (expected_ppi, expected_ppi) or int(resolution_unit.value) != 2:
            raise OSError("Print TIFF resolution metadata does not match the requested PPI.")


def process_image(
    source_path: Path,
    output_root: Path,
    options: dict[str, object],
    progress: Callable[[str], None] | None = None,
    inference_runtime: InferenceRuntime | None = None,
) -> dict[str, object]:
    started = time.monotonic()
    output_dirs = {name: output_root / name for name in (
        "01_ENHANCED", "02_UPSCALED", "03_PRINT_READY",
        "04_ILLUSTRATOR", "05_PDF", "06_REPORTS", "07_PREVIEWS",
    )}
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "source": str(source_path),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": {},
        "warnings": [],
        "errors": [],
        "stages": {"ai_upscaling": "not configured; no AI enhancement was run"},
    }
    try:
        if source_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {source_path.suffix}")
        target_size: tuple[int, int] | None = None
        paper_name, paper_mm, margin_mm, bleed_mm, crop_marks = _resolve_print_layout(
            options
        )
        ai_runtime = inference_runtime
        upscale_scale = 1
        if bool(options.get("enable_ai_upscale")):
            if ai_runtime is None:
                ai_runtime = prepare_inference(
                    Path(str(options.get("model_path") or "")),
                    str(options.get("ai_backend", "auto")),
                )
            upscale_scale = ai_runtime.scale
        if bool(options.get("enable_print_output")):
            ppi = int(options.get("target_ppi", 700))
            bleed_size = (
                paper_mm[0] + bleed_mm * 2,
                paper_mm[1] + bleed_mm * 2,
            )
            target_size = target_pixels(*bleed_size, ppi)
        if target_size:
            validate_cmyk_profile(Path(str(options.get("icc_profile") or "")))
        _resource_preflight(
            source_path,
            output_root,
            target_size,
            upscale_scale,
            int(options.get("ai_tile_size", 256)),
            int(options.get("ai_tile_overlap", 16)),
        )
        analysis = analyze_image(source_path)
        report["analysis"] = analysis
        report["warnings"].extend(analysis["warnings"])

        if progress:
            progress(f"Enhancing {source_path.name}")
        with Image.open(source_path) as source:
            enhanced, actions = _enhance(source, analysis)
        report["stages"]["traditional_enhancement"] = actions or ["No enhancement filter was indicated."]

        enhanced_path = unique_path(output_dirs["01_ENHANCED"] / f"{source_path.stem}_enhanced.png")
        enhanced.save(
            enhanced_path,
            format="PNG",
            optimize=False,
            icc_profile=enhanced.info.get("icc_profile"),
        )
        report["outputs"]["enhanced"] = str(enhanced_path)

        with Image.open(source_path) as original:
            before = ImageOps.exif_transpose(original).convert("RGB")
            before.thumbnail((720, 720), Image.Resampling.LANCZOS)
            before.load()
        production_path = enhanced_path
        if bool(options.get("enable_ai_upscale")):
            if ai_runtime is None:
                raise RuntimeError("AI model runtime was not initialized.")
            if progress:
                progress(f"Running tiled {ai_runtime.backend} super-resolution for {source_path.name}")
            upscaled_path = unique_path(
                output_dirs["02_UPSCALED"] / f"{source_path.stem}_upscaled.tif"
            )
            width, height, detected_scale = upscale_with_session(
                enhanced,
                ai_runtime.session,
                upscaled_path,
                tile_size=int(options.get("ai_tile_size", 256)),
                overlap=int(options.get("ai_tile_overlap", 16)),
                progress=progress,
                model_info=(
                    ai_runtime.input_name,
                    ai_runtime.output_name,
                    ai_runtime.scale,
                ),
            )
            production_path = upscaled_path
            report["outputs"]["upscaled"] = str(upscaled_path)
            report["stages"]["ai_upscaling"] = {
                "status": "completed",
                "backend": ai_runtime.backend,
                "scale": detected_scale,
                "model_sha256": model_sha256(Path(str(options.get("model_path") or ""))),
                "tile_size": int(options.get("ai_tile_size", 256)),
                "tile_overlap": int(options.get("ai_tile_overlap", 16)),
                "dimensions": [width, height],
            }
            if ai_runtime.fallback_reason:
                report["warnings"].append(
                    f"Automatic inference fallback: {ai_runtime.fallback_reason}; "
                    f"used {ai_runtime.backend}."
                )
            report["warnings"].append(
                "AI super-resolution may synthesize plausible texture; inspect faces, text, "
                "logos, artwork, and products before print use."
            )
        with Image.open(production_path) as production_image:
            preview = (
                production_image
                if production_image.mode == "RGB"
                else production_image.convert("RGB")
            )
            preview.thumbnail((720, 720), Image.Resampling.LANCZOS)
            preview.load()
        canvas = Image.new("RGB", (before.width + preview.width, max(before.height, preview.height)), "white")
        canvas.paste(before, (0, 0))
        canvas.paste(preview, (before.width, 0))
        preview_path = unique_path(output_dirs["07_PREVIEWS"] / f"{source_path.stem}_before_after.jpg")
        canvas.save(preview_path, quality=88, optimize=True)
        report["outputs"]["preview"] = str(preview_path)

        if target_size:
            if progress:
                progress(f"Preparing {paper_name} CMYK TIFF for {source_path.name}")
            profile_path = Path(str(options.get("icc_profile") or ""))
            print_path = unique_path(
                output_dirs["03_PRINT_READY"] /
                f"{source_path.stem}_{paper_name.replace(' ', '_').replace('x', 'by')}_"
                f"{int(options.get('target_ppi', 700))}PPI_CMYK.tif"
            )
            with Image.open(production_path) as production_image:
                size, effective_ppi = _write_print_tiff(
                    production_image,
                    print_path,
                    profile_path,
                    int(options.get("target_ppi", 700)),
                    str(options.get("fit_mode", "fit")),
                    paper_mm,
                    (int(analysis["width"]), int(analysis["height"])),
                    margin_mm,
                    bleed_mm,
                    crop_marks,
                )
            report["outputs"]["print_ready"] = str(print_path)
            report["print"] = {
                "pixel_dimensions": list(size),
                "paper": paper_name,
                "paper_dimensions_mm": list(paper_mm),
                "margin_mm": margin_mm,
                "bleed_mm": bleed_mm,
                "crop_marks": crop_marks,
                "media_dimensions_mm": [
                    paper_mm[0] + bleed_mm * 2,
                    paper_mm[1] + bleed_mm * 2,
                ],
                "metadata_ppi": int(options.get("target_ppi", 700)),
                "effective_source_ppi": effective_ppi,
                "color_mode": "CMYK",
                "icc_profile": str(profile_path),
            }
            if effective_ppi < int(options.get("target_ppi", 700)):
                report["warnings"].append(
                    f"Effective source detail is about {effective_ppi:.1f} PPI; "
                    "the output metadata PPI does not represent recovered detail."
                )
            if bool(options.get("enable_illustrator")):
                if progress:
                    progress(f"Creating Illustrator files for {source_path.name}")
                illustrator_path = Path(str(options.get("illustrator_path") or ""))
                ai_path = unique_path(
                    output_dirs["04_ILLUSTRATOR"] / f"{source_path.stem}_print.ai"
                )
                pdf_path = unique_path(
                    output_dirs["05_PDF"] / f"{source_path.stem}_print_PDFX4.pdf"
                )
                deliverables = create_illustrator_files(
                    print_path,
                    ai_path,
                    pdf_path,
                    illustrator_path,
                    (
                        paper_mm[0] + bleed_mm * 2,
                        paper_mm[1] + bleed_mm * 2,
                    ),
                    str(options.get("fit_mode", "fit")),
                )
                report["outputs"]["illustrator"] = deliverables["ai"]
                if deliverables["pdf"]:
                    report["outputs"]["pdf_x4"] = deliverables["pdf"]
                if deliverables["pdf_preset"]:
                    report["print"]["pdf_x4_preset"] = deliverables["pdf_preset"]
                if deliverables["warning"]:
                    report["warnings"].append(deliverables["warning"])
                    report["stages"]["pdf_x4"] = "not produced"
                else:
                    report["stages"]["pdf_x4"] = (
                        "created using Illustrator PDF/X-4 preset; PDF/X-4 markers, "
                        "output intent, and page dimensions checked"
                    )
                report["stages"]["illustrator"] = (
                    "saved by Illustrator and verified non-empty"
                )
        else:
            report["warnings"].append("Print output disabled; no CMYK TIFF was generated.")
            if bool(options.get("enable_illustrator")):
                raise ValueError("Illustrator output requires print-ready CMYK TIFF output.")

        report["status"] = (
            "partial" if report["stages"].get("pdf_x4") == "not produced" else "completed"
        )
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report_path = unique_path(
            output_dirs["06_REPORTS"] / f"{source_path.stem}_report.json"
        )
        report["report_path"] = str(report_path)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
