from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def analyze_image(path: Path) -> dict[str, Any]:
    with Image.open(path) as source:
        original_width, original_height = source.size
        image = ImageOps.exif_transpose(source)
        width, height = image.size
        image_format = source.format
        mode = source.mode
        icc_profile = source.info.get("icc_profile")
        dpi = source.info.get("dpi")
        bit_depth = (
            16 if mode.startswith("I;16") or mode == "I"
            else 32 if mode == "F"
            else 8
        )
        thumbnail = image.convert("RGB")
        thumbnail.thumbnail((512, 512), Image.Resampling.LANCZOS)
        pixels = np.asarray(thumbnail)

    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray) / 255.0)
    high_frequency = gray.astype(np.float32) - cv2.GaussianBlur(
        gray.astype(np.float32), (0, 0), 1.2
    )
    noise_score = float(np.std(high_frequency))
    hsv = cv2.cvtColor(pixels, cv2.COLOR_RGB2HSV)
    saturation = float(np.mean(hsv[:, :, 1]) / 255.0)
    channel_means = np.mean(pixels, axis=(0, 1))
    grayscale_like = saturation < 0.08 and float(np.std(channel_means)) < 8
    sepia_like = (
        float(channel_means[0]) > float(channel_means[1]) + 5
        and float(channel_means[1]) > float(channel_means[2]) + 5
        and saturation < 0.42
    )

    face_count = 0
    face_confidence = 0.0
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if cascade_path.is_file():
        detector = cv2.CascadeClassifier(str(cascade_path))
        if not detector.empty():
            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(24, 24),
            )
            face_count = len(faces)
            face_confidence = min(0.9, 0.65 + max(0, face_count - 1) * 0.05) if face_count else 0

    edges = cv2.Canny(gray, 80, 160)
    line_segments = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=max(24, min(gray.shape) // 8),
        maxLineGap=8,
    )
    line_count = 0 if line_segments is None else len(line_segments)
    edge_density = float(np.count_nonzero(edges) / edges.size)
    orientation = "portrait" if height >= width else "landscape"
    warnings: list[str] = []
    if blur_score < 60:
        warnings.append("Low measured edge detail; inspect the source before aggressive sharpening.")
    if brightness < 0.18 or brightness > 0.88:
        warnings.append("Image may be underexposed or overexposed.")
    if not icc_profile:
        warnings.append("Source has no embedded ICC profile.")
    if dpi is None:
        warnings.append("Source resolution metadata is absent.")
    category = "unknown"
    category_confidence = 0.0
    category_evidence = "No high-confidence visual category detected."
    if face_count:
        category = "portrait"
        category_confidence = face_confidence
        category_evidence = f"Detected {face_count} face(s) using OpenCV Haar cascade."
    elif (grayscale_like or sepia_like) and noise_score >= 3:
        category = "old photograph candidate"
        category_confidence = 0.4
        category_evidence = "Low-saturation/grayscale or sepia-like color with visible grain; review manually."
    elif line_count >= 12 and edge_density >= 0.04:
        category = "architecture candidate"
        category_confidence = 0.45
        category_evidence = "High straight-line and edge density; review manually."
    elif orientation == "landscape":
        category = "landscape"
        category_confidence = 0.35
        category_evidence = "Landscape aspect ratio only; scene subject was not classified."

    quality = max(
        0,
        min(
            100,
            round(
                min(100.0, blur_score / 4) * 0.7
                + max(0, 30 * (1 - abs(brightness - 0.5) * 2))
                - min(20, noise_score),
            ),
        ),
    )
    profile_description = None
    if icc_profile:
        try:
            profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile))
            profile_description = profile.profile.profile_description
        except (OSError, ImageCms.PyCMSError):
            warnings.append("Embedded ICC profile could not be parsed.")

    return {
        "filename": path.name,
        "format": image_format,
        "width": width,
        "height": height,
        "original_width": original_width,
        "original_height": original_height,
        "file_size_bytes": path.stat().st_size,
        "mode": mode,
        "bit_depth_estimate": bit_depth,
        "icc_profile_embedded": bool(icc_profile),
        "icc_profile_description": profile_description,
        "color_space": "CMYK" if "C" in mode and "M" in mode and "Y" in mode and "K" in mode else (
            "RGB" if mode in {"RGB", "RGBA"} else "grayscale/palette/other"
        ),
        "source_dpi": list(dpi) if dpi else None,
        "orientation": orientation,
        "aspect_ratio": round(width / height, 5),
        "blur_score": round(blur_score, 2),
        "noise_estimate": round(noise_score, 2),
        "edge_density": round(edge_density, 4),
        "face_count": face_count,
        "exposure_mean": round(brightness, 4),
        "quality_score": quality,
        "category": category,
        "category_confidence": category_confidence,
        "category_evidence": category_evidence,
        "warnings": warnings,
    }
