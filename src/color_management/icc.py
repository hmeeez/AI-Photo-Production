from __future__ import annotations

from pathlib import Path

from PIL import ImageCms


def validate_cmyk_profile(path: Path) -> ImageCms.ImageCmsProfile:
    if not path.is_file():
        raise FileNotFoundError(f"ICC profile does not exist: {path}")
    try:
        profile = ImageCms.getOpenProfile(str(path))
    except (OSError, ImageCms.PyCMSError) as exc:
        raise ValueError(f"Could not read ICC profile: {exc}") from exc
    if profile.profile.xcolor_space.strip() != "CMYK":
        raise ValueError("The selected printer profile is not a CMYK profile.")
    return profile
