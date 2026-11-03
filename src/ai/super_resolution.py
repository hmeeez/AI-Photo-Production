from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import onnxruntime as ort
import tifffile
from PIL import Image, ImageCms


class InferenceSession(Protocol):
    def get_inputs(self) -> list[object]: ...

    def get_outputs(self) -> list[object]: ...

    def run(self, output_names: list[str], input_feed: dict[str, np.ndarray]) -> list[np.ndarray]: ...


@dataclass(frozen=True)
class InferenceRuntime:
    session: InferenceSession
    backend: str
    fallback_reason: str | None
    input_name: str
    output_name: str
    scale: int


class _OpenVINOInput:
    def __init__(self, name: str, shape: list[int | None]) -> None:
        self.name = name
        self.shape = shape


class OpenVINOInferenceSession:
    def __init__(self, model_path: Path, device: str = "GPU") -> None:
        import openvino as ov

        core = ov.Core()
        if device not in core.available_devices:
            raise RuntimeError(
                f"OpenVINO device {device!r} is unavailable. Found: {core.available_devices!r}"
            )
        model = core.read_model(str(model_path))
        self.compiled = core.compile_model(
            model,
            device,
            {"PERFORMANCE_HINT": "LATENCY"},
        )
        self.request = self.compiled.create_infer_request()
        self.input_port = self.compiled.input(0)
        self.output_port = self.compiled.output(0)
        self._inputs = [
            _OpenVINOInput(
                self.input_port.get_any_name(),
                [
                    dimension.get_length() if dimension.is_static else None
                    for dimension in self.input_port.partial_shape
                ],
            )
        ]
        self._outputs = [
            _OpenVINOInput(
                self.output_port.get_any_name(),
                [
                    dimension.get_length() if dimension.is_static else None
                    for dimension in self.output_port.partial_shape
                ],
            )
        ]

    def get_inputs(self) -> list[_OpenVINOInput]:
        return self._inputs

    def get_outputs(self) -> list[_OpenVINOInput]:
        return self._outputs

    def run(
        self,
        output_names: list[str],
        input_feed: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        input_name = self._inputs[0].name
        if input_name not in input_feed:
            raise ValueError(f"OpenVINO model input {input_name!r} was not provided.")
        values = self.request.infer({self.input_port: input_feed[input_name]})
        return [np.asarray(values[self.output_port])]


def model_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_upscaled_tiff(
    path: Path,
    expected_size: tuple[int, int],
    expected_profile: bytes,
) -> None:
    with tifffile.TiffFile(path) as check:
        page = check.pages[0]
        if (
            page.imagewidth != expected_size[0]
            or page.imagelength != expected_size[1]
            or page.samplesperpixel != 3
            or page.photometric != tifffile.PHOTOMETRIC.RGB
        ):
            raise OSError("Upscaled TIFF failed its dimensions or color-mode validation.")
        profile_tag = page.tags.get(34675)
        if profile_tag is None or profile_tag.value != expected_profile:
            raise OSError("Upscaled TIFF is missing its embedded sRGB profile.")


def create_cpu_session(model_path: Path) -> ort.InferenceSession:
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model does not exist: {model_path}")
    if model_path.suffix.lower() != ".onnx":
        raise ValueError("Select an ONNX (.onnx) super-resolution model.")
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = min(4, max(1, os.cpu_count() or 1))
    options.inter_op_num_threads = 1
    options.enable_mem_pattern = True
    options.enable_cpu_mem_arena = True
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    providers = session.get_providers()
    if providers != ["CPUExecutionProvider"]:
        raise RuntimeError(f"Expected CPU-only inference, but providers are {providers!r}.")
    return session


def create_vulkan_session(model_path: Path) -> ort.InferenceSession:
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model does not exist: {model_path}")
    if model_path.suffix.lower() != ".onnx":
        raise ValueError("Select an ONNX (.onnx) super-resolution model.")
    if "VulkanExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(
            "This ONNX Runtime build does not include VulkanExecutionProvider. "
            "Install a Windows build with Vulkan EP support."
        )
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = min(4, max(1, os.cpu_count() or 1))
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["VulkanExecutionProvider", "CPUExecutionProvider"],
    )
    providers = session.get_providers()
    if "VulkanExecutionProvider" not in providers:
        raise RuntimeError(
            f"VulkanExecutionProvider was requested but is not active: {providers!r}."
        )
    return session


def prepare_inference(model_path: Path, backend: str = "auto") -> InferenceRuntime:
    if backend not in {"auto", "onnx_cpu", "openvino_gpu", "onnx_vulkan"}:
        raise ValueError(f"Unsupported inference backend: {backend}")
    fallback_reason = None
    if backend in {"auto", "openvino_gpu"}:
        try:
            session = OpenVINOInferenceSession(model_path, "GPU")
            input_name, output_name, scale = validate_model(session)
            return InferenceRuntime(
                session, "OpenVINO Intel GPU", None, input_name, output_name, scale
            )
        except Exception as exc:
            if backend == "openvino_gpu":
                raise RuntimeError(f"OpenVINO Intel GPU initialization/test failed: {exc}") from exc
            fallback_reason = (
                f"Intel GPU unavailable or incompatible ({type(exc).__name__}: {exc})"
            )

    if backend == "onnx_vulkan" or backend == "auto":
        try:
            session = create_vulkan_session(model_path)
            input_name, output_name, scale = validate_model(session)
            return InferenceRuntime(
                session,
                "ONNX Runtime Vulkan",
                fallback_reason,
                input_name,
                output_name,
                scale,
            )
        except Exception as exc:
            if backend == "onnx_vulkan":
                raise RuntimeError(f"ONNX Runtime Vulkan initialization/test failed: {exc}") from exc
            vulkan_fallback_reason = (
                f"Vulkan unavailable or incompatible ({type(exc).__name__}: {exc})"
            )
            fallback_reason = (
                f"{fallback_reason}; {vulkan_fallback_reason}"
                if fallback_reason
                else vulkan_fallback_reason
            )

    session = create_cpu_session(model_path)
    input_name, output_name, scale = validate_model(session)
    return InferenceRuntime(
        session, "ONNX Runtime CPU", fallback_reason, input_name, output_name, scale
    )


def validate_model(session: InferenceSession, probe_size: int = 32) -> tuple[str, str, int]:
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("The selected model must have exactly one input and one output tensor.")
    input_info = inputs[0]
    output_info = outputs[0]
    input_shape = input_info.shape
    output_shape = output_info.shape
    if len(input_shape) != 4 or input_shape[1] != 3:
        raise ValueError("Model input must be a dynamic NCHW RGB tensor.")
    if len(output_shape) != 4 or output_shape[1] != 3:
        raise ValueError("Model output must be an NCHW RGB tensor.")
    if any(isinstance(dimension, int) and dimension > 0 for dimension in input_shape[2:4]):
        raise ValueError(
            "This model has fixed spatial input dimensions; use a dynamic-spatial ONNX export."
        )
    probe = np.zeros((1, 3, probe_size, probe_size), dtype=np.float32)
    result = session.run([output_info.name], {input_info.name: probe})[0]
    if result.ndim != 4 or result.shape[0] != 1 or result.shape[1] != 3:
        raise ValueError("Model returned an incompatible output tensor.")
    if result.shape[2] != result.shape[3] or result.shape[2] % probe_size:
        raise ValueError("Model must return a uniform integer-scale RGB image.")
    scale = result.shape[2] // probe_size
    if scale not in {2, 4}:
        raise ValueError(f"Only 2x and 4x super-resolution models are supported, got {scale}x.")
    return input_info.name, output_info.name, scale


def _predict_tile(
    session: InferenceSession,
    input_name: str,
    output_name: str,
    tile: np.ndarray,
    scale: int,
) -> np.ndarray:
    tensor = np.ascontiguousarray(tile.transpose(2, 0, 1)[None, ...], dtype=np.float32) / 255.0
    output = session.run([output_name], {input_name: tensor})[0]
    expected_shape = (1, 3, tile.shape[0] * scale, tile.shape[1] * scale)
    if output.shape != expected_shape:
        raise ValueError(
            f"Model output shape {output.shape!r} does not match expected {expected_shape!r}."
        )
    if not np.isfinite(output).all():
        raise ValueError("Model produced NaN or infinite pixel values.")
    rgb = output[0].transpose(1, 2, 0)
    if float(rgb.min()) < -0.5 or float(rgb.max()) > 1.5:
        raise ValueError("Model output is outside a plausible normalized RGB range.")
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def _upscale_to_tiff(
    image: Image.Image,
    destination: Path,
    session: InferenceSession,
    tile_size: int,
    overlap: int,
    progress: Callable[[str], None] | None = None,
    model_info: tuple[str, str, int] | None = None,
) -> tuple[int, int, int]:
    if tile_size < 64 or tile_size > 1024:
        raise ValueError("AI tile size must be between 64 and 1024 pixels.")
    if overlap < 0 or overlap >= tile_size // 2:
        raise ValueError("Tile overlap must be non-negative and less than half the tile size.")
    input_name, output_name, scale = model_info or validate_model(session)
    source = np.asarray(image.convert("RGB"))
    height, width = source.shape[:2]
    output_width, output_height = width * scale, height * scale
    output_path = destination.with_name(f"{destination.stem}.partial{destination.suffix}")
    memory_map_path = destination.with_name(f"{destination.stem}.pixels.npy")
    output_map = np.lib.format.open_memmap(
        memory_map_path,
        mode="w+",
        dtype=np.uint8,
        shape=(output_height, output_width, 3),
    )
    columns = (width + tile_size - 1) // tile_size
    rows = (height + tile_size - 1) // tile_size
    total_tiles = rows * columns
    tile_number = 0
    try:
        for core_y in range(0, height, tile_size):
            core_bottom = min(height, core_y + tile_size)
            for core_x in range(0, width, tile_size):
                core_right = min(width, core_x + tile_size)
                tile_number += 1
                y0 = max(0, core_y - overlap)
                x0 = max(0, core_x - overlap)
                y1 = min(height, core_bottom + overlap)
                x1 = min(width, core_right + overlap)
                prediction = _predict_tile(
                    session, input_name, output_name, source[y0:y1, x0:x1], scale
                )
                crop_top = (core_y - y0) * scale
                crop_left = (core_x - x0) * scale
                crop_bottom = crop_top + (core_bottom - core_y) * scale
                crop_right = crop_left + (core_right - core_x) * scale
                output_map[
                    core_y * scale : core_bottom * scale,
                    core_x * scale : core_right * scale,
                ] = prediction[crop_top:crop_bottom, crop_left:crop_right]
                if progress:
                    progress(f"CPU AI upscale tile {tile_number}/{total_tiles}")

        output_map.flush()
        srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
        with tifffile.TiffWriter(
            output_path,
            bigtiff=output_width * output_height * 3 >= 4_000_000_000,
        ) as writer:
            source_dpi = image.info.get("dpi")
            output_resolution = (
                (float(source_dpi[0]) * scale, float(source_dpi[1]) * scale)
                if source_dpi and len(source_dpi) == 2
                else None
            )
            writer.write(
                output_map,
                photometric="rgb",
                tile=(256, 256),
                compression="deflate",
                iccprofile=srgb.tobytes(),
                resolution=output_resolution,
                resolutionunit="INCH" if output_resolution else None,
                metadata=None,
                maxworkers=1,
            )
        _validate_upscaled_tiff(
            output_path,
            (output_width, output_height),
            srgb.tobytes(),
        )
        output_path.replace(destination)
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    finally:
        del output_map
        memory_map_path.unlink(missing_ok=True)
    return output_width, output_height, scale


def upscale_with_session(
    image: Image.Image,
    session: InferenceSession,
    destination: Path,
    tile_size: int = 256,
    overlap: int = 16,
    progress: Callable[[str], None] | None = None,
    model_info: tuple[str, str, int] | None = None,
) -> tuple[int, int, int]:
    return _upscale_to_tiff(
        image, destination, session, tile_size, overlap, progress, model_info
    )


def upscale_image(
    image: Image.Image,
    model_path: Path,
    destination: Path,
    tile_size: int = 256,
    overlap: int = 16,
    progress: Callable[[str], None] | None = None,
) -> tuple[int, int, int]:
    return upscale_with_session(
        image,
        create_cpu_session(model_path),
        destination,
        tile_size,
        overlap,
        progress,
    )
