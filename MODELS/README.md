# Real-ESRGAN x4 ONNX model

- File: `real_esrgan_x4.onnx`
- Source: [SceneWorks/real-esrgan-onnx](https://huggingface.co/SceneWorks/real-esrgan-onnx)
- Model card: [model README](https://huggingface.co/SceneWorks/real-esrgan-onnx/blob/main/README.md)
- License reported by the model card: BSD-3-Clause
- Model-card SHA-256: `5c586662929cbc686c1a5c38d9c060dbdb4ea5863a1f7672b8c0761e6b89c033`
- Input: dynamic `[1, 3, h, w]`, float32 RGB normalized to `[0, 1]`
- Output: dynamic `[1, 3, 4h, 4w]`

The model card attributes the weights to the canonical Real-ESRGAN x4plus
weights and reports parity testing. The application independently validates
the ONNX graph shape and runs CPU/OpenVINO inference probes. It does not claim
that generated texture is faithful recovered detail.

To download a fresh copy, run
`.\SCRIPTS\download_realesrgan_model.ps1`. The script validates SHA-256 before
installing and does not replace an existing model.
