# AI Photo Production

Windows desktop app for local photo inspection, AI upscaling, ICC-managed
print preparation, and optional Adobe Illustrator/PDF export.

## Features

- Sequential photo batches with a SQLite job queue, retry, pause/resume, and
  optional folder monitoring.
- Conservative image analysis and traditional enhancement; reports include
  warnings and effective source PPI.
- Tiled ONNX x2/x4 super-resolution with CPU inference, optional OpenVINO
  Intel GPU acceleration, and optional Vulkan through a custom ONNX Runtime
  build that exposes its Vulkan execution provider.
- A1 and custom-size CMYK TIFF output with margins, bleed, optional crop marks,
  a user-selected printer ICC profile, and tiled disk streaming.
- Optional Illustrator automation for embedded, proportional placement and
  PDF export through an installed PDF/X-4 preset.
- Per-image JSON reports and before/after previews.

AI enhancement may synthesize plausible texture; inspect faces, text, logos,
and artwork before production use. The face/scene labels currently use
heuristics and are not trained semantic classification.

## Requirements

- 64-bit Windows and Python 3.11.
- Optional Intel GPU acceleration: an Intel graphics device and OpenVINO.
- Optional AI upscaling: the separately downloaded Real-ESRGAN x4 ONNX model.
- Optional CMYK production: a printer-provided CMYK ICC profile.
- Optional Illustrator/PDF: Adobe Illustrator with COM scripting enabled and
  an installed PDF/X-4 preset.

## Install and run

From PowerShell in the project directory:

```powershell
.\install.ps1
.\SCRIPTS\download_realesrgan_model.ps1
.\launch.ps1
```

For optional Intel GPU inference, install OpenVINO with:

```powershell
.\install.ps1 -EnableOpenVINO
```

Place photographs in `INPUT` (or choose another input folder in the app).
Generated files go to `OUTPUT` by default. Existing outputs are not overwritten.
User-specific settings are stored in the ignored local `config.json`; the
tracked `config.example.json` contains portable defaults.

## Output folders

| Folder | Contents |
|---|---|
| `01_ENHANCED` | Conservatively enhanced source-size images |
| `02_UPSCALED` | Optional AI-upscaled TIFFs |
| `03_PRINT_READY` | Optional A1 or custom-size CMYK TIFFs |
| `04_ILLUSTRATOR` | Optional native `.ai` files |
| `05_PDF` | Optional Illustrator PDF/X-4 exports |
| `06_REPORTS` | JSON analysis and processing reports |
| `07_PREVIEWS` | Before/after previews |

## Development

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m unittest discover -s tests -v
```

The Windows GitHub Actions workflow runs lint and tests on Python 3.11.
Real-model tests run when the model file exists; otherwise they skip. See
[CONTRIBUTING.md](CONTRIBUTING.md) for development and pull request guidance.

To run the real-model integration smoke test with a printer profile and
detected Illustrator installation:

```powershell
python .\SCRIPTS\integration_smoke.py `
  --icc-profile "C:\path\to\printer.icc" `
  --illustrator
```

It defaults to 72 PPI; specify `--ppi 700` for a high-resolution run. Test
outputs are created below `TEMP\integration-smoke`.

## Known limitations

- GFPGAN face restoration and trained semantic image classification are not
  implemented. Candidate model/code terms have not been verified for
  commercial production, so those assets are intentionally excluded.
- The default ONNX Runtime dependency does not expose a Vulkan provider.
  Vulkan is selectable only when a compatible custom runtime build is installed.
  Automatic mode tries Intel GPU, then Vulkan if present, then CPU.
- PDF/X-4 markers, output intent, and page dimensions are checked, but this
  is not an independent certified PDF/X-4 preflight; no independent validator
  is currently integrated.
- The application has been tested on this Windows workstation; compatibility
  with other Windows/Illustrator/driver configurations is not guaranteed.

## Model and repository licensing

Application source code is licensed under MIT; see [LICENSE](LICENSE).
Model weights are separate assets, excluded from Git, and downloaded with a
SHA-256 check. Their provenance and reported license are documented in
[MODELS/README.md](MODELS/README.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
User photos, generated results, local settings, ICC profiles, and model files
are excluded by `.gitignore`.
