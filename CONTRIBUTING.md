# Contributing

## Development setup

This application targets 64-bit Windows and Python 3.11.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

For optional Intel GPU inference, install the OpenVINO runtime:

```powershell
python -m pip install -r requirements-openvino.txt
```

## Checks

```powershell
python -m ruff check .
python -m unittest discover -s tests -v
```

The real-model integration tests run when the model file is available and
otherwise skip. Obtain it with
`.\SCRIPTS\download_realesrgan_model.ps1`; inspect its provenance and license
before redistributing model weights.

## Pull requests

Keep changes focused, add regression tests for bug fixes, and update the
README when behavior or setup changes. Do not commit photographs, generated
output, printer profiles, local configuration, secrets, or model binaries.
