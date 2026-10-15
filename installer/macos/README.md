# macOS daemon package

Build the package on a Mac with Python 3.12 and the Xcode command line tools. The build uses the architecture of the Python installation; run it with an Apple Silicon Python to produce an `arm64` package.

Install the project requirements and PyInstaller into the selected Python environment:

```bash
python3.12 -m pip install -r requirements.txt pyinstaller
bash installer/macos/build_pkg.sh
```

An optional first argument overrides the version read from `setup.py`. The output package is written to `installer/macos/dist/` and installs the self-contained daemon under `/Applications/NamselOCR/namsel_daemon/`. The package does not require Python on the target Mac.

If `namsel_BUDA_OCR/line_model/` exists, its ONNX and JSON files are copied into the bundle. Without those files, single-line requests use the existing recognizer.
