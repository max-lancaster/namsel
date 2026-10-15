# PyInstaller spec for the standalone macOS Namsel daemon.
import os
from pathlib import Path

ROOT = Path(SPECPATH).resolve().parents[1]

def add_files(paths, destination):
    return [(str(path), destination) for path in paths if path.is_file()]

data_files = []
for name in (
    "3gram_stack_dict.json.gz",
    "allchars.json.gz",
    "extended_char_dig.json.gz",
    "label_chars.json.gz",
    "syllable_bigram.json.gz",
    "termset_syllables.json.gz",
    "logistic-cls",
    "rbf-cls",
):
    path = ROOT / name
    if not path.is_file():
        raise SystemExit("namsel_daemon.spec: missing runtime data file: %s" % path)
    data_files.extend(add_files([path], "."))

data_files.extend(add_files(ROOT.glob("*.npz"), "."))
data_files.extend(add_files((ROOT / "features").glob("*.npz"), "features"))

confs_dir = ROOT / "confs"
if confs_dir.is_dir():
    for path in confs_dir.rglob("*"):
        if path.is_file():
            data_files.extend(add_files([path], str(path.parent.relative_to(ROOT))))

model_dir = ROOT / "namsel_BUDA_OCR"
if model_dir.is_dir():
    data_files.extend(add_files(model_dir.glob("*.onnx"), "namsel_BUDA_OCR"))
    data_files.extend(add_files(model_dir.glob("*.pth"), "namsel_BUDA_OCR"))
    data_files.extend(add_files(model_dir.glob("*.json"), "namsel_BUDA_OCR"))
    line_model_dir = model_dir / "line_model"
    if line_model_dir.is_dir():
        for path in line_model_dir.iterdir():
            if path.is_file() and path.suffix in (".onnx", ".json"):
                data_files.extend(add_files([path], "namsel_BUDA_OCR/line_model"))

hidden_imports = [
    "char_placement",
    "classify",
    "combine_boxes",
    "config_manager",
    "config_util",
    "contour_processor",
    "feature_extraction",
    "layout_detector",
    "line_breaker",
    "line_reader",
    "namsel",
    "namsel_BUDA_OCR",
    "namsel_BUDA_OCR.model",
    "namsel_BUDA_OCR.predict",
    "page_elements2",
    "recognize",
    "root_based_finder",
    "scale_calculator",
    "scale_invariant_preprocessing",
    "segment",
    "segment_experimental",
    "termset",
    "tparser",
    "utils",
    "utils_extra.scantailor_multicore",
    "yik",
    "fast_utils",
    "sobel_features",
    "transitions",
    "viterbi_cython",
    "zernike_moments",
    "cv2",
    "joblib",
    "lxml.etree",
    "onnxruntime",
    "PIL",
    "scipy.special",
    "skimage.morphology",
    "sklearn.linear_model",
    "sklearn.svm",
]

a = Analysis(
    [str(ROOT / "daemon.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=data_files,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["IPython", "matplotlib", "notebook", "pytest", "tkinter"],
    noarchive=False,
)

pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="namsel_daemon",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="namsel_daemon",
)
