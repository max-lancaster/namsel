# Namsel-BUDA OCR

**Modern OCR for machine-print Tibetan — a fully modernized fork of [Namsel OCR](https://github.com/thubtenrigzin/namsel-ocr), ported to Python 3.12 with a convolutional-neural-network character recognizer and safe, pickle-free model formats.**

Namsel-BUDA is maintained by **Tashi Rabten** ([Associação BUDA](https://github.com/TashiRabten)) as the Tibetan OCR engine of the **BUDA** application suite. It keeps the proven Namsel OCR pipeline — page layout analysis, line segmentation, feature extraction, and HMM/probability-based recognition with post-processing — and replaces the aging hand-crafted classifier with a trained CNN, while modernizing the entire code base for current Python and safe model loading.

---

## Where it is used

Namsel-BUDA is not a stand-alone research toy — it is the **OCR engine embedded in [TradutorBUDA](https://TashiRabten.github.io/BUDA_APPs_Port/)**, a Java Tibetan translation and text-processing application for Buddhist terminology workflows. TradutorBUDA calls this engine (via its daemon interface) to turn scanned Tibetan pages into editable Unicode text, which then feeds dictionary lookup, transliteration, tokenization, and translation.

- **TradutorBUDA & the BUDA suite** — [portfolio](https://TashiRabten.github.io/BUDA_APPs_Port/) · [demo video](https://youtu.be/6blrTYdL17w)
- **Interactive OCR demo** — [Hugging Face Space `trabten/tibetan-ocr`](https://huggingface.co/spaces/trabten/tibetan-ocr) · [presentation](https://youtu.be/UMSwbuFfDLk)
- **Models & datasets** — related Tibetan OCR models and character datasets are published on Hugging Face under [`trabten`](https://huggingface.co/trabten)

If you are a researcher or developer, the engine here is the production source behind those demos.

---

## What was modernized

The upstream Namsel OCR is an excellent but aging Python 2.7 code base. This fork brings it fully up to date:

- **Python 3.12.** Ported from Python 2.7. The five accelerated modules each ship a pure-Python fallback (`fast_utils.py`, `sobel_features.py`, `transitions.py`, `viterbi_cython.py`, `zernike_moments.py`), so a fresh clone runs with no compilation step at all. Building the Cython extensions is optional and only makes it faster — compiled binaries are **not** committed to this repository.
- **CNN character recognizer.** A PyTorch convolutional model (**1,020 Tibetan character classes, ~95.5% validation accuracy**) replaces the legacy scikit-learn logistic/RBF classifiers. The engine loads the CNN automatically when present and falls back to the sklearn classifier if it is not. See [`namsel_BUDA_OCR/`](namsel_BUDA_OCR).
- **No `pickle` in the load path.** Every bundled model and dataset now loads through a **data-only** format that cannot execute code on load:
  | Data | Old | New |
  |---|---|---|
  | character maps, n-gram & bigram tables | `pickle` / `shelve` | **gzip + JSON** (`safe_model_io.py`) |
  | Zernike feature matrices | `pickle`, then `skops` | **NumPy `.npz`**, `allow_pickle=False` (`features/zernike_features.npz`) |
  | CNN training datasets | `pickle` | **NumPy `.npy`** |
  | classifiers | `pickle` | **joblib** |

  The safe formats are also smaller (e.g. the syllable-bigram table shrank from 57 MB to 11 MB).
- **Daemon mode.** [`daemon.py`](daemon.py) exposes a request/response OCR service (`process_request`, `segment_lines`) used by TradutorBUDA and the Docker image.
- **Hardened engine.** Numerous runtime-robustness fixes and substantial complexity reductions across the segmentation and recognition code.

### Recognition backends

| Backend | Files | Notes |
|---|---|---|
| **CNN (default)** | `namsel_BUDA_OCR/best_model.pth` + `label_mapping.json` | PyTorch; `best_model.onnx` / `best_model_int8.onnx` provided for ONNX Runtime |
| sklearn (fallback) | `logistic-cls`, `rbf-cls` (joblib) | used automatically when the CNN model is absent |

---

## Install

Requires **Python 3.12** and the packages in `requirements.txt` (`numpy`, `opencv-python`, `scikit-learn`, `scikit-image`, `scipy`, `Pillow`, `joblib`, `onnxruntime`, `matplotlib`, `lxml`). `torch` is optional — it is only needed to load `best_model.pth`; the bundled `best_model.onnx` runs on `onnxruntime` alone. `skops` is **no longer required**.

No compiler is needed: every accelerated module has a pure-Python fallback. To build the faster Cython extensions you additionally need `Cython` and a C toolchain.

```bash
pip install -r requirements.txt

# Optional: build the Cython extensions in place for speed.
# Needs Cython (not in requirements.txt) and a C compiler. Without this step the
# pure-Python fallbacks are used automatically.
pip install Cython
python setup.py build_ext --inplace
```

### Build a standalone macOS daemon package

On macOS, install the Python 3.12 requirements and run `bash installer/macos/build_pkg.sh`. The package is built for the architecture of the Python installation; use Apple Silicon Python for an `arm64` build. See [`installer/macos/README.md`](installer/macos/README.md) for details.

The original project also publishes ready-to-run Docker images:

```bash
docker pull thubtenrigzin/namsel-ocr:[tag]
```

## Quickstart

Preprocess a folder of page images, then run OCR:

```bash
python namsel.py preprocess /path/to/myfolder
python namsel.py recognize-volume --page_type=book --format=text /path/to/myfolder/out
```

OCR a single page:

```bash
python namsel.py recognize-page --page_type=book --format=text /path/to/page-01.tif
```

Results are written to `ocr_output.txt`.

---

## Preprocessing

Scanned documents (PDFs or images) must be cleaned and thresholded to black-and-white before OCR. Black-and-white TIFF is the preferred input.

**Scanning tips:** scan in black and white at 400–600 dpi; deskew and crop in the scanner software; save as sequentially-named TIFF (`001.tif`, `002.tif`, …).

**From PDF:** convert pages to black-and-white images, e.g. with Ghostscript:

```bash
gs -r600x600 -sDEVICE=tiffg4 -sOutputFile=ocr_%04d.tif -dBATCH -dNOPAUSE mytibetanfile.pdf
```

**With Scantailor** ([scantailor](https://github.com/scantailor/scantailor)) — page splitting, deskewing, content isolation, noise removal, thresholding. For batch multicore processing use the bundled helper:

```bash
python scantailor_multicore.py <my-image-folder> [threshold]
```

A positive threshold thickens strokes; a negative one thins them (good range: −40 to 40). This produces an `out/` folder of cleaned images. You can also drive Scantailor from Namsel's own `preprocess` command:

```bash
python namsel.py preprocess --layout=double --st_threshold=-15 /path/to/tiffs
```

## OCR command-line options

```bash
python namsel.py recognize-page  mytibetantextimage.tif      # single page
python namsel.py recognize-volume folder-of-tiff-images      # whole volume
```

Key tunable parameters:

- **`--page_type`** — `book` or `pecha`. Auto-detected from page dimensions if omitted.
- **`--recognizer`** — `hmm` (use most of the time) or `probout` (better for unusual character combinations / complex segmentation).
- **`--line_break_method`** — `line_cut` (book pages) or `line_cluster` (pecha and book pages). Auto-chosen if omitted.
- **`--break_width`** — controls how horizontally-connected stacks are segmented. Typical good values 2–3.5; high values under-segment, low values over-segment.
- **`--segmenter`** — segmentation strategy; `stochastic` (default) is almost always best.
- **`--low_ink`** — compensate for poorly-inked texts where glyph strokes are broken.
- **`--line_cluster_pos`** — `top` or `center`; used with `line_cluster`.
- **`--postprocess`** — experimental tsek-insertion pass (can mangle otherwise-good output).
- **`--detect_o`** — temporarily remove long na-ro vowels that inflate width measurements before segmentation.
- **`--clear_hr`** — detect and remove a horizontal rule / header line (use with caution on pecha).
- **`--line_cut_inflation`** — dilation iterations in line-cut (rarely changed; default 4).

---

## Training the CNN

The recognizer is trained from the Tibetan character datasets in `datasets/` — each sample is a label plus a 32×32 grayscale glyph (label + 1024 pixels). See [`namsel_BUDA_OCR/`](namsel_BUDA_OCR):

- `dataset.py` — dataset loading (`.npy`, data-only) and CPU-friendly augmentation
- `model.py` — the CNN architecture
- `predict.py` — inference (`TibetanCNNPredictor`)

Datasets are stored as data-only NumPy `.npy` (label + 1024 pixels per sample).
Training is GPU/Colab-friendly.

---

## Legacy files

Superseded artifacts are kept under [`legacy/`](legacy) rather than deleted, with a
[README](legacy/README.md) recording why each was retired. Nothing there is on any code
path: the Python 2.7 install scripts, the `.skops` Zernike matrices (replaced by
`features/zernike_features.npz`), the `zernike_scaler-latest` backups, the unused
`portable_*.pkl` character maps, three orphaned modules, and the CPython 3.10 extension
binaries. Do not restore anything from `legacy/` without reading the reason it was retired.

## About the BUDA project

The **BUDA** (Associação BUDA) suite is a set of tools for Tibetan Buddhist study and translation — TradutorBUDA (translation/text editor), GlossarioBUDA (glossary), and related utilities. Namsel-BUDA provides their shared Tibetan OCR capability. See the [BUDA Apps portfolio](https://TashiRabten.github.io/BUDA_APPs_Port/) and [GitHub](https://github.com/TashiRabten).

## Credits & lineage

This project is a fork of **Namsel OCR**, created by **Zach Rowinski** and contributors, with Docker packaging by **Thubten Rigzin** ([thubtenrigzin/namsel-ocr](https://github.com/thubtenrigzin/namsel-ocr)). The original project is described in [*Namsel: An Optical Character Recognition System for Tibetan Text*](https://escholarship.org/uc/item/6d5781k5) (Himalayan Linguistics); OCR'd Tibetan e-texts are available via [BDRC/TBRC](http://tbrc.org).

The namsel_buda modernization — Python 3.12 port, CNN recognizer, pickle-free model/data formats, and daemon integration — is by **Tashi Rabten** for the Associação BUDA suite.

### About the name

*Namsel* renders the Tibetan རྣམ་གསལ (*rnam gsal*) — "making clear the details," "thoroughly illuminating." *BUDA* marks this fork's home in the BUDA project ecosystem.

## License

Retains the original Namsel OCR license (see [`LICENSE`](LICENSE)). Bundled third-party components retain their respective licenses.
