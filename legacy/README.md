# Legacy

Files kept for reference that are **no longer on any code path**. Nothing in the
engine imports, opens or ships anything in this folder. They are retained because
they document how the pre-modernisation engine worked and because some of them are
the only surviving copy of a superseded artifact.

Do not restore anything from here without reading the reason it was retired.

| Folder | What | Why it was retired |
|---|---|---|
| `install-scripts-python2/` | `windows_install.bat`, `ubuntu_install.sh`, `mac_install.sh` | Python 2.7-era installers. They install Python 2.7, GTK+ 2.22, pygtk/py2cairo and `scikit-learn==0.18.1`, and they reference two things that no longer exist in this repo: an `install_windows/` directory and `datasets/datapickles.zip`. The engine is Python 3.12 and pickle-free; running any of these would not produce a working install. |
| `zernike-skops/` | `Bpqk17.skops`, `D_matrix.skops`, `Ipi32.skops` | Superseded by `features/zernike_features.npz`, which holds byte-identical data under the same three key names. Loading them made `skops` a hard, module-level import of the whole engine, so an environment without it could not import `line_breaker` → `namsel` → `daemon` at all. `feature_extraction.py` now reads the `.npz` with `allow_pickle=False`. |
| `zernike-scaler-backups/` | `zernike_scaler-latest.backup`, `.512dim.backup`, `.backup_current` | Backups of `zernike_scaler-latest`, which is **not present in this repo and not shipped**. `feature_extraction.py` guards its load with `os.path.exists` and sets `SCALER_DEFINED = False` when it is absent, which is the current behaviour everywhere. Restoring one of these would silently turn Zernike feature scaling back on and change recognition output. |
| `portable-char-maps/` | `portable_char_to_dig.pkl`, `portable_dig_to_char.pkl` | Referenced by no code in this repo (verified by grep across every `.py`). The live character maps are `allchars.json.gz` and `label_chars.json.gz`, loaded through `safe_model_io.load_model` (gzip + JSON, data-only). |
| `orphaned-modules/` | `enhanced_feature_extraction.py`, `check_results.py`, `generate_scaler.py` | Nothing imports them and nothing references them by name. `enhanced_feature_extraction.py` is an unused 414-line variant of the live 218-line `feature_extraction.py`. |
| `cp310-extensions/` | `*.cp310-win_amd64.pyd` (5) | Compiled for CPython 3.10. The repo targets Python 3.12, which only loads `cp312` tags, so these can never be imported. They were never tracked in git and are archived here for local reference only. |

## What replaced the pickle formats

| data | was | now |
|---|---|---|
| character maps, n-gram & bigram tables | `pickle` / `shelve` | gzip + JSON (`safe_model_io.py`) |
| Zernike feature matrices | `pickle`, then `skops` | NumPy `.npz`, `allow_pickle=False` (`features/zernike_features.npz`) |
| CNN training datasets | `pickle` | NumPy `.npy` |
| sklearn classifiers | `pickle` | `joblib` (`logistic-cls`, `rbf-cls`) |

The original `.pkl` training datasets are **not** in this folder. They live outside the
repo in the pre-migration working copies (`datasets/datapickles.zip` and the loose
`datasets/*.pkl` there); `datasets/` is git-ignored. The migration to `.npy` was verified
array-for-array: all 58 converted files compare equal to their pickles, 19,592 rows total.
