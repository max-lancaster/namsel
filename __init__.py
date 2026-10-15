# Namsel OCR Package
# Prevent joblib/loky from running wmic/powershell to detect CPU cores on Windows
# (causes ugly traceback on some Windows installs where wmic is unavailable)
import os
if os.name == 'nt' and 'LOKY_MAX_CPU_COUNT' not in os.environ:
    os.environ['LOKY_MAX_CPU_COUNT'] = str(max(1, (os.cpu_count() or 4) - 1))

from .namsel import run_recognize_remote, PageRecognizer
from .config_manager import Config, default_config

__all__ = ["run_recognize_remote", "PageRecognizer", "Config", "default_config"]

__version__ = "1.0.0"