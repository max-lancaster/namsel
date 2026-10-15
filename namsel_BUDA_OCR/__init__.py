"""Namsel BUDA OCR - CNN-based Tibetan character recognition.

Replaces the hand-crafted feature pipeline (Zernike + Sobel + transitions)
with a lightweight CNN that learns features directly from 32x32 character images.

Usage:
    # Training
    python -m namsel_BUDA_OCR.train --data-dir namsel_ocr/datasets

    # Inference (drop-in replacement for sklearn classifier)
    from namsel_BUDA_OCR.predict import TibetanCNNPredictor
    predictor = TibetanCNNPredictor('best_model.pth', 'label_mapping.json')
    log_probs = predictor.predict_log_proba(image_32x32)
"""

try:
    from .model import TibetanCNN
except ImportError:
    TibetanCNN = None  # PyTorch not available

from .predict import TibetanCNNPredictor

__all__ = ["TibetanCNN", "TibetanCNNPredictor"]
