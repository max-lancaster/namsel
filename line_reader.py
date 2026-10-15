# encoding: utf-8
"""Namsel's line reader: one printed line read whole, left to right, by a small CNN + BiLSTM with CTC.

Trained by Notebooks/namsel_line_reader_colab.ipynb on OpenPecha's line collections plus rendered lines that mix
Tibetan with Western digits, Latin words (English, Portuguese, IAST, Wylie), punctuation and symbols — so it reads
"1959", "(skt. śūnyatā)" or "p. 45" inside a Tibetan line, and never cuts a stack apart the way the
segmenter + stack classifier can ("བཅོས" read as "ག་ཙུས").

Used for single lines (force_single_line) when its model is installed in namsel_BUDA_OCR/line_model/
(namsel_line.onnx, charset.json, spec.json); pages, and single lines without the model, keep the segmenter +
classifier. The steps below follow spec.json, which the training notebook writes with the model.
"""
import json
import os
import unicodedata

import cv2
import numpy as np

MODEL_FILES = ("namsel_line.onnx", "charset.json", "spec.json")


def find_model_dir():
    """namsel_BUDA_OCR/line_model beside this package (app layout) or inside it (standalone repo), else None."""
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (os.path.join(here, "namsel_BUDA_OCR", "line_model"),
              os.path.join(here, "..", "namsel_BUDA_OCR", "line_model")):
        if all(os.path.isfile(os.path.join(d, f)) for f in MODEL_FILES):
            return os.path.abspath(d)
    return None


def gray_of(img):
    """uint8 gray, ink dark: the darkest of R, G, B; light-on-dark lines inverted (median below 110)."""
    a = np.asarray(img)
    if a.ndim == 3:
        a = a[..., :3].min(axis=2)
    a = a.astype(np.uint8)
    if np.median(a) < 110:
        a = 255 - a
    return np.ascontiguousarray(a)


class LineReader(object):
    def __init__(self, model_dir, threads=2):
        import onnxruntime as ort
        with open(os.path.join(model_dir, "charset.json"), encoding="utf-8") as f:
            self.charset = json.load(f)
        with open(os.path.join(model_dir, "spec.json"), encoding="utf-8") as f:
            spec = json.load(f)
        self.height = int(spec["height"])
        self.stride = int(spec["stride"])
        self.max_w = int(spec["max_width"])
        self.min_w = int(spec["min_width"])
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(os.path.join(model_dir, "namsel_line.onnx"), opts,
                                            providers=["CPUExecutionProvider"])

    def to_input(self, gray):
        """[1, 1, height, W] float32: height fixed, aspect kept, (255 - gray) / 255 (ink 1, paper 0)."""
        h, w = gray.shape
        nw = int(round(w * self.height / max(1, h)))
        nw = max(self.min_w, min(self.max_w, nw))
        interp = cv2.INTER_AREA if h > self.height else cv2.INTER_LINEAR
        r = cv2.resize(gray, (nw, self.height), interpolation=interp)
        return ((255.0 - r.astype(np.float32)) / 255.0)[None, None]

    def decode(self, logits, width):
        """Greedy CTC over the line's own steps: best class per step, repeats merged, blanks (0) dropped."""
        best = logits[0, :width // self.stride].argmax(-1)
        chars, prev = [], 0
        for t in best:
            if t != prev and t != 0:
                chars.append(self.charset[t])
            prev = t
        return " ".join(unicodedata.normalize("NFC", "".join(chars)).split())

    def read_gray(self, gray):
        x = self.to_input(gray)
        logits = self.session.run(None, {"line": x})[0]
        return self.decode(logits, x.shape[-1])

    def read_path(self, path):
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError("cannot read " + path)
        if img.ndim == 3:
            img = img[..., :3][..., ::-1]  # BGR -> RGB (the minimum over channels does not care, but keep it honest)
        if img.dtype != np.uint8:  # 16-bit scans: stretched to 0-255 as in training (line_lib.gray_of)
            a = img.astype(np.float32)
            img = (255 * (a - a.min()) / max(1.0, float(a.max() - a.min()))).astype(np.uint8)
        return self.read_gray(gray_of(img))


_READER = None


def shared_reader():
    """The reader, loaded once per process; None when its model is not installed or does not load."""
    global _READER
    if _READER is None:
        d = find_model_dir()
        if d is None or os.environ.get("NAMSEL_LINE_READER") == "0":
            _READER = False
        else:
            try:
                _READER = LineReader(d)
            except Exception as e:  # a broken model must not take the daemon down: the stack reader still works
                import sys
                sys.stderr.write("[line_reader] model in %s did not load: %s\n" % (d, e))
                _READER = False
    return _READER or None
