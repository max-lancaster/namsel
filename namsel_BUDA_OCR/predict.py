"""Inference module for CNN-based Tibetan character recognition.

Provides a drop-in replacement for the sklearn classifier used in
Namsel's segment.py and recognize.py. The TibetanCNNPredictor class
exposes the same interface as sklearn's LogisticRegression:
    - predict_log_proba(x) -> (1, n_classes) log probability array
    - classes_ -> array of original Namsel label IDs

Supports three inference backends (fastest to slowest):
    1. ONNX Runtime (2-3x faster than PyTorch on CPU)
    2. PyTorch with INT8 quantization (~2x faster than FP32)
    3. PyTorch FP32 (fallback)

Usage:
    from namsel_BUDA_OCR.predict import TibetanCNNPredictor

    predictor = TibetanCNNPredictor('best_model.pth', 'label_mapping.json')

    # Single image (drop-in for sklearn)
    log_probs = predictor.predict_log_proba(image_32x32)

    # Batch of images (much faster for processing a full line)
    log_probs = predictor.predict_log_proba_batch(list_of_images)

    # Export for fastest CPU inference
    predictor.export_onnx('model.onnx')
"""

import json
import os

import numpy as np

# PyTorch is optional — only needed for .pth loading and export, not ONNX inference
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from .model import TibetanCNN
except ImportError:
    try:
        from model import TibetanCNN
    except ImportError:
        TibetanCNN = None  # Not needed for ONNX-only mode


class TibetanCNNPredictor:
    """CNN-based predictor compatible with Namsel's sklearn classifier interface.

    Accepts 32x32 character images and outputs predictions in the same
    format as sklearn's LogisticRegression, making it a drop-in replacement
    for the existing pipeline.

    On init, automatically selects the fastest available backend:
        ONNX Runtime > PyTorch quantized > PyTorch FP32
    """

    def __init__(self, model_path, mapping_path, device=None, backend='auto'):
        """Load trained model and label mapping.

        Args:
            model_path: path to best_model.pth checkpoint
            mapping_path: path to label_mapping.json
            device: torch device (default: cpu)
            backend: 'auto', 'onnx', 'pytorch', or 'quantized'
        """
        self.device = device if device else (torch.device('cpu') if TORCH_AVAILABLE else None)

        # Load label mapping
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)

        self.label_to_idx = {int(k): v for k, v in mapping['label_to_idx'].items()}
        self.idx_to_label = {int(k): v for k, v in mapping['idx_to_label'].items()}
        self.num_classes = mapping['num_classes']

        # classes_ array for sklearn compatibility (used by Viterbi decoder)
        self._classes = np.array([self.idx_to_label[i] for i in range(self.num_classes)])

        # Store model path for ONNX export
        self._model_path = model_path
        self._checkpoint = None

        # Select backend
        self._onnx_session = None
        self._backend = 'pytorch'
        val_acc = self._select_backend(model_path, backend)

        print(f"Loaded CNN model: {self.num_classes} classes, "
              f"val_acc={val_acc}, backend={self._backend}")

    def _select_backend(self, model_path, backend):
        """Resolve the inference backend (auto/onnx/quantized/pytorch); returns val_acc for logging."""
        val_acc = 'N/A'
        if backend == 'auto':
            onnx_path = model_path.replace('.pth', '.onnx')
            if os.path.exists(onnx_path) and self._try_load_onnx(onnx_path):
                self._backend = 'onnx'
                # Try to read val_acc from checkpoint if torch available
                if TORCH_AVAILABLE:
                    try:
                        self._checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                        val_acc = self._checkpoint.get('val_acc', 'N/A')
                    except Exception as e:
                        print(f"[predict] val_acc unavailable from checkpoint: {e}")
            elif TORCH_AVAILABLE:
                self._checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                val_acc = self._checkpoint.get('val_acc', 'N/A')
                self._load_pytorch_model()
            else:
                raise ImportError(
                    f"No ONNX model found at {onnx_path} and PyTorch is not installed. "
                    "Install onnxruntime ('pip install onnxruntime') and ensure best_model.onnx exists, "
                    "or install PyTorch."
                )
        elif backend == 'onnx':
            onnx_path = model_path.replace('.pth', '.onnx')
            if not os.path.exists(onnx_path):
                if not TORCH_AVAILABLE:
                    raise ImportError("Cannot export ONNX model without PyTorch")
                self._checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                self._load_pytorch_model()
                self.export_onnx(onnx_path)
            self._try_load_onnx(onnx_path)
            self._backend = 'onnx'
        elif backend == 'quantized':
            if not TORCH_AVAILABLE:
                raise ImportError("Quantized backend requires PyTorch")
            self._checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self._load_pytorch_model()
            self._quantize_model()
            self._backend = 'quantized'
        else:
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch backend requires PyTorch")
            self._checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self._load_pytorch_model()
        return val_acc

    def _load_pytorch_model(self):
        """Load the PyTorch model from checkpoint."""
        self.model = TibetanCNN(
            num_classes=self.num_classes,
            dropout=self._checkpoint.get('dropout', 0.3),
        )
        self.model.load_state_dict(self._checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def _try_load_onnx(self, onnx_path):
        """Try to load ONNX Runtime session. Returns True on success."""
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = os.cpu_count() or 4
            self._onnx_session = ort.InferenceSession(
                onnx_path, opts, providers=['CPUExecutionProvider']
            )
            self._onnx_input_name = self._onnx_session.get_inputs()[0].name
            return True
        except (ImportError, Exception) as e:
            print(f"  ONNX Runtime not available: {e}")
            self._onnx_session = None
            return False

    def _quantize_model(self):
        """Apply dynamic INT8 quantization to the model for faster CPU inference."""
        self.model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8,
        )

    @property
    def classes_(self):
        """Original Namsel label IDs, ordered by CNN output index.
        Compatible with sklearn's classifier.classes_ attribute.
        """
        return self._classes

    def _prepare_input(self, x):
        """Convert various input formats to (N, 1, 32, 32) numpy array.

        Accepts single or batch inputs:
            - (32, 32) ndarray: single 32x32 image
            - (1024,) ndarray: flattened 32x32 image
            - (1, 1024) ndarray: sklearn-style row vector
            - (1, 32, 32) ndarray: single image with batch dim
            - (1, 1, 32, 32) ndarray: ready to use
            - (N, 32, 32) ndarray: batch of images
            - (N, 1024) ndarray: batch of flattened images
            - (N, 1, 32, 32) ndarray: batch ready to use
            - list of arrays: batch of images
        """
        # Handle list input
        if isinstance(x, list):
            return self._prepare_batch(x)

        if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
            x = x.cpu().numpy()

        x = np.asarray(x, dtype=np.float32)

        # Single image cases
        if x.shape in ((32, 32), (1024,), (1, 1024), (1, 32, 32)):
            x = x.reshape((1, 1, 32, 32))
        elif x.shape == (1, 1, 32, 32):
            pass
        # Batch cases
        elif x.ndim == 3 and x.shape[1:] == (32, 32):
            x = x.reshape((x.shape[0], 1, 32, 32))
        elif x.ndim == 2 and x.shape[1] == 1024:
            x = x.reshape((x.shape[0], 1, 32, 32))
        elif x.ndim == 4 and x.shape[1:] == (1, 32, 32):
            pass
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        # Normalize to [0, 1] if needed
        if x.max() > 1.0:
            x = x / 255.0

        return x

    def _prepare_batch(self, images):
        """Convert a list of images to (N, 1, 32, 32) numpy array.

        Args:
            images: list of (32, 32) or (1024,) arrays

        Returns:
            ndarray of shape (N, 1, 32, 32)
        """
        # Fast path: stack as numpy array and reshape in one go
        arr = np.array(images, dtype=np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        n = arr.shape[0]
        if arr.ndim == 2 and arr.shape[1] == 1024:
            return arr.reshape((n, 1, 32, 32))
        elif arr.ndim == 3 and arr.shape[1:] == (32, 32):
            return arr.reshape((n, 1, 32, 32))
        # Fallback for mixed shapes
        batch = np.empty((n, 1, 32, 32), dtype=np.float32)
        for i, img in enumerate(images):
            batch[i] = self._prepare_input(img)
        return batch

    def _forward_numpy(self, x_np):
        """Run forward pass, dispatch to best available backend.

        Args:
            x_np: numpy array of shape (N, 1, 32, 32)

        Returns:
            logits as numpy array (N, num_classes)
        """
        if self._onnx_session is not None:
            outputs = self._onnx_session.run(
                None, {self._onnx_input_name: x_np}
            )
            return outputs[0]
        elif TORCH_AVAILABLE:
            tensor = torch.from_numpy(x_np).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
            return logits.cpu().numpy()
        else:
            raise RuntimeError("No inference backend available (need ONNX Runtime or PyTorch)")

    def predict_log_proba(self, x):
        """Predict log probabilities for character image(s).

        Drop-in replacement for sklearn's cls.predict_log_proba().
        Handles both single images and batches.

        Args:
            x: single image or batch in any supported format (see _prepare_input)

        Returns:
            ndarray (N, num_classes) of log probabilities,
            columns ordered by self.classes_
        """
        x_np = self._prepare_input(x)
        logits = self._forward_numpy(x_np)
        # Stable log-softmax
        max_logit = logits.max(axis=1, keepdims=True)
        shifted = logits - max_logit
        log_sum_exp = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        return shifted - log_sum_exp

    def predict_log_proba_batch(self, images):
        """Predict log probabilities for multiple character images at once.

        Much faster than calling predict_log_proba() in a loop because
        it batches the forward pass.

        Args:
            images: list of character images (32x32 arrays)

        Returns:
            ndarray (N, num_classes) of log probabilities
        """
        if len(images) == 0:
            return np.empty((0, self.num_classes))
        batch = self._prepare_batch(images)
        logits = self._forward_numpy(batch)
        # Stable log-softmax
        max_logit = logits.max(axis=1, keepdims=True)
        shifted = logits - max_logit
        log_sum_exp = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        return shifted - log_sum_exp

    def predict_proba(self, x):
        """Predict probabilities for character image(s).

        Args:
            x: single image or batch in any supported format

        Returns:
            ndarray (N, num_classes) of probabilities
        """
        log_probs = self.predict_log_proba(x)
        return np.exp(log_probs)

    def predict(self, x, label_chars=None):
        """Predict the most likely character for a 32x32 image.

        Args:
            x: character image in any supported format
            label_chars: optional dict mapping Namsel label IDs to characters

        Returns:
            (label_or_char, probability) tuple
        """
        probs = self.predict_proba(x)
        idx = np.argmax(probs[0])
        prob = probs[0][idx]
        original_label = self.idx_to_label[idx]

        if label_chars is not None:
            char = label_chars.get(original_label, f"?{original_label}")
            return char, prob
        return original_label, prob

    def predict_top_k(self, x, k=5, label_chars=None):
        """Return top-k predictions with probabilities.

        Args:
            x: character image
            k: number of top predictions to return
            label_chars: optional label->character mapping

        Returns:
            list of (label_or_char, probability) tuples, sorted by probability
        """
        probs = self.predict_proba(x)[0]
        top_indices = np.argsort(probs)[::-1][:k]

        results = []
        for idx in top_indices:
            prob = probs[idx]
            original_label = self.idx_to_label[idx]
            if label_chars is not None:
                char = label_chars.get(original_label, f"?{original_label}")
                results.append((char, prob))
            else:
                results.append((original_label, prob))
        return results

    def export_onnx(self, output_path=None):
        """Export model to ONNX format for fast CPU inference.

        ONNX Runtime is typically 2-3x faster than PyTorch on CPU.

        Args:
            output_path: path for .onnx file (default: same dir as .pth)

        Returns:
            path to exported ONNX file
        """
        if output_path is None:
            output_path = self._model_path.replace('.pth', '.onnx')

        # Ensure we have a PyTorch model loaded
        if not hasattr(self, 'model'):
            self._load_pytorch_model()

        dummy = torch.randn(1, 1, 32, 32, device=self.device)
        export_kwargs = dict(
            input_names=['image'],
            output_names=['logits'],
            dynamic_axes={
                'image': {0: 'batch_size'},
                'logits': {0: 'batch_size'},
            },
            opset_version=18,
        )
        try:
            torch.onnx.export(self.model, dummy, output_path, dynamo=False, **export_kwargs)
        except TypeError:
            torch.onnx.export(self.model, dummy, output_path, **export_kwargs)
        print(f"Exported ONNX model to {output_path}")

        # Auto-load the ONNX session
        if self._try_load_onnx(output_path):
            self._backend = 'onnx'
            print("  Switched to ONNX Runtime backend")

        return output_path

    def export_quantized_onnx(self, output_path=None):
        """Export INT8 quantized ONNX model for maximum CPU speed.

        Requires onnxruntime and onnx packages.

        Args:
            output_path: path for quantized .onnx file

        Returns:
            path to exported quantized ONNX file
        """
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
        except ImportError:
            print("Install onnxruntime for ONNX quantization: pip install onnxruntime")
            return None

        if output_path is None:
            output_path = self._model_path.replace('.pth', '_int8.onnx')

        # Reuse existing FP32 ONNX if available, otherwise create a temp one
        main_onnx = self._model_path.replace('.pth', '.onnx')
        if os.path.exists(main_onnx):
            fp32_path = main_onnx
            created_temp = False
        else:
            fp32_path = self._model_path.replace('.pth', '_fp32_tmp.onnx')
            self.export_onnx(fp32_path)
            created_temp = True

        quantize_dynamic(
            fp32_path,
            output_path,
            weight_type=QuantType.QUInt8,
        )
        print(f"Exported INT8 quantized ONNX to {output_path}")

        # Auto-load
        if self._try_load_onnx(output_path):
            self._backend = 'onnx'
            print("  Switched to quantized ONNX Runtime backend")

        # Clean up temp FP32 file only
        if created_temp and os.path.exists(fp32_path):
            os.remove(fp32_path)

        return output_path
