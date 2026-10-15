"""Dataset loading and augmentation for Tibetan character training data.

Loads existing Namsel training data (font-drawn + manually labeled samples)
and provides PyTorch Dataset with CPU-friendly augmentation.

Training data format: each sample is (label, pixel_0, pixel_1, ..., pixel_1023)
where pixels form a 32x32 grayscale character image.
"""

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import rotate, shift, zoom, gaussian_filter, map_coordinates


def _load_csv_dataset(data_dir, filename, label, reshape_1d=False):
    """Loads a comma-delimited uint32 dataset by name, or None when the file is absent."""
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        return None
    data = np.genfromtxt(path, np.uint32, delimiter=',')
    if reshape_1d and data.ndim == 1:
        data = data.reshape(1, -1)
    print(f"  {label}: {data.shape[0]} samples")
    return data


def load_all_training_data(data_dir):
    """Load all Namsel training datasets from the datasets directory.

    Args:
        data_dir: path to namsel_ocr/datasets/

    Returns:
        images: ndarray (N, 32, 32) uint8
        labels: ndarray (N,) int
    """
    all_data = []

    # 1-3. CSV-style datasets (comma-delimited uint32).
    for filename, label in (
        ('font-draw-samples.txt', 'font-draw-samples'),  # primary training data
        ('tibcharsamples.txt', 'tibcharsamples'),
        ('ui_samples.csv', 'ui_samples'),               # manually labeled via UI
    ):
        data = _load_csv_dataset(data_dir, filename, label)
        if data is not None:
            all_data.append(data)

    # 4. normalized_3216_to_3232_training.npy
    norm_npy = os.path.join(data_dir, 'normalized_3216_to_3232_training.npy')
    if os.path.exists(norm_npy):
        data = np.load(norm_npy)
        all_data.append(data)
        print(f"  normalized_3216_to_3232: {data.shape[0]} samples")

    # 5. symbols.txt (may be a single 1-D row)
    data = _load_csv_dataset(data_dir, 'symbols.txt', 'symbols', reshape_1d=True)
    if data is not None:
        all_data.append(data)

    # 6. Character-specific additions, stored as .npy (data-only; migrated off
    #    pickle via convert_datasets_to_npy.py). np.load(allow_pickle=False)
    #    cannot execute code, unlike pickle.load.
    already_loaded = {'normalized_3216_to_3232_training.npy'}  # loaded by name above
    extra_count = 0
    for npy_file in sorted(glob.glob(os.path.join(data_dir, '*.npy'))):
        if os.path.basename(npy_file) in already_loaded:
            continue
        try:
            data = np.array(np.load(npy_file, allow_pickle=False))
            if data.ndim == 2 and data.shape[1] == 1025:
                all_data.append(data)
                extra_count += data.shape[0]
        except Exception as e:
            print(f"  skipping unreadable npy {npy_file}: {e}")
    if extra_count > 0:
        print(f"  npy files: {extra_count} samples")

    if not all_data:
        raise FileNotFoundError(f"No training data found in {data_dir}")

    # Combine all datasets
    combined = np.concatenate(all_data, axis=0)

    # Deduplicate
    combined = np.unique(combined, axis=0)
    print(f"  Total after dedup: {combined.shape[0]} samples")

    labels = combined[:, 0].astype(np.int64)
    images = combined[:, 1:].astype(np.uint8).reshape(-1, 32, 32)

    return images, labels


def build_label_mapping(labels):
    """Create contiguous 0-indexed label mapping for PyTorch.

    Args:
        labels: array of original Namsel label IDs (non-contiguous integers)

    Returns:
        label_to_idx: dict mapping original_label -> contiguous_index
        idx_to_label: dict mapping contiguous_index -> original_label
    """
    unique_labels = sorted(set(labels.tolist()))
    label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    idx_to_label = {idx: lbl for idx, lbl in enumerate(unique_labels)}
    return label_to_idx, idx_to_label


def compute_class_weights(labels, label_to_idx, method='inverse_sqrt'):
    """Compute per-class weights for imbalanced training data.

    Args:
        labels: array of original Namsel label IDs
        label_to_idx: mapping from original label to contiguous index
        method: 'inverse' (1/count), 'inverse_sqrt' (1/sqrt(count)),
                or 'effective' (effective number of samples)

    Returns:
        torch.FloatTensor of shape (num_classes,) with per-class weights,
        normalized so mean weight = 1.0
    """
    num_classes = len(label_to_idx)
    counts = np.zeros(num_classes, dtype=np.float64)

    for lbl in labels:
        idx = label_to_idx[int(lbl)]
        counts[idx] += 1

    # Avoid division by zero for empty classes
    counts = np.maximum(counts, 1.0)

    if method == 'inverse':
        weights = 1.0 / counts
    elif method == 'inverse_sqrt':
        weights = 1.0 / np.sqrt(counts)
    elif method == 'effective':
        # Effective number of samples (Cui et al. 2019)
        beta = 0.999
        weights = (1.0 - beta) / (1.0 - beta ** counts)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize so mean weight = 1.0
    weights = weights / weights.mean()

    return torch.FloatTensor(weights)


# --- Augmentation functions (numpy/scipy, no torchvision dependency) ---

def random_rotation(img, max_angle=5.0):
    """Rotate image by a random angle within [-max_angle, max_angle]."""
    angle = np.random.uniform(-max_angle, max_angle)
    # Use order=1 (bilinear) and fill with background (1.0 for white bg)
    rotated = rotate(img, angle, reshape=False, order=1, cval=1.0)
    return rotated


def random_shift(img, max_pixels=2):
    """Translate image by random offset in x and y."""
    dx = np.random.uniform(-max_pixels, max_pixels)
    dy = np.random.uniform(-max_pixels, max_pixels)
    shifted = shift(img, [dy, dx], order=1, cval=1.0)
    return shifted


def random_scale(img, scale_range=(0.9, 1.1)):
    """Scale image by a random factor, keeping 32x32 output."""
    factor = np.random.uniform(*scale_range)
    h, w = img.shape
    scaled = zoom(img, factor, order=1, cval=1.0)
    sh, sw = scaled.shape

    # Crop or pad to original size
    result = np.ones((h, w), dtype=img.dtype)
    y_off = (sh - h) // 2
    x_off = (sw - w) // 2

    if factor >= 1.0:
        # Crop center
        result = scaled[y_off:y_off + h, x_off:x_off + w]
    else:
        # Pad with background
        result = np.ones((h, w), dtype=img.dtype)
        py = (h - sh) // 2
        px = (w - sw) // 2
        result[py:py + sh, px:px + sw] = scaled

    return result[:h, :w]


def elastic_deformation(img, alpha=3.0, sigma=0.5):
    """Apply elastic deformation to simulate print artifacts.

    alpha: deformation intensity
    sigma: smoothness of deformation field
    """
    shape = img.shape
    dx = gaussian_filter(np.random.randn(*shape), sigma) * alpha
    dy = gaussian_filter(np.random.randn(*shape), sigma) * alpha

    y, x = np.mgrid[0:shape[0], 0:shape[1]]
    indices = [np.clip(y + dy, 0, shape[0] - 1),
               np.clip(x + dx, 0, shape[1] - 1)]

    return map_coordinates(img, indices, order=1, cval=1.0)


def morphological_noise(img, prob=0.3):
    """Randomly erode or dilate to simulate ink thickness variation."""
    if np.random.random() > prob:
        return img

    from scipy.ndimage import binary_erosion, binary_dilation

    # Work with binary mask (foreground = values < 0.5)
    binary = img < 0.5
    kernel = np.ones((2, 2), dtype=bool)

    if np.random.random() > 0.5:
        result = binary_dilation(binary, structure=kernel)
    else:
        result = binary_erosion(binary, structure=kernel)

    # Convert back to float image
    out = np.ones_like(img)
    out[result] = 0.0
    return out


def gaussian_noise(img, sigma=0.03):
    """Add small Gaussian noise."""
    noise = np.random.normal(0, sigma, img.shape)
    noisy = np.clip(img + noise, 0.0, 1.0)
    return noisy


def augment_image(img):
    """Apply random augmentations to a 32x32 character image.

    Args:
        img: ndarray (32, 32) float in [0, 1] range

    Returns:
        Augmented image, same shape and range
    """
    # Each augmentation applied with some probability
    if np.random.random() > 0.3:
        img = random_rotation(img, max_angle=5.0)

    if np.random.random() > 0.4:
        img = random_shift(img, max_pixels=2)

    if np.random.random() > 0.5:
        img = random_scale(img, scale_range=(0.9, 1.1))

    if np.random.random() > 0.5:
        img = elastic_deformation(img, alpha=2.0, sigma=0.4)

    img = morphological_noise(img, prob=0.2)

    if np.random.random() > 0.6:
        img = gaussian_noise(img, sigma=0.02)

    return np.clip(img, 0.0, 1.0).astype(np.float32)


class TibetanCharDataset(Dataset):
    """PyTorch Dataset for Tibetan character images.

    Loads 32x32 character images with their labels, applies
    optional augmentation during training.
    """

    def __init__(self, images, labels, label_to_idx, augment=False):
        """
        Args:
            images: ndarray (N, 32, 32) uint8
            labels: ndarray (N,) original Namsel label IDs
            label_to_idx: dict mapping original label -> contiguous index
            augment: whether to apply random augmentation
        """
        self.images = images
        self.labels = labels
        self.label_to_idx = label_to_idx
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32)

        # Normalize to [0, 1]
        if img.max() > 1.0:
            img = img / 255.0

        if self.augment:
            img = augment_image(img)

        # Convert to tensor: (1, 32, 32) with channel dim
        tensor = torch.from_numpy(img).unsqueeze(0)

        # Map original label to contiguous index
        label = self.label_to_idx[int(self.labels[idx])]

        return tensor, label


def create_data_loaders(data_dir, batch_size=64, val_split=0.15,
                        augment=True, num_workers=0, pin_memory=False,
                        seed=42):
    """Create train and validation DataLoaders from Namsel training data.

    Args:
        data_dir: path to namsel_ocr/datasets/
        batch_size: mini-batch size (64 recommended for CPU)
        val_split: fraction of data for validation
        augment: apply augmentation to training set
        num_workers: dataloader workers (0 for Windows, 2+ for Linux/Colab)
        pin_memory: True for GPU training (faster CPU->GPU transfer)
        seed: random seed for reproducible split

    Returns:
        train_loader, val_loader, label_to_idx, idx_to_label, num_classes,
        class_weights (FloatTensor)
    """
    print("Loading training data...")
    images, labels = load_all_training_data(data_dir)

    label_to_idx, idx_to_label = build_label_mapping(labels)
    num_classes = len(label_to_idx)
    print(f"  Classes: {num_classes}")

    # Compute class weights for imbalanced data
    class_weights = compute_class_weights(labels, label_to_idx)
    min_w, max_w = class_weights.min().item(), class_weights.max().item()
    print(f"  Class weight range: {min_w:.2f} - {max_w:.2f}")

    # Stratified-ish split: shuffle then split
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(labels))
    val_size = int(len(labels) * val_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_dataset = TibetanCharDataset(
        images[train_indices], labels[train_indices],
        label_to_idx, augment=augment
    )
    val_dataset = TibetanCharDataset(
        images[val_indices], labels[val_indices],
        label_to_idx, augment=False
    )

    persistent = num_workers > 0
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent,
    )

    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    return train_loader, val_loader, label_to_idx, idx_to_label, num_classes, class_weights
