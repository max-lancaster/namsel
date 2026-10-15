#!/usr/bin/env python3
"""
Scale-Invariant Preprocessing for Namsel OCR

Integrated preprocessing module that automatically handles scale-dependent
OCR issues by applying conservative upscaling and denoising when needed.
"""

import cv2 as cv
import numpy as np
import logging

logger = logging.getLogger(__name__)

def estimate_character_size(img: np.ndarray) -> float:
    """
    Estimate typical character size in the image
    """
    # Convert to uint8 if needed
    if img.dtype == np.float64:
        img_uint8 = (img * 255).astype(np.uint8)
    else:
        img_uint8 = img.astype(np.uint8)
    
    # Simple thresholding to find text
    _, binary = cv.threshold(img_uint8, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 16.0  # Default character size
        
    # Get widths and heights of reasonable contours
    widths = []
    heights = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        # Filter out very small noise and very large regions
        if 3 <= w <= img.shape[1] * 0.3 and 3 <= h <= img.shape[0] * 0.3:
            widths.append(w)
            heights.append(h)
    
    if not widths:
        return 16.0
        
    # Use median character size as estimate
    median_width = np.median(widths)
    median_height = np.median(heights)
    
    # Take the larger dimension as character size estimate
    estimated_size = max(median_width, median_height)
    
    return estimated_size

def apply_scale_invariant_preprocessing(img_array: np.ndarray, min_char_size: int = 12) -> np.ndarray:
    """
    Apply conservative scale-invariant preprocessing
    
    Only applies minimal processing to avoid breaking OCR pipeline.
    Focus on fixing scale issues for very small images only.
    
    Args:
        img_array: Input image array in [0,1] range from PIL
        min_char_size: Minimum character size requiring upscaling
        
    Returns:
        Preprocessed image array in [0,1] range
    """
    logger.info("Applying conservative scale-invariant preprocessing")
    
    # Convert to uint8 for processing
    if img_array.max() <= 1.0:
        img_uint8 = (img_array * 255).astype(np.uint8)
    else:
        img_uint8 = img_array.astype(np.uint8)
    
    # Estimate character size
    char_size = estimate_character_size(img_uint8)
    logger.info(f"Estimated character size: {char_size:.1f}px")
    
    # FIXED: Only upscale very small images to avoid contour artifacts
    # Relaxed thresholds to avoid over-processing that breaks segmentation
    image_is_very_small = img_uint8.shape[0] < 100 or img_uint8.shape[1] < 200
    chars_are_very_small = char_size < 10
    
    if chars_are_very_small or image_is_very_small:
        # Use more conservative scaling and avoid artifacts
        scale_factor = max(1.2, min_char_size / char_size)
        scale_factor = min(scale_factor, 1.5)  # More conservative cap at 1.5x
        
        new_width = int(img_uint8.shape[1] * scale_factor)
        new_height = int(img_uint8.shape[0] * scale_factor)
        
        # Use INTER_LINEAR to avoid artifacts that break character boundaries
        img_scaled = cv.resize(img_uint8, (new_width, new_height), interpolation=cv.INTER_LINEAR)
        logger.info(f"Conservative upscale by {scale_factor:.2f}x: {img_uint8.shape} -> {img_scaled.shape}")
    else:
        img_scaled = img_uint8
        logger.info("Character size adequate, no scaling needed")
    
    # DISABLED: Bilateral filtering can blur character boundaries and affect contour detection
    # img_denoised = cv.bilateralFilter(img_scaled, 5, 50, 50)
    img_denoised = img_scaled  # Skip filtering to preserve character boundaries
    logger.info("Skipped filtering to preserve character boundaries")
    
    # Convert back to [0,1] range
    result = img_denoised.astype(np.float64) / 255.0
    
    logger.info(f"Conservative preprocessing complete: {result.shape}, range: {result.min():.3f}-{result.max():.3f}")
    return result

# Enlargement of a pre-segmented line (force_single_line) before it is read. Fixed: the page rule below decides
# from the IMAGE's height (< 100 px) and from components under 30% of it, so the paper a line finder leaves
# above or below a line changed whether and how much it was enlarged (1.0x to 1.5x for lines of one font and
# size). 1.2x was measured against 1.35x and a target-width rule; it read best on rendered pages, the user's
# print and captures alike.
BAND_SCALE = 1.2


def apply_band_preprocessing(img_array: np.ndarray) -> np.ndarray:
    """A pre-segmented line, enlarged by BAND_SCALE; returned in the [0, 1] range like the page path."""
    img_uint8 = (img_array * 255).astype(np.uint8) if img_array.max() <= 1.0 else img_array.astype(np.uint8)
    height, width = img_uint8.shape[:2]
    img_uint8 = cv.resize(img_uint8, (int(width * BAND_SCALE), int(height * BAND_SCALE)),
                          interpolation=cv.INTER_LINEAR)
    return img_uint8.astype(np.float64) / 255.0


def should_apply_preprocessing(img_array: np.ndarray) -> bool:
    """
    Determine if preprocessing should be applied based on image characteristics
    """
    # Estimate character size
    char_size = estimate_character_size(img_array)
    
    # Apply preprocessing if characters are small or image has quality issues
    should_preprocess = (
        char_size < 12 or  # Small characters
        img_array.shape[0] < 150 or  # Small image height
        img_array.shape[1] < 300    # Small image width
    )
    
    return should_preprocess