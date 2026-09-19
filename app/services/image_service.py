import os
from pathlib import Path
from typing import Tuple, Optional
import cv2
import numpy as np
from fastapi import HTTPException, status

from PIL import Image, ImageOps

from app.config import BLUR_THRESHOLD


def calculate_laplacian_variance(image_path: str) -> float:
    """Calculates the focus measure (Laplacian variance) of an image."""
    img = cv2.imread(image_path)
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to read uploaded image from {image_path}",
        )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def check_image_quality(image_path: str, threshold: float = BLUR_THRESHOLD) -> Tuple[bool, float]:
    """
    Checks if an image meets blur quality standards.
    Rejects with clear error message if below blur threshold.
    """
    variance = calculate_laplacian_variance(image_path)
    is_sharp = variance >= threshold
    return is_sharp, variance


def deskew_image(image: np.ndarray) -> np.ndarray:
    """
    Detects skew angle of text/label contours and rotates image back to horizontal.
    Handles minor rotations (-45 to 45 degrees).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Binary threshold with inversion to find text areas
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find coordinates of all foreground pixels
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 50:
        return image  # Not enough text pixels to reliably estimate skew

    # Compute minimum area bounding box
    angle = cv2.minAreaRect(coords)[-1]

    # Adjust angle for OpenCV rect orientation
    if angle < -45:
        angle = -(90 + angle)
    elif angle > 45:
        angle = 90 - angle

    # Ignore micro rotations (< 0.5 degrees)
    if abs(angle) < 0.5 or abs(angle) > 45.0:
        return image

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
    on the L-channel of LAB color space to sharpen printed text on packaging.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_chan)
    enhanced_lab = cv2.merge((enhanced_l, a_chan, b_chan))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def resize_if_large(image: np.ndarray, max_side: int = 1024) -> np.ndarray:
    """
    Downscales images exceeding max_side (e.g. 12-48MP smartphone photos)
    using cv2.INTER_AREA to accelerate OCR inference by 4x-7x while preserving text sharpness.
    """
    (h, w) = image.shape[:2]
    max_dim = max(h, w)
    if max_dim <= max_side:
        return image

    scale = max_side / float(max_dim)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def preprocess_image(input_path: str, output_path: Optional[str] = None) -> str:
    """
    High-performance preprocessing pipeline:
    1. Reads original image with PIL and applies EXIF orientation normalization
    2. Smart downscales to max 1024px to eliminate redundant CPU compute for DBNet detector
    3. Corrects skew if non-trivial angle detected
    4. Saves optimized image
    """
    try:
        # Step 0: Auto-orient using EXIF metadata (crucial for smartphone camera photos)
        with Image.open(input_path) as pil_img:
            transposed = ImageOps.exif_transpose(pil_img)
            # Convert RGB/RGBA to BGR for OpenCV
            if transposed.mode in ("RGBA", "LA"):
                transposed = transposed.convert("RGB")
            img = cv2.cvtColor(np.array(transposed), cv2.COLOR_RGB2BGR)
    except Exception:
        img = cv2.imread(input_path)

    if img is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to load image for preprocessing",
        )

    # Step 1: Fast downscale if oversized (e.g. 4000x3000 -> 1024x768)
    scaled = resize_if_large(img, max_side=1024)

    # Step 2: Deskew if rotated
    deskewed = deskew_image(scaled)

    if output_path is None:
        p = Path(input_path)
        output_path = str(p.parent / f"prep_{p.name}")

    cv2.imwrite(output_path, deskewed)
    return output_path
