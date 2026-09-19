import logging
import warnings
from typing import List, Dict, Any, Optional
import numpy as np
import cv2

# Suppress harmless PaddlePaddle C++ compiler cache warning on Windows
warnings.filterwarnings("ignore", category=UserWarning, module="paddle")
warnings.filterwarnings("ignore", message=".*ccache.*")

logger = logging.getLogger(__name__)

# Apply compatibility patch for PaddlePaddle 3.x + PaddleX on Windows CPU
try:
    from paddlex.inference.models.runners.paddle_static.runner import PaddleStaticRunner

    orig_create = PaddleStaticRunner._create

    def _patched_runner_create(self):
        # Disable PIR new IR executor on CPU to prevent onednn ConvertPirAttribute2RuntimeAttribute issue
        if hasattr(self, "_config"):
            self._config["enable_new_ir"] = False
            self._config["run_mode"] = "paddle"
        return orig_create(self)

    PaddleStaticRunner._create = _patched_runner_create
    logger.info("Successfully configured PaddleStaticRunner compatibility patch for Windows CPU.")
except Exception as e:
    logger.warning("Could not patch PaddleStaticRunner (might already be compatible): %s", e)

# Global lazy-loaded OCR instance
_ocr_instance = None
_ocr_available = True


def get_ocr_engine():
    """Lazily initializes and returns high-speed singleton PaddleOCR instance for fast CPU inference."""
    global _ocr_instance, _ocr_available
    if not _ocr_available:
        return None
    if _ocr_instance is None:
        try:
            from paddleocr import PaddleOCR

            logger.info("Initializing high-speed PaddleOCR engine (PP-OCRv4 mobile CPU models)...")
            _ocr_instance = PaddleOCR(
                ocr_version="PP-OCRv4",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_recognition_batch_size=8,
                lang="en",
            )
            logger.info("PaddleOCR PP-OCRv4 engine initialized successfully.")
        except Exception as e:
            logger.warning("PaddleOCR not available (%s). System will use cloud/Gemini Vision OCR fallback.", e)
            _ocr_available = False
            _ocr_instance = None
    return _ocr_instance


def warmup_ocr():
    """Pre-loads PaddleOCR model weights into memory at startup to eliminate first-request cold-start latency."""
    try:
        engine = get_ocr_engine()
        if engine is not None:
            # Warmup with tiny blank image
            dummy = np.full((32, 128, 3), 255, dtype=np.uint8)
            engine.predict(dummy)
            logger.info("PaddleOCR warmup completed successfully.")
    except Exception as e:
        logger.warning("PaddleOCR warmup encountered exception: %s", e)


class OCRBlock:
    def __init__(self, text: str, confidence: float, box: Any, height: float, image_index: int = 0):
        self.text = text.strip()
        self.confidence = float(confidence)
        if isinstance(box, list) and len(box) > 0 and isinstance(box[0], (list, tuple)):
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            self.box = {
                "polygon": box,
                "x_min": min(xs),
                "y_min": min(ys),
                "x_max": max(xs),
                "y_max": max(ys),
            }
        else:
            self.box = box
        self.height = float(height)
        self.image_index = int(image_index)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "bounding_box": self.box,
            "height": round(self.height, 2),
            "image_index": self.image_index,
        }


def run_fallback_ocr(image_path: str, image_index: int = 0) -> Dict[str, Any]:
    """
    Cloud/Gemini fallback OCR when PaddleOCR is not installed (e.g. Vercel serverless deployment).
    Uses Gemini Vision to read text lines and construct OCR blocks.
    """
    from app.services.gemini_service import is_ai_available, generate_content_resilient
    from google.genai import types
    from PIL import Image

    blocks: List[OCRBlock] = []
    full_text = ""

    if is_ai_available():
        try:
            pil_img = Image.open(image_path)
            prompt = (
                "You are an accurate OCR engine for Indian Legal Metrology compliance verification. "
                "Read and transcribe EVERY line of text visible on this packaging label image exactly as printed. "
                "Output each line on a new line. Do not summarize, alter words, or add markdown code blocks."
            )
            response = generate_content_resilient(
                contents=[pil_img, prompt],
                config=types.GenerateContentConfig(temperature=0.0)
            )
            if response and response.text:
                full_text = response.text.strip()
                lines = [line.strip() for line in full_text.splitlines() if line.strip()]
                for idx, line in enumerate(lines):
                    blocks.append(
                        OCRBlock(
                            text=line,
                            confidence=0.92,
                            box=[[0, idx * 25], [200, idx * 25], [200, idx * 25 + 20], [0, idx * 25 + 20]],
                            height=20.0,
                            image_index=image_index
                        )
                    )
        except Exception as e:
            logger.error("Fallback Gemini OCR encountered error: %s", e)

    avg_conf = (sum(b.confidence for b in blocks) / len(blocks)) if blocks else 0.0
    return {
        "blocks": [b.to_dict() for b in blocks],
        "full_text": full_text,
        "average_confidence": round(avg_conf, 4),
        "median_height": 20.0,
    }


def run_ocr(image_path: str, image_index: int = 0) -> Dict[str, Any]:
    """
    Runs high-speed PaddleOCR inference on the provided preprocessed image.
    Falls back gracefully to Gemini Vision OCR when PaddleOCR is not installed (e.g. on Vercel).
    """
    ocr = get_ocr_engine()
    if ocr is None:
        return run_fallback_ocr(image_path, image_index)

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at path: {image_path}")

    # High-speed prediction with batching
    predictions = ocr.predict(img)
    blocks: List[OCRBlock] = []

    for result in predictions:
        # PaddleOCR 3.7 returns dict-like objects containing rec_texts, rec_scores, rec_polys or rec_boxes
        if isinstance(result, dict):
            rec_texts = result.get("rec_texts", [])
            rec_scores = result.get("rec_scores", [])
            rec_polys = result.get("rec_polys", [])

            for idx, text in enumerate(rec_texts):
                if not text or not text.strip():
                    continue
                score = float(rec_scores[idx]) if idx < len(rec_scores) else 0.8
                poly = rec_polys[idx] if idx < len(rec_polys) else None

                # Convert poly to list of [x, y] coordinates and calculate bounding box height
                poly_list = []
                height = 16.0
                if poly is not None:
                    if isinstance(poly, np.ndarray):
                        poly_list = poly.tolist()
                    else:
                        poly_list = list(poly)

                    # Compute height from poly points: top edge to bottom edge
                    try:
                        y_coords = [p[1] for p in poly_list]
                        height = max(y_coords) - min(y_coords)
                    except Exception:
                        height = 16.0

                blocks.append(
                    OCRBlock(text=text, confidence=score, box=poly_list, height=height, image_index=image_index)
                )

        elif isinstance(result, list):
            # Traditional paddleocr nested list format: [[[points], (text, conf)], ...]
            for line in result:
                if not line or len(line) < 2:
                    continue
                poly = line[0]
                text_info = line[1]
                if isinstance(text_info, (tuple, list)):
                    text = text_info[0]
                    score = float(text_info[1])
                else:
                    text = str(text_info)
                    score = 0.8

                if not text or not text.strip():
                    continue

                poly_list = poly if isinstance(poly, list) else poly.tolist()
                try:
                    y_coords = [p[1] for p in poly_list]
                    height = max(y_coords) - min(y_coords)
                except Exception:
                    height = 16.0

                blocks.append(
                    OCRBlock(text=text, confidence=score, box=poly_list, height=height, image_index=image_index)
                )

    # Calculate aggregate metrics
    if blocks:
        avg_conf = sum(b.confidence for b in blocks) / len(blocks)
        heights = [b.height for b in blocks if b.height > 0]
        median_h = float(np.median(heights)) if heights else 16.0
        full_text = "\n".join(b.text for b in blocks)
    else:
        avg_conf = 0.0
        median_h = 16.0
        full_text = ""

    return {
        "blocks": [b.to_dict() for b in blocks],
        "full_text": full_text,
        "average_confidence": round(avg_conf, 4),
        "median_height": round(median_h, 2),
    }
