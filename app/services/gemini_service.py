import os
import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from google import genai
from google.genai import types
from google.genai.errors import APIError

from app.config import GEMINI_API_KEY, AI_MATCH_LOG_FILE

logger = logging.getLogger(__name__)

_client = None
if GEMINI_API_KEY:
    try:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize Gemini Client: {e}")

SUPPORTED_MODELS = ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-flash-latest", "gemini-3.5-flash"]

def is_ai_available() -> bool:
    return _client is not None

def generate_content_resilient(contents: Any, config: Optional[types.GenerateContentConfig] = None) -> Optional[Any]:
    if not is_ai_available():
        return None
    last_err = None
    for model_name in SUPPORTED_MODELS:
        try:
            response = _client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            return response
        except APIError as e:
            last_err = e
            err_str = str(e)
            if "404" in err_str or "not found" in err_str.lower() or "no longer available" in err_str.lower():
                logger.warning(f"Model {model_name} unavailable: {e}. Falling back immediately.")
                continue
            if "503" in err_str or "high demand" in err_str.lower() or "429" in err_str:
                logger.warning(f"Model {model_name} busy or rate limited ({e}). Cascading to next model.")
                continue
            logger.error(f"Gemini API error on {model_name}: {e}")
            continue
        except Exception as e:
            last_err = e
            logger.error(f"Gemini unexpected error on {model_name}: {e}")
            continue
    if last_err:
        logger.warning(f"All Gemini models failed. Last error: {last_err}")
    return None

def log_ai_resolved_match(input_snippet: str, field_name: str, confidence: float, reason: str, scan_id: Optional[int]):
    try:
        from datetime import datetime
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "scan_id": scan_id,
            "input_snippet": input_snippet,
            "classified_field": field_name,
            "confidence": confidence,
            "reason": reason,
            "model": "gemini-flash-latest"
        }
        with open(AI_MATCH_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to log AI match: {e}")

def classify_with_gemini(text_snippet: str, candidate_fields: List[str]) -> Optional[Tuple[str, float, str]]:
    if not is_ai_available():
        return None
        
    prompt = f"""
You are a highly accurate data extraction system for Legal Metrology compliance.
You are given a text snippet extracted via OCR from a product label, and a list of expected candidate fields.
Your task is to classify which candidate field this text snippet corresponds to, if any.
If the snippet does not match any candidate field, return "none" for field_name.

Candidate fields: {', '.join(candidate_fields)}
Text snippet: "{text_snippet}"

Return ONLY valid JSON with no markdown block formatting, in this exact schema:
{{
  "field_name": "exact_string_from_candidate_list_or_none",
  "confidence": 0.95,
  "reason": "short_explanation"
}}
"""
    try:
        response = generate_content_resilient(
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            )
        )
        if not response:
            return None
        raw = response.text.strip()
        data = json.loads(raw)
        field = data.get("field_name")
        conf = data.get("confidence", 0.0)
        reason = data.get("reason", "No reason provided")
        
        if field and field in candidate_fields and conf >= 0.7:
            return (field, float(conf), str(reason))
        return None
    except Exception as e:
        logger.debug(f"Gemini classification error: {e}")
        return None

def batch_extract_with_gemini_cot(
    all_snippets: List[str],
    missing_fields: List[str],
    extracted_so_far: Dict[str, str],
    scan_id: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    if not is_ai_available():
        return {}
        
    snippets_formatted = "\n".join(f"[{i:02d}] {s}" for i, s in enumerate(all_snippets) if s.strip())
    already_extracted_formatted = (
        "\n".join(f"- {k.replace('_', ' ').title()}: '{v}'" for k, v in extracted_so_far.items())
        if extracted_so_far
        else "None yet"
    )
    missing_fields_formatted = ", ".join(missing_fields)

    prompt = f"""
You are an expert Indian Legal Metrology regulatory compliance auditor.
We are verifying packaging declarations under the Legal Metrology (Packaged Commodities) Rules, 2011 (Rule 6).

PACKAGING OCR TEXT SNIPPETS (in visual order from label):
\"\"\"
{snippets_formatted}
\"\"\"

ALREADY CONFIRMED DECLARATIONS:
{already_extracted_formatted}

MISSING MANDATORY FIELDS TO DEDUCE:
{missing_fields_formatted}

LEGAL METROLOGY FIELD GUIDELINES:
- generic_name: Common or generic name of the commodity (e.g., 'Ball Pens', 'Ball Point Pen', 'Liquid Ink Pen', 'Biscuits', 'Notebook'). Often mentioned in title, subtitle, or inside net quantity statement like '5 Ball Pens'.
- mrp: Maximum Retail Price (inclusive of all taxes, e.g. 'Rs. 50.00 incl. of all taxes').
- net_quantity: Quantity in standard units (e.g. '5 Ball Pens', '100 g', '50 ml').
- mfg_date: Month & Year of manufacture or packing (e.g., '02/2026', 'Feb 2026').
- expiry_date: Best before period or expiry date if applicable.
- country_of_origin: Country where manufactured/produced (e.g., 'India', 'Japan').
- manufacturer_details: Full legal name and postal address of manufacturer.
- consumer_care: Contact details for consumer complaints under Rule 6(1)(da). specifically look for toll-free numbers (e.g., '1800-xxx-xxxx', 'T-Free Number...'), phone numbers, customer care email addresses (e.g. 'care@...', 'feedback@...'), or customer care executive addresses. Combine phone and email if both appear on the package.

INSTRUCTIONS FOR LOGICAL REASONING (Chain of Thought):
1. Identify the product commodity type from the brand, claims, and packaging text.
2. For each missing field in the list, logically deduce which packaging text represents it. Explain your logical reasoning in 'thinking'.
3. Extract the exact declaration value without omitting address details, toll-free numbers, or email addresses.
4. If a field is genuinely absent, do not include it in resolved_fields.

Return ONLY valid JSON in this exact schema:
{{
  "thinking": "step by step deduction",
  "resolved_fields": {{
    "field_name_from_list": {{
      "extracted_value": "exact_matched_text_from_corpus",
      "confidence": 0.95,
      "logical_rationale": "why this matches"
    }}
  }}
}}
"""
    try:
        response = generate_content_resilient(
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            )
        )
        if not response:
            return {}
        data = json.loads(response.text.strip())
        resolved = data.get("resolved_fields", {})
        thinking = data.get("thinking", "")
        
        valid_results = {}
        for f_name, info in resolved.items():
            if f_name in missing_fields and isinstance(info, dict):
                val = info.get("extracted_value")
                conf = float(info.get("confidence", 0.0))
                if val and conf >= 0.65:
                    reason = info.get("logical_rationale", thinking)
                    valid_results[f_name] = {
                        "extracted_value": str(val).strip(),
                        "raw_snippet": str(val).strip(),
                        "confidence": conf,
                        "logical_rationale": reason,
                    }
                    log_ai_resolved_match(str(val), f_name, conf, reason, scan_id)
        return valid_results
    except Exception as e:
        logger.debug(f"Gemini batch CoT error: {e}")
        return {}

def read_image_region_with_gemini(image_path: str, expected_field: str) -> Optional[str]:
    if not is_ai_available():
        return None
        
    prompt = f"This is a cropped region from a product label that may be rotated, wrinkled, or affected by glare. Read and return the exact text visible for the expected field: '{expected_field}'. If you cannot determine it with reasonable confidence or it is not visible, return exactly 'not visible'."
    
    try:
        from PIL import Image
        img = Image.open(image_path)
        
        response = generate_content_resilient(
            contents=[img, prompt],
            config=types.GenerateContentConfig(
                temperature=0.0
            )
        )
        if not response:
            return None
        text = response.text.strip()
        if text.lower() == "not visible" or "not visible" in text.lower():
            return None
        return text
    except Exception as e:
        logger.debug(f"Gemini vision error: {e}")
        return None


def batch_vision_extract_fields(
    image_path: str,
    missing_fields: List[str],
) -> Dict[str, str]:
    """
    Sends the packaging image to Gemini ONCE and asks it to extract ALL missing
    fields in a single call. This replaces N sequential read_image_region_with_gemini
    calls with 1 call, reducing scan time drastically.

    Returns a dict of {field_name: extracted_text} for successfully found fields.
    """
    if not is_ai_available() or not missing_fields:
        return {}

    fields_list = "\n".join(f"- {f}" for f in missing_fields)
    prompt = f"""You are a Legal Metrology compliance officer reading a product packaging label image.
The image may be slightly blurry, rotated, or have glare — do your best to read the visible text.

Extract the following mandatory declaration fields from the label image:
{fields_list}

Field definitions:
- generic_name: Product name / commodity name
- mrp: Maximum Retail Price (look for Rs., ₹, MRP, or decimal prices like 135.00)
- net_quantity: Net Qty / Net Weight / Net Volume / Piece count
- mfg_date: Date of Manufacture / Mfg Date / PKD (e.g. 03/2026, 3/27)
- expiry_date: Best Before / Expiry Date
- manufacturer_details: Full Manufacturer / Packer name and address
- consumer_care: Consumer Care contact — phone, toll-free (1800-xxx-xxxx), or email
- country_of_origin: Country of Origin / Made in

For each field, return the exact text you can read from the label.
If a field is not visible or not present on this label, omit it from the response.

Return ONLY valid JSON with no markdown, in this exact schema:
{{
  "generic_name": "extracted text or omit key if absent",
  "mrp": "extracted text or omit key if absent",
  "net_quantity": "extracted text or omit key if absent"
}}
Only include keys for fields you can actually read.
"""

    try:
        from PIL import Image
        img = Image.open(image_path)

        response = generate_content_resilient(
            contents=[img, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            )
        )
        if not response:
            return {}
        raw = response.text.strip()
        data = json.loads(raw)
        # Only return fields that were requested, with non-empty string values
        result = {}
        for field in missing_fields:
            val = data.get(field)
            if val and isinstance(val, str) and val.strip():
                result[field] = val.strip()
        logger.info(f"Gemini batch vision resolved {len(result)}/{len(missing_fields)} missing fields")
        return result
    except Exception as e:
        logger.debug(f"Gemini batch vision error: {e}")
        return {}


def get_ai_resolved_matches(limit: int = 50) -> List[Dict[str, Any]]:
    if not AI_MATCH_LOG_FILE.exists():
        return []
    records = []
    try:
        with open(AI_MATCH_LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records[-limit:]
    except Exception as e:
        logger.error(f"Failed to read AI-resolved match log: {e}")
        return []
