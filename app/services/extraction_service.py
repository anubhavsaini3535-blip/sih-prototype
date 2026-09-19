import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from app.services.field_matcher import (
    field_matcher,
    FieldMatchResult,
)
from app.services.gemini_service import batch_extract_with_gemini_cot

logger = logging.getLogger(__name__)


class ExtractedField:
    def __init__(
        self,
        name: str,
        value: Optional[str],
        confidence: float,
        bounding_box: Optional[Dict[str, Any]] = None,
        height: Optional[float] = None,
        raw_snippet: Optional[str] = None,
        match_layer: Optional[str] = "exact_synonym",
        ai_reasoning: Optional[str] = None,
    ):
        self.name = name
        self.value = value.strip() if value else None
        self.confidence = round(confidence, 4)
        self.bounding_box = bounding_box
        self.height = height
        self.raw_snippet = raw_snippet
        self.match_layer = match_layer
        self.ai_reasoning = ai_reasoning

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_name": self.name,
            "extracted_value": self.value,
            "confidence_score": self.confidence,
            "bounding_box": self.bounding_box,
            "height": self.height,
            "raw_snippet": self.raw_snippet,
            "match_layer": self.match_layer,
            "ai_reasoning": self.ai_reasoning,
        }


# Standard regex patterns for Legal Metrology Rule 6 declarations
MRP_PATTERNS = [
    re.compile(
        r"(?:(?:M\.?R\.?P\.?|Maximum\s*Retail\s*Price|Price)\s*[:.-]?\s*)?(?:₹|Rs\.?|INR)\s*(\d+(?:[.,]\d{1,2})?(?:\s*\/-\s*(?:Per\s*[^,\n]+)?)?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(\d+(?:[.,]\d{1,2})?\s*\/-\s*(?:Per\s*[^,\n]+)?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:₹|Rs\.?|INR)\s*(\d+(?:[.,]\d{1,2})?)",
        re.IGNORECASE,
    ),
    # Standalone decimal numbers like 135.00, 135.00/-, MRP 135.00
    re.compile(
        r"(?:(?:M\.?R\.?P\.?|Maximum\s*Retail\s*Price|Price|Rate)\s*[:.-]?\s*)?(?:₹|Rs\.?|INR)?\s*(\b\d+\.\d{2}\b)(?:\s*\/-)?",
        re.IGNORECASE,
    ),
]

NET_QTY_PATTERNS = [
    re.compile(
        r"(?:Net\s*(?:Quantity|Qty|Wt\.?|Weight|Volume|Content)|Package\s*Quantity|Net|Qty)\s*[:.-]?\s*(\d+(?:\.\d+)?\s*(?:[A-Za-z]+\s+)*(?:kg|g|gm|gms|ml|l|ltr|litres?|mg|pieces|pcs|units?|N|U|pens?|refills?|tablets?|capsules?|sheets?|items?|numbers?|nos?)\b.*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(\d+(?:\.\d+)?\s*(?:[A-Za-z]+\s+)*(?:kg|g|gm|gms|ml|l|ltr|litres?|mg|pieces|pcs|units?|N|U|pens?|refills?|tablets?|capsules?|sheets?|items?|numbers?|nos?)\b.*)",
        re.IGNORECASE,
    ),
]

MFG_DATE_PATTERNS = [
    re.compile(
        r"(?:(?:Mfg(?:\s*Date)?|Mfg\.?|Date\s*of\s*Mfg|Date\s*of\s*Packing|PKD|Packed|Date\s*of\s*Pkg|Mfd|Pkg|Month\s*and\s*year\s*of(?:\s*Manufacture)?)\s*[:.-]?\s*)?([0-3]?\d[\/\-\.](?:0?[1-9]|1[0-2])[\/\-\.](?:20\d{2}|\d{2})|(?:0?[1-9]|1[0-2])[\/\-\.](?:20\d{2}|\d{2})|[A-Za-z]{3,9}[\s\/\-\.](?:20\d{2}|\d{2}))",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\s\/\-\.]+(20\d{2}|\d{2})\b",
        re.IGNORECASE,
    ),
    # Matches dates like 3/27, 03/27, 12/26, 3-27, 3.27, 3/2027
    re.compile(
        r"\b(0?[1-9]|1[0-2])[\/\-\.](20\d{2}|\d{2})\b",
        re.IGNORECASE,
    ),
    # Full date formats like 15/03/27, 15/3/27, 15-03-2027
    re.compile(
        r"\b([0-3]?\d[\/\-\.](?:0?[1-9]|1[0-2])[\/\-\.](?:20\d{2}|\d{2}))\b",
        re.IGNORECASE,
    ),
]

EXPIRY_PATTERNS = [
    re.compile(
        r"(?:Best\s*Before|Expiry|Exp\.?|Exp\s*Date|Use\s*By)\s*[:.-]?\s*(\d{1,2}\s*months?\s*(?:from|of)\s*(?:mfg|pkg|packaging|date)?|[0-3]?\d[\/\-\.][0-1]?\d[\/\-\.]\d{2,4}|[0-1]?\d[\/\-\.]\d{4}|[A-Za-z]{3,9}[\s\/\-\.]\d{2,4})",
        re.IGNORECASE,
    ),
]

COUNTRY_OF_ORIGIN_PATTERNS = [
    re.compile(
        r"(?:Country\s*of\s*Origin|Made\s*in|Origin|Manufactured\s*in)\s*[:.-]?\s*([A-Za-z\s]{3,25})",
        re.IGNORECASE,
    ),
]

KNOWN_COUNTRIES = [
    "India", "Bharat", "Japan", "China", "USA", "United States", "Germany",
    "United Kingdom", "UK", "France", "Italy", "Vietnam", "Thailand",
    "Malaysia", "Taiwan", "Korea", "South Korea", "Indonesia",
    "Bangladesh", "Sri Lanka", "Switzerland", "Canada", "Australia", "Singapore"
]

DISALLOWED_ORIGIN_WORDS = {
    "manufacture", "manufacturing", "manufacturer", "commodity", "company",
    "private", "limited", "package", "quantity", "retail", "price", "pack",
    "date", "month", "year", "number", "product", "blue", "black", "point",
    "name", "care", "customer", "pen", "ink", "box"
}

# Consumer Care Toll-Free, Telephone, and Email Patterns (Legal Metrology Rule 6(1)(da))
TOLL_FREE_PATTERNS = [
    # 1800 / 1860 toll-free numbers (e.g. 1800-11-2233, 1800 200 1122, 18004253242)
    re.compile(r"\b18[06]0[- ]?\d{2,4}[- ]?\d{3,4}\b"),
    # Landline with STD code (e.g. 011-23456789, 080-22998877)
    re.compile(r"\b0\d{2,4}[- ]?\d{6,8}\b"),
    # Mobile / Helpline with +91 or standard 10 digits
    re.compile(r"(?:\+91[- ]?|(?<!\d))([6-9]\d{4}[- ]?\d{5})\b"),
    # Labeled Toll-Free or Helpline strings
    re.compile(r"(?:Toll[- ]?Free|T[- ]?Free|Helpline|Customer\s*Care|Call)\s*(?:No\.?|Number)?\s*[:.-]?\s*([+0-9A-Za-z\(\)\s-]{7,35})", re.IGNORECASE),
]

EMAIL_PATTERNS = [
    # Standard email format
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    # OCR corrupted @ symbols (e.g. care©brand.com, care(a)brand.com, care[at]brand.com)
    re.compile(r"\b([A-Za-z0-9._%+-]+)\s*(?:[©®]|(?:\(a\))|\[at\])\s*([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b", re.IGNORECASE),
]


def normalize_ocr_contact_text(text: str) -> str:
    """
    Normalizes OCR email corruptions (e.g. care©brand.com -> care@brand.com)
    and removes duplicate punctuation/spacing in contact declarations.
    """
    if not text:
        return ""
    # Normalize @ symbol errors from OCR font misrecognition
    text = re.sub(r"\b([A-Za-z0-9._%+-]+)\s*[©®]\s*([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b", r"\1@\2", text)
    text = re.sub(r"\b([A-Za-z0-9._%+-]+)\s*(?:\(a\)|\[at\])\s*([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b", r"\1@\2", text, flags=re.IGNORECASE)
    text = re.sub(r"\b([A-Za-z0-9._%+-]+)\s*@\s*([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b", r"\1@\2", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _vertical_overlap_ratio(b1: Dict[str, Any], b2: Dict[str, Any]) -> float:
    box1, box2 = b1.get("bounding_box"), b2.get("bounding_box")
    if not box1 or not box2:
        return 0.0
    y1_min, y1_max = box1.get("y_min", 0), box1.get("y_max", 0)
    y2_min, y2_max = box2.get("y_min", 0), box2.get("y_max", 0)
    overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    h1 = max(1, y1_max - y1_min)
    h2 = max(1, y2_max - y2_min)
    return overlap / min(h1, h2)


def find_spatial_candidates(
    key_block: Dict[str, Any],
    all_blocks: List[Dict[str, Any]],
    key_index: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Finds value blocks spatially related to a key label block in tabular/label layouts:
    1. Horizontal right-hand blocks on the same row (y-overlap > 0.30, x_min >= key.x_min + 15)
    2. Vertical child blocks directly underneath (allowing compact line box overlap down to -35px, and up to 65px vertical gap)
    """
    k_box = key_block.get("bounding_box")
    if not k_box:
        subsequent = all_blocks[key_index + 1 : min(len(all_blocks), key_index + 4)]
        return subsequent, []

    candidates_right: List[Tuple[float, Dict[str, Any]]] = []
    candidates_below: List[Tuple[float, Dict[str, Any]]] = []

    for i, b in enumerate(all_blocks):
        if i == key_index:
            continue
        # Only evaluate spatial neighbors belonging to the same image panel
        if b.get("image_index", 0) != key_block.get("image_index", 0):
            continue
        b_box = b.get("bounding_box")
        if not b_box:
            continue

        v_over = _vertical_overlap_ratio(key_block, b)
        if v_over > 0.30 and b_box["x_min"] >= (k_box["x_min"] + 15):
            candidates_right.append((b_box["x_min"], b))
        elif (
            (-35 <= (b_box["y_min"] - k_box["y_max"]) < 65)
            and (b_box["y_min"] > k_box["y_min"] + 10)
            and (abs(b_box["x_min"] - k_box["x_min"]) < 160 or (min(k_box["x_max"], b_box["x_max"]) - max(k_box["x_min"], b_box["x_min"]) > 20))
        ):
            candidates_below.append((b_box["y_min"], b))

    candidates_right.sort(key=lambda item: item[0])
    candidates_below.sort(key=lambda item: item[0])
    return [c[1] for c in candidates_right], [c[1] for c in candidates_below]


def _accumulate_address_block(
    key_block: Dict[str, Any],
    all_blocks: List[Dict[str, Any]],
    key_index: int,
) -> str:
    """
    Gathers all sequential continuation lines of an address block (e.g. Manufactured by,
    Marketed by, or Consumer Care) until a new declaration header, large vertical jump,
    or Indian postal PIN code boundary is reached.
    """
    text = key_block["text"].strip()
    lines = []
    clean_first = re.sub(
        r"^(?:A\s*Quality\s*Product\s*by|Quality\s*Product\s*by|Product\s*by|Produced\s*by|Manufactured\s*by|Marketed\s*by|Packed\s*by|Mfg\s*by|Mfd\s*by|Imported\s*by|Consumer\s*Care|Customer\s*Care|Feedback|Suggestions|Contact)\s*[:.-]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    if clean_first:
        lines.append(clean_first)

    curr_box = key_block.get("bounding_box")
    if not curr_box:
        return text

    HEADER_PATTERNS = [
        re.compile(r"^(?:Net\s*Qty|Net\s*Quantity|Package\s*Quantity|Qty|Net)\b", re.IGNORECASE),
        re.compile(r"^(?:MRP|M\.R\.P\.|Maximum\s*Retail\s*Price|Price|Rs\.?|₹)\b", re.IGNORECASE),
        re.compile(r"^(?:Mfg\s*Date|Date\s*of\s*Mfg|Date\s*of\s*Packing|Mfd|PKD|Packed|Month\s*and\s*year)\b", re.IGNORECASE),
        re.compile(r"^(?:Best\s*Before|Expiry|Exp\s*Date|Use\s*By)\b", re.IGNORECASE),
        re.compile(r"^(?:Country\s*of\s*Origin|Made\s*in|Origin)\b", re.IGNORECASE),
        re.compile(r"^(?:Marketed\s*by|Manufactured\s*by|Packed\s*by)\b", re.IGNORECASE),
        re.compile(r"^(?:FOR\s*FEEDBACK|FEEDBACK|CONSUMER\s*CARE|CUSTOMER\s*CARE|TOLL\s*FREE|CONTACT)\b", re.IGNORECASE),
    ]

    last_box = curr_box
    for i in range(key_index + 1, len(all_blocks)):
        cand = all_blocks[i]
        # Only accumulate continuation lines belonging to the same image panel
        if cand.get("image_index", 0) != key_block.get("image_index", 0):
            break
        cand_box = cand.get("bounding_box")
        if not cand_box:
            break
        cand_txt = cand["text"].strip()
        if not cand_txt:
            continue
        # Skip isolated 1-2 character stray OCR noise inside address block
        if len(cand_txt) <= 2 and not cand_txt.isdigit():
            continue
        # Stop if next block matches another regulatory field header
        if any(pat.search(cand_txt) for pat in HEADER_PATTERNS):
            break
        # Stop if pure standalone digit (like quantity count or barcode)
        if re.match(r"^\d{1,3}$", cand_txt):
            break
        # Vertical jump check (allow much larger gaps for high-res images)
        # We rely on HEADER_PATTERNS to break naturally at the next section,
        # so we can allow up to 6x the line height or 300px absolute gap.
        line_height = max(15, last_box["y_max"] - last_box["y_min"])
        y_diff = cand_box["y_min"] - last_box["y_max"]
        if y_diff > max(300, line_height * 6) or cand_box["y_min"] < last_box["y_min"] - 25:
            break

        lines.append(cand_txt)
        last_box = cand_box

        # Indian 6-digit PIN code marks end of standard postal address
        if re.search(r"\b[1-9]\d{2}\s?\d{3}\b", cand_txt):
            break

    # Join address lines naturally with comma/spacing cleanup
    joined = " ".join(lines)
    joined = re.sub(r"\s*,\s*", ", ", joined)
    joined = re.sub(r"\s+", " ", joined).strip()
    return joined if joined else text


def _extract_field_value_spatial(
    field_name: str,
    key_block: Dict[str, Any],
    right_blocks: List[Dict[str, Any]],
    below_blocks: List[Dict[str, Any]],
    all_blocks: List[Dict[str, Any]],
    key_index: int,
) -> Optional[str]:
    """
    Extracts the actual value corresponding to a key label by inspecting:
    1. Adjacent horizontal table cell blocks (right column)
    2. Adjacent vertical child blocks (below lines)
    3. The key block itself (if inline, e.g. "MRP: Rs. 299")
    """
    text = key_block["text"].strip()
    right_texts = [b["text"].strip() for b in right_blocks]
    below_texts = [b["text"].strip() for b in below_blocks]

    # 1. GENERIC / COMMON PRODUCT NAME
    if field_name == "generic_name":
        if ":" in text:
            val = text.split(":", 1)[1].strip()
            if len(val) >= 2 and not any(sk in val.lower() for sk in ["mrp", "net qty", "mfg date"]):
                return val
        if right_texts:
            return right_texts[0]
        if below_texts:
            b0 = below_texts[0]
            if not any(sk in b0.lower() for sk in ["mrp", "net qty", "mfg date", "product number", "price", "package"]):
                return b0
        cleaned = re.sub(r"^(?:Product\s*Name|Generic\s*Name|Commodity(?:\s*Name)?|Common\s*Name)\s*[:.-]?\s*", "", text, flags=re.IGNORECASE).strip()
        return cleaned if len(cleaned) >= 2 else text

    # 2. MAXIMUM RETAIL PRICE (MRP)
    elif field_name == "mrp":
        # Check inline key block first if it contains a price
        for pat in MRP_PATTERNS:
            m = pat.search(text)
            if m:
                val = m.group(0).strip()
                if not re.search(r"₹|Rs|INR", val, re.IGNORECASE):
                    val = f"₹ {val}"
                return val

        # Check right side blocks (and rows immediately below right side in table layout)
        prices = []
        for cand in right_texts + below_texts[:3]:
            cand_s = cand.strip()
            for pat in MRP_PATTERNS:
                m = pat.search(cand_s)
                if m:
                    p_str = cand_s
                    if not re.search(r"₹|Rs|INR", p_str, re.IGNORECASE):
                        p_str = f"₹ {p_str}"
                    if p_str not in prices:
                        prices.append(p_str)
                    break
            # Numeric price fallback: check if candidate is a price number like 65.00 or 65
            if not prices:
                num_m = re.match(r"^(?:₹|Rs\.?|INR)?\s*(\d+(?:[.,]\d{1,2})?)\s*(?:\/-)?$", cand_s, re.IGNORECASE)
                if num_m and not any(k in cand_s.lower() for k in ["page", "size", "cm", "mm", "gm", "ml", "qty", "net"]):
                    p_str = f"₹ {num_m.group(1)}"
                    if p_str not in prices:
                        prices.append(p_str)

        if key_index + 1 < len(all_blocks):
            for offset in range(1, 4):
                if key_index + offset >= len(all_blocks):
                    break
                chk_b = all_blocks[key_index + offset]
                chk_txt = chk_b["text"].strip()
                if any(chk_txt.startswith(curr) for curr in ["₹", "Rs", "INR"]) or "/-" in chk_txt:
                    if not re.search(r"₹|Rs|INR", chk_txt, re.IGNORECASE):
                        chk_txt = f"₹ {chk_txt}"
                    if chk_txt not in prices:
                        prices.append(chk_txt)
                elif re.match(r"^(?:₹|Rs\.?|INR)?\s*(\d+(?:[.,]\d{1,2})?)\s*(?:\/-)?$", chk_txt, re.IGNORECASE):
                    if not any(k in chk_txt.lower() for k in ["page", "size", "cm", "mm", "gm", "ml", "qty", "net"]):
                        val = re.search(r"(\d+(?:[.,]\d{1,2})?)", chk_txt).group(1)
                        chk_txt = f"₹ {val}"
                        if chk_txt not in prices:
                            prices.append(chk_txt)

        if prices:
            return ", ".join(prices)

        if re.search(r"MRP|Price", text, re.IGNORECASE):
            num_m = re.search(r"(\d+(?:[.,]\d{1,2})?)", text)
            if num_m:
                return f"₹ {num_m.group(1)}"
        return None

    # 3. NET QUANTITY
    elif field_name == "net_quantity":
        # Check inline key block first
        for pat in NET_QTY_PATTERNS:
            m = pat.search(text)
            if m:
                return m.group(1 if m.lastindex else 0).strip()
        # Check right table cell - join all horizontal items on the same row (e.g. '5' + 'Ball Pens' -> '5 Ball Pens')
        if right_texts:
            joined_right = " ".join(right_texts).strip()
            for pat in NET_QTY_PATTERNS:
                m = pat.search(joined_right)
                if m:
                    return m.group(0).strip()
            return joined_right
        # Check below cell (only if it matches quantity or does not start with another header)
        if below_texts:
            b0 = below_texts[0]
            if not any(sk in b0.lower() for sk in ["mrp", "mfg", "date", "country", "origin", "pkg"]):
                for pat in NET_QTY_PATTERNS:
                    m = pat.search(b0)
                    if m:
                        return m.group(0).strip()
                return b0
        return None

    # 4. DATE OF MANUFACTURE / PACK / IMPORT
    elif field_name == "mfg_date":
        # Check inline key block first
        for pat in MFG_DATE_PATTERNS:
            m = pat.search(text)
            if m:
                return m.group(1 if m.lastindex else 0).strip()
        # Check right table cell
        if right_texts:
            for pat in MFG_DATE_PATTERNS:
                m = pat.search(right_texts[0])
                if m:
                    return right_texts[0]
            return right_texts[0]
        # Check below cell
        if below_texts:
            b0 = below_texts[0]
            if not any(sk in b0.lower() for sk in ["mrp", "net qty", "country", "origin", "care"]):
                for pat in MFG_DATE_PATTERNS:
                    m = pat.search(b0)
                    if m:
                        return b0
        return None

    # 5. EXPIRY DATE / BEST BEFORE
    elif field_name == "expiry_date":
        for pat in EXPIRY_PATTERNS:
            m = pat.search(text)
            if m:
                return m.group(1 if m.lastindex else 0).strip()
        if right_texts:
            return right_texts[0]
        if below_texts:
            return below_texts[0]
        return None

    # 6. MANUFACTURER / PACKER / IMPORTER DETAILS
    elif field_name == "manufacturer_details":
        return _accumulate_address_block(key_block, all_blocks, key_index)

    # 7. CONSUMER CARE DETAILS
    elif field_name == "consumer_care":
        raw_base = _accumulate_address_block(key_block, all_blocks, key_index)
        base_val = normalize_ocr_contact_text(raw_base)
        extra_contact = []
        for idx, b in enumerate(all_blocks):
            if idx == key_index:
                continue
            t = normalize_ocr_contact_text(b["text"].strip())
            t_low = t.lower()
            has_phone = any(pat.search(t) for pat in TOLL_FREE_PATTERNS)
            has_email = any(pat.search(t) for pat in EMAIL_PATTERNS)
            is_contact_kw = any(k in t_low for k in ["t-free", "toll free", "toll-free", "helpline", "1800", "email", "customer care", "consumer care", "feedback"])
            if (has_phone or has_email or is_contact_kw) and len(t) >= 4:
                if t.lower() not in base_val.lower() and base_val.lower() not in t.lower():
                    if t not in extra_contact:
                        extra_contact.append(t)
        if extra_contact:
            combined = f"{base_val}, {', '.join(extra_contact)}" if base_val else ", ".join(extra_contact)
            return normalize_ocr_contact_text(combined)
        return base_val

    # 8. COUNTRY OF ORIGIN
    elif field_name == "country_of_origin":
        clean_cand = text.strip()
        if clean_cand.lower() in DISALLOWED_ORIGIN_WORDS:
            return None

        if right_texts:
            for country in KNOWN_COUNTRIES:
                if country.lower() in right_texts[0].lower():
                    return country

        for country in KNOWN_COUNTRIES:
            if country.lower() in text.lower():
                return country

        for pat in COUNTRY_OF_ORIGIN_PATTERNS:
            m = pat.search(text)
            if m:
                c_val = m.group(1).strip()
                if c_val.lower() not in DISALLOWED_ORIGIN_WORDS:
                    return c_val
        return None

    return text.strip()


def extract_from_blocks(
    blocks: List[Dict[str, Any]], scan_id: Optional[int] = None
) -> Dict[str, ExtractedField]:
    """
    Extracts Legal Metrology mandatory fields using a 3-Layer Matching Architecture
    with spatial table-layout correlation:
    1. Exact / Synonym Dictionary Match (with spatial row/column neighbor extraction)
       - Prioritizes 'Manufactured by' over 'Marketed by' and captures both if present
    2. Fuzzy String Match via RapidFuzz (threshold >= 85%) for typos/near-misses
    3. Standalone pattern & country of origin verification fallbacks
    4. Batch Chain-of-Thought Local Ollama LLM (phi4-mini) fallback for unmatched text
    """
    results: Dict[str, ExtractedField] = {}
    total_blocks = len(blocks)
    assigned_block_indices = set()

    all_rule_fields = [
        "manufacturer_details",
        "mrp",
        "net_quantity",
        "mfg_date",
        "expiry_date",
        "country_of_origin",
        "consumer_care",
        "generic_name",
    ]

    # =========================================================================
    # PASS 1a: MANUFACTURER & PACKER & MARKETER PRIORITY DETECTION
    # Under Legal Metrology Rule 6, if BOTH "Manufactured by" and "Marketed by"
    # appear, "Manufactured by" MUST take precedence, and both are captured.
    # =========================================================================
    mfg_cand_idx = None
    mktd_cand_idx = None
    for idx, b in enumerate(blocks):
        t_low = b["text"].strip().lower()
        if (
            any(k in t_low for k in [
                "manufactured by", "mfg by", "mfd by", "manufactured & packed",
                "manufacturer", "a quality product by", "quality product by"
            ])
            and mfg_cand_idx is None
        ):
            mfg_cand_idx = idx
        elif (
            any(k in t_low for k in ["marketed by", "mktd by", "marketed and distributed", "marketed & distributed"])
            and mktd_cand_idx is None
        ):
            mktd_cand_idx = idx

    if mfg_cand_idx is not None:
        mfg_b = blocks[mfg_cand_idx]
        mfg_val = _accumulate_address_block(mfg_b, blocks, mfg_cand_idx)
        if mktd_cand_idx is not None:
            mktd_b = blocks[mktd_cand_idx]
            mktd_val = _accumulate_address_block(mktd_b, blocks, mktd_cand_idx)
            final_val = f"{mfg_val} (Marketed by: {mktd_val})"
            assigned_block_indices.add(mktd_cand_idx)
        else:
            final_val = mfg_val

        results["manufacturer_details"] = ExtractedField(
            name="manufacturer_details",
            value=final_val,
            confidence=mfg_b["confidence"],
            bounding_box=mfg_b.get("bounding_box"),
            height=mfg_b.get("height"),
            raw_snippet=mfg_b["text"],
            match_layer="exact_synonym",
        )
        assigned_block_indices.add(mfg_cand_idx)
    elif mktd_cand_idx is not None:
        mktd_b = blocks[mktd_cand_idx]
        mktd_val = _accumulate_address_block(mktd_b, blocks, mktd_cand_idx)
        results["manufacturer_details"] = ExtractedField(
            name="manufacturer_details",
            value=mktd_val,
            confidence=mktd_b["confidence"],
            bounding_box=mktd_b.get("bounding_box"),
            height=mktd_b.get("height"),
            raw_snippet=mktd_b["text"],
            match_layer="exact_synonym",
        )
        assigned_block_indices.add(mktd_cand_idx)

    # =========================================================================
    # PASS 1b: LAYER 1 (Exact / Synonym Dictionary Match for remaining fields)
    # =========================================================================
    for field in all_rule_fields:
        if field in results:
            continue
        for idx, b in enumerate(blocks):
            if idx in assigned_block_indices:
                continue
            text = b["text"].strip()
            text_lower = text.lower()

            if field == "country_of_origin" and text_lower in DISALLOWED_ORIGIN_WORDS:
                continue
            if field == "manufacturer_details" and text_lower in ["manufacture", "manufacturing", "packed"]:
                continue

            match = field_matcher.match_text(text, target_fields=[field], enable_layer3=False)
            if match and match.match_layer == "exact_synonym":
                rights, belows = find_spatial_candidates(b, blocks, idx)
                extracted_val = _extract_field_value_spatial(field, b, rights, belows, blocks, idx)
                if extracted_val:
                    results[field] = ExtractedField(
                        name=field,
                        value=extracted_val,
                        confidence=b["confidence"],
                        bounding_box=b.get("bounding_box"),
                        height=b.get("height"),
                        raw_snippet=text,
                        match_layer="exact_synonym",
                    )
                    assigned_block_indices.add(idx)
                    break

    # =========================================================================
    # PASS 2: LAYER 2 (Fuzzy String Matching via RapidFuzz)
    # =========================================================================
    unresolved_fields = [f for f in all_rule_fields if f not in results]
    for field in unresolved_fields:
        for idx, b in enumerate(blocks):
            if idx in assigned_block_indices:
                continue
            text = b["text"].strip()
            text_lower = text.lower()

            if len(text) < 4:
                continue
            if field == "country_of_origin" and text_lower in DISALLOWED_ORIGIN_WORDS:
                continue
            if field == "manufacturer_details" and text_lower in ["manufacture", "manufacturing", "packed"]:
                continue

            match = field_matcher.match_text(text, target_fields=[field], enable_layer3=False)
            if match and match.match_layer == "fuzzy_matching":
                rights, belows = find_spatial_candidates(b, blocks, idx)
                extracted_val = _extract_field_value_spatial(field, b, rights, belows, blocks, idx)
                if extracted_val:
                    combined_conf = round(b["confidence"] * (match.similarity_score / 100.0), 4)
                    results[field] = ExtractedField(
                        name=field,
                        value=extracted_val,
                        confidence=combined_conf,
                        bounding_box=b.get("bounding_box"),
                        height=b.get("height"),
                        raw_snippet=text,
                        match_layer="fuzzy_matching",
                    )
                    assigned_block_indices.add(idx)
                    break

    # =========================================================================
    # PASS 3: Standalone Fallbacks for Unresolved Fields
    # =========================================================================
    # Country of Origin: check for known sovereign countries across all blocks
    if "country_of_origin" not in results:
        for idx, b in enumerate(blocks):
            t = b["text"].strip()
            for country in KNOWN_COUNTRIES:
                m = re.search(r"(?:Made in|from|Origin[:\s]+|Country of Origin[:\s]+)?\s*\b(" + country + r")\b", t, re.IGNORECASE)
                if m:
                    results["country_of_origin"] = ExtractedField(
                        name="country_of_origin",
                        value=country,
                        confidence=b["confidence"],
                        bounding_box=b.get("bounding_box"),
                        height=b.get("height"),
                        raw_snippet=t,
                        match_layer="exact_synonym",
                    )
                    assigned_block_indices.add(idx)
                    break
            if "country_of_origin" in results:
                break

    # MRP Fallback: check unassigned blocks for currency or standalone decimal price patterns
    if "mrp" not in results:
        for idx, b in enumerate(blocks):
            if idx in assigned_block_indices:
                continue
            t = b["text"].strip()
            # Standalone decimal numbers like 135.00 or with currency
            matched_price = None
            for pat in MRP_PATTERNS:
                m = pat.search(t)
                if m:
                    extracted = m.group(1 if m.lastindex else 0).strip()
                    if not any(unit in t.lower() for unit in ["cm", "mm", "gm", "ml", "kg", "pcs", "page"]):
                        matched_price = extracted
                        break
            if matched_price:
                if not re.search(r"₹|Rs|INR", matched_price, re.IGNORECASE):
                    matched_price = f"₹ {matched_price}"
                results["mrp"] = ExtractedField(
                    name="mrp",
                    value=matched_price,
                    confidence=b["confidence"],
                    bounding_box=b.get("bounding_box"),
                    height=b.get("height"),
                    raw_snippet=t,
                    match_layer="exact_synonym",
                )
                assigned_block_indices.add(idx)
                break

    # Date of Manufacture / PKD Fallback:
    # Scan unassigned blocks for standalone date patterns like "3/27", "03/27", "12/26", etc.
    if "mfg_date" not in results:
        for idx, b in enumerate(blocks):
            if idx in assigned_block_indices:
                continue
            t = b["text"].strip()
            # Guard against consumer care numbers, web domains, or pure numbers
            if any(k in t.lower() for k in ["toll", "care", "tel", "phone", "email", "www", "http"]):
                continue
            for pat in MFG_DATE_PATTERNS:
                m = pat.search(t)
                if m:
                    extracted = m.group(1 if m.lastindex else 0).strip()
                    if any(exp in t.lower() for exp in ["exp", "expiry", "best before", "use by"]):
                        if "expiry_date" not in results:
                            results["expiry_date"] = ExtractedField(
                                name="expiry_date",
                                value=extracted,
                                confidence=b["confidence"],
                                bounding_box=b.get("bounding_box"),
                                height=b.get("height"),
                                raw_snippet=t,
                                match_layer="exact_synonym",
                            )
                            assigned_block_indices.add(idx)
                            break
                    else:
                        results["mfg_date"] = ExtractedField(
                            name="mfg_date",
                            value=extracted,
                            confidence=b["confidence"],
                            bounding_box=b.get("bounding_box"),
                            height=b.get("height"),
                            raw_snippet=t,
                            match_layer="exact_synonym",
                        )
                        assigned_block_indices.add(idx)
                        break
            if "mfg_date" in results:
                break

    # Net Quantity Fallback: check unassigned blocks
    if "net_quantity" not in results:
        for idx, b in enumerate(blocks):
            if idx in assigned_block_indices:
                continue
            for pat in NET_QTY_PATTERNS:
                m = pat.search(b["text"])
                if m:
                    extracted = m.group(1 if m.lastindex else 0).strip()
                    results["net_quantity"] = ExtractedField(
                        name="net_quantity",
                        value=extracted,
                        confidence=b["confidence"],
                        bounding_box=b.get("bounding_box"),
                        height=b.get("height"),
                        raw_snippet=b["text"],
                        match_layer="exact_synonym",
                    )
                    assigned_block_indices.add(idx)
                    break
            if "net_quantity" in results:
                break

    # Generic Commodity Name Fallback:
    # 1. Deduce from net quantity if it contains commodity nouns (e.g. "5 Ball Pens" -> "Ball Pens")
    if "generic_name" not in results and "net_quantity" in results and results["net_quantity"].value:
        nq_val = results["net_quantity"].value
        m_noun = re.search(r"\d+\s*(?:N|U|units?|pieces|pcs)?\s*([A-Za-z\s]{3,30})$", nq_val)
        if m_noun:
            noun_cand = m_noun.group(1).strip()
            unit_exclusions = {"kg", "g", "gm", "gms", "ml", "l", "ltr", "litres", "mg", "pack", "count"}
            if noun_cand.lower() not in unit_exclusions:
                results["generic_name"] = ExtractedField(
                    name="generic_name",
                    value=noun_cand,
                    confidence=results["net_quantity"].confidence,
                    raw_snippet=nq_val,
                    match_layer="exact_synonym",
                )

    # 2. Check unassigned blocks for known stationery/packaging commodity keywords
    COMMODITY_KEYWORDS = [
        "long book", "exercise book", "notebook", "drawing book", "graph book",
        "scrap book", "practical book", "register", "ball pen", "gel pen", "fountain pen",
        "roller pen", "marker", "highlighter", "pencil", "eraser", "sharpener",
        "scale", "ruler", "geometry box", "maths book", "book"
    ]
    if "generic_name" not in results:
        for idx, b in enumerate(blocks):
            if idx in assigned_block_indices:
                continue
            cand_t = b["text"].strip()
            cand_low = cand_t.lower()
            if any(cw in cand_low for cw in COMMODITY_KEYWORDS):
                # Ensure it's not a company name
                if not any(corp in cand_low for corp in ["limited", "ltd", "pvt", "private", "product by"]):
                    # Clean up copyright or year symbols if present (e.g. '©24 Exercise Book' -> 'Exercise Book')
                    clean_name = re.sub(r"^[©®\d\s\-\/\.]+", "", cand_t).strip()
                    if len(clean_name) >= 3:
                        results["generic_name"] = ExtractedField(
                            name="generic_name",
                            value=clean_name if len(clean_name) >= 3 else cand_t,
                            confidence=b["confidence"],
                            bounding_box=b.get("bounding_box"),
                            height=b.get("height"),
                            raw_snippet=cand_t,
                            match_layer="exact_synonym",
                        )
                        assigned_block_indices.add(idx)
                        break

    # 3. Check top 5 headline blocks for commodity product name
    if "generic_name" not in results:
        for idx in range(min(5, len(blocks))):
            if idx in assigned_block_indices:
                continue
            cand_t = blocks[idx]["text"].strip()
            cand_low = cand_t.lower()
            if (
                len(cand_t) >= 4
                and not any(sk in cand_low for sk in ["mrp", "mfg", "date", "origin", "waste", "other", "net", "pages", "incl"])
                and not re.search(r"₹|Rs|\d{4}", cand_t)
                and not any(corp in cand_low for corp in ["limited", "ltd", "pvt", "private", "inc", "corp"])
            ):
                results["generic_name"] = ExtractedField(
                    name="generic_name",
                    value=cand_t,
                    confidence=blocks[idx]["confidence"],
                    bounding_box=blocks[idx].get("bounding_box"),
                    height=blocks[idx].get("height"),
                    raw_snippet=cand_t,
                    match_layer="exact_synonym",
                )
                assigned_block_indices.add(idx)
                break

    # Consumer Care Fallback (Rule 6(1)(da)):
    # If consumer_care is not yet resolved, scan unassigned blocks for toll-free numbers, phone, or email patterns
    if "consumer_care" not in results:
        for idx, b in enumerate(blocks):
            if idx in assigned_block_indices:
                continue
            t = b["text"].strip()
            t_low = t.lower()
            has_phone = any(pat.search(t) for pat in TOLL_FREE_PATTERNS)
            has_email = any(pat.search(t) for pat in EMAIL_PATTERNS)
            is_contact = any(
                k in t_low
                for k in ["t-free", "toll-free", "toll free", "helpline", "customer care", "consumer care", "feedback"]
            )
            if (has_phone or has_email or is_contact) and len(t) >= 4:
                val = normalize_ocr_contact_text(t)
                rights, belows = find_spatial_candidates(b, blocks, idx)
                if belows:
                    extra_lines = [
                        normalize_ocr_contact_text(bb["text"].strip())
                        for bb in belows[:2]
                        if len(bb["text"].strip()) >= 3
                    ]
                    if extra_lines:
                        val = f"{val}, {', '.join(extra_lines)}"
                results["consumer_care"] = ExtractedField(
                    name="consumer_care",
                    value=val,
                    confidence=b["confidence"],
                    bounding_box=b.get("bounding_box"),
                    height=b.get("height"),
                    raw_snippet=t,
                    match_layer="exact_synonym",
                )
                assigned_block_indices.add(idx)
                break

    # Country of Origin Fallback (Legal Metrology Rule 6(10) Domestic Origin):
    # If Country of Origin is not explicitly labeled, check manufacturer address or text blocks
    # for Indian states, major cities, or 6-digit postal PIN code, or GS1 India barcode prefix '890'.
    if "country_of_origin" not in results:
        INDIAN_STATES_AND_CITIES = [
            "Maharashtra", "Mumbai", "Delhi", "New Delhi", "Karnataka", "Bengaluru", "Bangalore",
            "Tamil Nadu", "Chennai", "Gujarat", "Ahmedabad", "Uttar Pradesh", "West Bengal",
            "Kolkata", "Rajasthan", "Telangana", "Hyderabad", "Kerala", "Madhya Pradesh",
            "Haryana", "Punjab", "Bihar", "Odisha", "Andhra Pradesh", "Assam", "Jharkhand",
            "Uttarakhand", "Goa", "Pune", "Dadar", "Noida", "Gurugram", "Gurgaon"
        ]
        mfg_addr = results.get("manufacturer_details")
        all_text_blobs = [mfg_addr.value] if (mfg_addr and mfg_addr.value) else []
        all_text_blobs.extend(b["text"] for b in blocks)

        has_indian_geo = False
        has_pin_code = False
        has_gs1_890 = False
        for blob in all_text_blobs:
            if not blob:
                continue
            if any(geo.lower() in blob.lower() for geo in INDIAN_STATES_AND_CITIES):
                has_indian_geo = True
            if re.search(r"\b[1-9]\d{2}\s?\d{3}\b", blob):
                has_pin_code = True
            if re.search(r"\b890\d{10}\b", blob):
                has_gs1_890 = True

        if (has_indian_geo and has_pin_code) or has_gs1_890:
            results["country_of_origin"] = ExtractedField(
                name="country_of_origin",
                value="India",
                confidence=0.98,
                raw_snippet="Domestic manufacturer address with Indian State/PIN code or GS1 890 barcode prefix",
                match_layer="exact_synonym",
            )

    # =========================================================================
    # PASS 4: LAYER 3 (Automated AI Deep Extraction via Gemini CoT)
    # Automatically extracts any remaining missing fields using Gemini AI
    # =========================================================================
    missing_fields = [f for f in all_rule_fields if f not in results]
    if missing_fields:
        from app.services.gemini_service import is_ai_available, batch_extract_with_gemini_cot
        if is_ai_available():
            all_snippets = [b["text"].strip() for b in blocks if b.get("text", "").strip()]
            extracted_so_far = {k: v.value for k, v in results.items() if v.value}
            try:
                ai_resolved = batch_extract_with_gemini_cot(
                    all_snippets=all_snippets,
                    missing_fields=missing_fields,
                    extracted_so_far=extracted_so_far,
                    scan_id=scan_id,
                )
                for f_name, info in ai_resolved.items():
                    if f_name not in results and info.get("extracted_value"):
                        results[f_name] = ExtractedField(
                            name=f_name,
                            value=info["extracted_value"],
                            confidence=info.get("confidence", 0.90),
                            raw_snippet=info.get("raw_snippet", info["extracted_value"]),
                            match_layer="gemini_api",
                            ai_reasoning=info.get("logical_rationale"),
                        )
            except Exception as e:
                logger.warning(f"Pass 4 Gemini CoT extraction failed: {e}")

    return results


def extract_from_text(raw_text: str, scan_id: Optional[int] = None) -> Dict[str, ExtractedField]:
    """
    Extracts fields directly from arbitrary text (e.g. scraped HTML from e-commerce listing).
    Treats each line as an OCR block with 3-layer matching.
    """
    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
    dummy_blocks = [
        {"text": line, "confidence": 0.95, "bounding_box": None, "height": 16.0}
        for line in lines
    ]
    return extract_from_blocks(dummy_blocks, scan_id=scan_id)
