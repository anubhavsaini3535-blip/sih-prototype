import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from rapidfuzz import fuzz

from app.config import FUZZY_MATCH_THRESHOLD
from app.services.gemini_service import classify_with_gemini, batch_extract_with_gemini_cot

logger = logging.getLogger(__name__)

# Standard Legal Metrology Rule 6 fields and their known synonym labels
FIELD_SYNONYMS: Dict[str, List[str]] = {
    "generic_name": [
        "Product Name",
        "Generic Name",
        "Common Name",
        "Commodity Name",
        "Name of Commodity",
        "Item Name",
        "Article",
        "Commodity",
        "Article Name",
        "Description",
    ],
    "mrp": [
        "Maximum Retail Price",
        "Price (Inclusive of all",
        "Price (Inclusive of",
        "Price (Incl",
        "MRP",
        "M.R.P.",
        "Retail Price",
        "Max Retail Price",
        "Max. Retail Price",
        "M. R. P.",
        "M.R.P",
        "MRP Rs",
        "MRP ₹",
        "MRP INR",
        "Price",
        "Max Price",
        "MRP Incl",
    ],
    "net_quantity": [
        "Package Quantity",
        "Net Quantity",
        "Net Qty",
        "Net Wt",
        "Net Weight",
        "Net Vol",
        "Net Volume",
        "Net Content",
        "Net Contents",
        "Quantity",
        "Qty",
        "Net Mass",
        "Net",
    ],
    "mfg_date": [
        "Month and year of Manufacture",
        "Month and year of",
        "Date of Manufacture",
        "Date of Manufacturing",
        "Mfg Date",
        "Date of Mfg",
        "Date of Packing",
        "Date of Pkg",
        "Mfg Dt",
        "PKD",
        "Packed",
        "Date of Pkd",
        "Mfg",
        "Mfd",
        "Month of Mfg",
        "Pkg Date",
    ],
    "expiry_date": [
        "Best Before",
        "Expiry Date",
        "Exp Date",
        "Date of Expiry",
        "Use By",
        "Expiry",
        "Exp",
        "Best By",
        "Best Before Date",
    ],
    "manufacturer_details": [
        "Manufactured by",
        "Mfg by",
        "Mfd by",
        "Manufactured & Packed by",
        "Manufacturer",
        "Packed by",
        "Marketed by",
        "Imported by",
        "Pkd by",
        "Mktd by",
        "Packer",
        "Marketed and Distributed by",
        "A Quality Product by",
        "Quality Product by",
        "Product by",
        "Produced by",
        "Reg Off",
        "Regd Off",
        "Regd. Off",
        "Registered Office",
        "Regd Office",
        "Regd. Office",
        "Corporate Office",
        "Head Office",
        "Office:",
        "Address:",
    ],
    "consumer_care": [
        "Consumer Care",
        "Customer Care",
        "Customer Support",
        "Customer Service",
        "Helpline",
        "Helpline No",
        "Toll Free",
        "Toll-Free",
        "Toll Free No",
        "Toll-Free No",
        "Consumer Cell",
        "Feedback",
        "Queries",
        "Contact Us",
        "Consumer Complaints",
        "For Customer Complaint",
        "For Complaints",
        "For Queries / Complaints",
        "Suggestions",
        "Feedback & Suggestions",
        "Feedback / Queries",
        "T-Free",
        "T-Free Number",
        "Contact:",
        "Contact :",
        "Consumer Helpline",
        "Customer Helpline",
        "Customer Care Executive",
        "Executive - Consumer Care",
        "Executive - Customer Care",
        "Helpdesk",
        "Email Us",
        "Call Us",
    ],
    "country_of_origin": [
        "Country of Origin",
        "Made in",
        "Country of Manufacture",
        "Produced in",
    ],
}


class FieldMatchResult:
    def __init__(
        self,
        field_name: str,
        match_layer: str,  # "exact_synonym" | "fuzzy_matching" | "gemini_api"
        matched_synonym: Optional[str] = None,
        similarity_score: float = 100.0,
        confidence: float = 1.0,
        ai_reasoning: Optional[str] = None,
    ):
        self.field_name = field_name
        self.match_layer = match_layer
        self.matched_synonym = matched_synonym
        self.similarity_score = round(similarity_score, 2)
        self.confidence = round(confidence, 4)
        self.ai_reasoning = ai_reasoning

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_name": self.field_name,
            "match_layer": self.match_layer,
            "matched_synonym": self.matched_synonym,
            "similarity_score": self.similarity_score,
            "confidence": self.confidence,
            "ai_reasoning": self.ai_reasoning,
        }





class FieldMatcher:
    """
    3-Layer Field Matching Architecture:
    Layer 1: Exact / Synonym dictionary match
    Layer 2: Fuzzy string matching using rapidfuzz (similarity >= 85%)
    Layer 3: Gemini API fallback for unmatched text
    """

    def __init__(self, synonym_dict: Optional[Dict[str, List[str]]] = None):
        self.synonyms = synonym_dict or FIELD_SYNONYMS

    def match_text(
        self,
        text: str,
        target_fields: Optional[List[str]] = None,
        scan_id: Optional[int] = None,
        enable_layer3: bool = True,
    ) -> Optional[FieldMatchResult]:
        """
        Runs 3-layer matching on the provided text snippet against target fields.
        """
        if not text or not text.strip():
            return None

        clean_text = text.strip()
        text_lower = clean_text.lower()
        candidate_fields = target_fields or list(self.synonyms.keys())

        # =========================================================================
        # LAYER 1: Exact / Synonym Dictionary Match (Longest match first)
        # =========================================================================
        all_candidates = []
        for field in candidate_fields:
            for syn in self.synonyms.get(field, []):
                all_candidates.append((field, syn))
        all_candidates.sort(key=lambda x: len(x[1]), reverse=True)

        for field, syn in all_candidates:
            syn_lower = syn.lower()
            if syn_lower in text_lower:
                # Corporate guard: company names with 'Limited', 'Ltd', etc. must not match generic_name
                if field == "generic_name" and any(
                    w in text_lower for w in ["limited", "ltd", "pvt", "private", "inc", "corp", "product by", "quality product"]
                ):
                    continue
                return FieldMatchResult(
                    field_name=field,
                    match_layer="exact_synonym",
                    matched_synonym=syn,
                    similarity_score=100.0,
                    confidence=1.0,
                )

        # =========================================================================
        # LAYER 2: Fuzzy String Matching (using rapidfuzz)
        # Only run on meaningful text (>= 4 characters) to avoid false noise matching
        # =========================================================================
        if len(clean_text) >= 4:
            # Common packaging noise words that should never trigger fuzzy matches
            DISALLOWED_NOISE = {"dry waste", "other", "sep 17", "blue", "black", "point", "product"}
            if text_lower in DISALLOWED_NOISE:
                pass
            else:
                best_field = None
                best_syn = None
                best_score = 0.0

                for field in candidate_fields:
                    synonyms = self.synonyms.get(field, [])
                    for syn in synonyms:
                        syn_lower = syn.lower()
                        # Corporate guard: company names must not fuzzy-match generic_name
                        if field == "generic_name" and any(
                            w in text_lower for w in ["limited", "ltd", "pvt", "private", "inc", "corp", "product by", "quality product"]
                        ):
                            continue
                        # Guard against a single generic word matching a multi-word phrase
                        if len(text_lower.split()) == 1 and len(syn_lower.split()) > 1:
                            score_partial = 0.0
                        else:
                            score_partial = fuzz.partial_ratio(text_lower, syn_lower)
                        # Token set ratio handles word reordering and insertions
                        score_tokens = fuzz.token_set_ratio(text_lower, syn_lower)
                        max_score = max(score_partial, score_tokens)

                        if max_score > best_score:
                            best_score = max_score
                            best_field = field
                            best_syn = syn

                if best_score >= FUZZY_MATCH_THRESHOLD and best_field is not None:
                    return FieldMatchResult(
                        field_name=best_field,
                        match_layer="fuzzy_matching",
                        matched_synonym=best_syn,
                        similarity_score=best_score,
                        confidence=round(best_score / 100.0, 4),
                    )

        # =========================================================================
        # LAYER 3: Gemini API Fallback (Single snippet mode)
        # =========================================================================
        if enable_layer3 and len(clean_text) >= 4:
            ai_result = classify_with_gemini(clean_text, candidate_fields)
            if ai_result:
                field, conf, reason = ai_result
                # Log this AI-resolved match for synonym dictionary review
                from app.services.gemini_service import log_ai_resolved_match
                log_ai_resolved_match(
                    input_snippet=clean_text,
                    field_name=field,
                    confidence=conf,
                    reason=reason,
                    scan_id=scan_id,
                )
                return FieldMatchResult(
                    field_name=field,
                    match_layer="gemini_api",
                    matched_synonym=None,
                    similarity_score=round(conf * 100, 2),
                    confidence=conf,
                    ai_reasoning=reason,
                )

        return None



# Global singleton instance
field_matcher = FieldMatcher()
