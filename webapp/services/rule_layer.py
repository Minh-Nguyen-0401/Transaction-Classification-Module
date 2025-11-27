import re
import unicodedata
from typing import Dict, Optional

from .data_loader import load_label_map


def strip_accents(text: str) -> str:
    """Lowercase and remove accents for simple keyword matching."""
    text = text.lower()
    nfkd_form = unicodedata.normalize("NFKD", text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


# Map transaction type to label code
TRANX_TYPE_MAP = {
    "bill_payment": "BIL",
    "mobile_topup": "BIL",
    "cashback": "FIN",
    "loan_repayment": "FIN",
    "opensaving": "FIN",
    "stock": "FIN",
    "qrcode_payment": "SHP",
    # transfer_in no longer maps to OTH here; OTH is only fallback after all layers fail
}

# Keyword heuristics per label
KEYWORDS = {
    "BIL": ["hoa don", "bill", "dien", "nuoc", "evn", "vnpt", "viettel", "internet", "cap quang"],
    "FOO": ["an uong", "do an", "am thuc", "cafe", "tra sua", "food", "nha hang", "com", "pizza", "bun", "pho"],
    "TRN": ["grab", "be ", "taxi", "xe bus", "bus", "metro", "samsung pay", "xang", "bot"],
    "HLT": ["benh vien", "hospital", "clinic", "thuoc", "y te"],
    "INS": ["bao hiem", "insurance", "thue", "tax"],
    "SHP": ["sieu thi", "mall", "aeon", "vinmart", "shop", "mua sam", "tiki", "lazada"],
    "ENT": ["xem phim", "cinema", "karoke", "karaoke", "game", "nhac", "giai tri"],
    "EDU": ["hoc phi", "university", "hoc", "khoa hoc", "education", "course"],
    "FIN": ["vay", "loan", "tiet kiem", "dau tu", "fin"],
}


def rule_classify(msg: str, tranx_type: Optional[str] = None) -> Dict:
    """
    Try simple rule-based classification.
    Returns dict with keys: success(bool), label(Optional[str]), reason(str)
    """
    label_map = load_label_map()
    cleaned = strip_accents(msg or "")
    # Rule by transaction type
    if tranx_type:
        lbl = TRANX_TYPE_MAP.get(tranx_type)
        if lbl and lbl in label_map:
            return {"success": True, "label": lbl, "reason": f"Matched tranx_type={tranx_type}"}
    # Rule by keyword
    for lbl, kws in KEYWORDS.items():
        for kw in kws:
            # match whole word/phrase to avoid substring hits like "ban" -> "an"
            if re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", cleaned):
                return {"success": True, "label": lbl, "reason": f"Keyword match '{kw}'"}
    return {"success": False, "label": None, "reason": "No rule matched"}
