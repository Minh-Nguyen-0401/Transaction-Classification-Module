import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from .schemas import (
    ClassifyRequest,
    ClassifyResponse,
    LayerResult,
    OptionItem,
    OptionsResponse,
)
from .services.data_loader import (
    get_default_recipient,
    get_default_sender,
    list_recipients,
    list_senders,
    load_recipient_names,
)
from .services.llm_layer import call_gemini
from .services.model_layer import model_infer
from .services.rule_layer import rule_classify

logger = logging.getLogger(__name__)

app = FastAPI(title="Tranx Classifier Demo")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/options", response_model=OptionsResponse)
def options(limit: int = 50):
    senders = [
        OptionItem(code=row["sender_id"], display_name=row["sender_id"])
        for row in list_senders(limit=limit)
    ]
    names = load_recipient_names()
    recipients = [
        OptionItem(
            code=row["recipient_entity_id"],
            display_name=names.get(row["recipient_entity_id"], row["recipient_entity_id"]),
        )
        for row in list_recipients(limit=limit)
    ]
    return OptionsResponse(
        default_sender=get_default_sender(),
        default_recipient=get_default_recipient(),
        senders=senders,
        recipients=recipients,
    )


def _layer_result(status: str, label: Optional[str] = None, confidence: Optional[float] = None, detail: str = ""):
    return LayerResult(status=status, label=label, confidence=confidence, detail=detail or None)


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    sender_id = req.sender_id or get_default_sender()
    recipient_id = req.recipient_entity_id or get_default_recipient()
    if not sender_id or not recipient_id:
        raise HTTPException(status_code=400, detail="Missing sender or recipient options")

    # Layer 1: rule-based
    l1 = rule_classify(req.message, req.tranx_type)
    layer1_res = _layer_result(
        "success" if l1["success"] else "fail",
        label=l1.get("label"),
        confidence=0.9 if l1["success"] else None,
        detail=l1.get("reason"),
    )
    if l1["success"]:
        return ClassifyResponse(
            layer1=layer1_res,
            layer2=_layer_result("skipped", detail="Stopped at layer 1"),
            layer3=_layer_result("skipped", detail="Stopped at layer 1"),
            final_label=l1["label"],
            final_confidence=0.9,
            decided_by="layer1_rule",
        )

    # Layer 2: LLM
    try:
        l2 = call_gemini(req.message, recipient_id)
        layer2_res = _layer_result(
            "success" if l2["success"] else "fail",
            label=l2.get("label"),
            confidence=l2.get("confidence"),
            detail=l2.get("raw"),
        )
        if l2["success"] and l2.get("label") != "unknown":
            return ClassifyResponse(
                layer1=layer1_res,
                layer2=layer2_res,
                layer3=_layer_result("skipped", detail="Stopped at layer 2"),
                final_label=l2["label"],
                final_confidence=l2.get("confidence") or 0.0,
                decided_by="layer2_llm",
            )
    except Exception as exc:
        logger.exception("LLM call failed: %s", exc)
        layer2_res = _layer_result("fail", detail=f"LLM error: {exc}")

    # Layer 3: model
    try:
        l3 = model_infer(
            amount=req.amount,
            msg=req.message,
            sender_id=sender_id,
            recipient_id=recipient_id,
            tranx_type=req.tranx_type or "transfer_in",
        )
        layer3_res = _layer_result(
            "success",
            label=l3.get("label"),
            confidence=l3.get("confidence"),
            detail="tri-tower inference",
        )
        final_label = l3.get("label") or "unknown"
        return ClassifyResponse(
            layer1=layer1_res,
            layer2=layer2_res,
            layer3=layer3_res,
            final_label=final_label,
            final_confidence=l3.get("confidence") or 0.0,
            decided_by="layer3_model",
        )
    except ValueError as exc:  # common data issues -> surface as 400
        logger.exception("Model inference failed: %s", exc)
        raise HTTPException(status_code=400, detail=f"Model inference failed: {exc}") from exc
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Model inference failed: %s", exc)
        layer3_res = _layer_result("fail", label=None, confidence=None, detail=str(exc))
        return ClassifyResponse(
            layer1=layer1_res,
            layer2=layer2_res,
            layer3=layer3_res,
            final_label="OTH",
            final_confidence=0.0,
            decided_by="fallback_oth",
        )


@app.get("/", response_class=HTMLResponse)
def index():
    path = Path(__file__).parent / "frontend" / "index.html"
    html = path.read_text(encoding="utf-8")
    return HTMLResponse(content=html)
