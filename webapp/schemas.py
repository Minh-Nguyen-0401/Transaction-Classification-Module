from typing import List, Optional

from pydantic import BaseModel, Field, validator


class ClassifyRequest(BaseModel):
    message: str = Field(..., description="Transaction message/content")
    amount: float = Field(..., gt=0, description="Amount to transfer")
    recipient_entity_id: str
    sender_id: Optional[str] = None
    tranx_type: Optional[str] = None

    @validator("message")
    def non_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("message must not be empty")
        return v


class LayerResult(BaseModel):
    status: str
    label: Optional[str] = None
    confidence: Optional[float] = None
    detail: Optional[str] = None


class ClassifyResponse(BaseModel):
    layer1: LayerResult
    layer2: LayerResult
    layer3: LayerResult
    final_label: str
    final_confidence: float
    decided_by: str


class OptionItem(BaseModel):
    code: str
    display_name: str


class OptionsResponse(BaseModel):
    default_sender: Optional[str]
    default_recipient: Optional[str]
    senders: List[OptionItem]
    recipients: List[OptionItem]
