from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Literal, Optional

class LocationInput(BaseModel):
    latitude: Optional[float] = Field(
        None, ge=-90, le=90, description="Mock GPS latitude coordinate"
    )
    longitude: Optional[float] = Field(
        None, ge=-180, le=180, description="Mock GPS longitude coordinate"
    )
    venue_override: Optional[str] = Field(None, description="Direct venue check-in string")

class LocationContext(BaseModel):
    venue_name: str
    inferred_category: str
    is_ambiguous: bool
    confidence_score: float = Field(..., ge=0.0, le=1.0)

class MerchantLookupInput(BaseModel):
    raw_merchant_name: str

class MerchantDetails(BaseModel):
    canonical_name: str
    mcc_code: str
    primary_category: str
    qualifies_for_supermarket_multipliers: bool
    notes: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)
    requires_confirmation: bool = False
    resolution_method: Literal[
        "catalog_exact", "ai_candidate", "unresolved"
    ] = "catalog_exact"
    match_confidence: float = Field(1.0, ge=0.0, le=1.0)
    requires_confirmation: bool = False


class MerchantResolution(BaseModel):
    canonical_name: Optional[str]
    confidence: float = Field(..., ge=0.0, le=1.0)

class CreditCardProfile(BaseModel):
    card_id: str
    card_name: str
    issuer: str
    base_multipliers: Dict[str, float]
    active_quarterly_categories: List[str]
    has_purchase_protection: bool
    has_extended_warranty: bool
    foreign_transaction_fee: float = Field(
        ..., ge=0.0, le=100.0,
        description="Foreign transaction fee as a percentage, e.g. 3.0 means 3%",
    )
    purchase_protection_value_rate: float = Field(0.0, ge=0.0, le=1.0)
    extended_warranty_value_rate: float = Field(0.0, ge=0.0, le=1.0)
    is_frozen: bool = False
    is_expired: bool = False

class UserWallet(BaseModel):
    user_id: str
    point_valuation_cents: Dict[str, float]
    cards: List[CreditCardProfile]
    tie_break_preference: Literal["cash_back", "travel_points"] = "travel_points"

class TransactionEvaluationRequest(BaseModel):
    merchant_name: str = Field(..., min_length=1)
    amount: float = Field(
        ...,
        gt=0,
        le=50_000,
        allow_inf_nan=False,
        description="Transaction amount in USD",
    )
    category_hint: Optional[str] = None
    is_international: bool = False

    @field_validator("merchant_name")
    @classmethod
    def normalize_merchant_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Merchant name cannot be empty.")
        return normalized

class CardRecommendation(BaseModel):
    card_id: str
    card_name: str
    expected_yield_cents: float
    net_value_cents: float
    rationale: str
    has_purchase_protection: bool
    has_extended_warranty: bool
    foreign_transaction_fee: float

class ThoughtNode(BaseModel):
    depth: int = Field(..., ge=1, le=3)
    card_id: str
    status: Literal["evaluated", "pruned", "skipped"]
    expected_yield_cents: float = 0.0
    perk_value_cents: float = 0.0
    penalty_fee_cents: float = 0.0
    net_value_cents: Optional[float] = None
    evidence: Optional[str] = None

class TransactionEvaluationResponse(BaseModel):
    merchant_canonical_name: str
    mcc_code: str
    primary_category: str
    recommended_card: Optional[CardRecommendation]
    alternative_cards: List[CardRecommendation]
    assistant_message: Optional[str] = None
    tree_assessment: Optional[str] = None
    requires_approval: bool = False
    approval_reason: Optional[str] = None
    audit_trail: List[str] = Field(default_factory=list)
    reasoning_tree: List[ThoughtNode] = Field(default_factory=list)
