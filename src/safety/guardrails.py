import re
from typing import Optional

from src.models import (
    LocationContext,
    MerchantDetails,
    TransactionEvaluationRequest,
    TransactionEvaluationResponse,
)


HIGH_VALUE_THRESHOLD_USD = 2500.0
LOCATION_CONFIDENCE_THRESHOLD = 0.70


def approval_reason(
    request: TransactionEvaluationRequest,
    location: LocationContext,
    has_recommendations: bool,
    merchant: Optional[MerchantDetails] = None,
    requested_category: Optional[str] = None,
) -> Optional[str]:
    if not has_recommendations:
        return "No eligible wallet card remains after ToT constraint evaluation."
    if request.amount > HIGH_VALUE_THRESHOLD_USD:
        return (
            f"Transaction amount (${request.amount}) exceeds "
            f"high-value threshold (${HIGH_VALUE_THRESHOLD_USD:,.0f})."
        )
    if merchant is not None and merchant.resolution_method == "ai_candidate":
        return (
            f"We matched “{request.merchant_name}” to {merchant.canonical_name} "
            "using AI. Please confirm this is the correct merchant; category "
            "and reward estimates depend on that match."
        )
    if merchant is not None and merchant.resolution_method == "unresolved":
        return (
            f"We couldn't match “{request.merchant_name}” to the merchant catalog. "
            "Please confirm the merchant and its category; the estimate uses a "
            "general category for now."
        )
    if merchant is not None and merchant.requires_confirmation:
        return (
            f"{merchant.canonical_name} was recognized, but its transaction "
            "category may vary by department, location, or payment type. "
            f"We used “{merchant.primary_category}”; confirm what you purchased "
            "before relying on the estimate."
        )
    if (
        merchant is not None
        and requested_category is not None
        and requested_category != merchant.primary_category
    ):
        return (
            f"Your category selection “{requested_category}” differs from the "
            f"catalog category “{merchant.primary_category}” for "
            f"{merchant.canonical_name}. We used the catalog category; please "
            "confirm it is appropriate."
        )
    if merchant is None and (
        location.confidence_score < LOCATION_CONFIDENCE_THRESHOLD
        or location.is_ambiguous
    ):
        return (
            f"We couldn't confidently verify the category for "
            f"{merchant.canonical_name if merchant else request.merchant_name} "
            "from location information. Please confirm the merchant category "
            "before relying on the estimate."
        )
    return None


def is_grounded_summary(
    text: str, response: TransactionEvaluationResponse
) -> bool:
    if response.recommended_card is None:
        return False
    lowered = text.casefold()
    restricted_terms = (
        "approval",
        "approved",
        "eligible",
        "interest",
        "apr",
        "dollar",
        "cent",
        "fee",
        "benefit",
        "warranty",
        "protection",
        "reward",
        "yield",
        "value",
        "offer",
        "guarantee",
        "cash back",
        "cashback",
    )
    return (
        response.recommended_card.card_name.casefold() in lowered
        and not any(character.isdigit() for character in text)
        and "$" not in text
        and "%" not in text
        and not any(term in lowered for term in restricted_terms)
    )


def is_grounded_critique(
    text: str, response: TransactionEvaluationResponse
) -> bool:
    lowered = text.casefold()
    cards_by_id = {
        card.card_id: card.card_name
        for card in [response.recommended_card, *response.alternative_cards]
        if card is not None
    }
    for node in response.reasoning_tree:
        card_name = cards_by_id.get(node.card_id)
        if (
            not card_name
            or not node.evidence
            or node.evidence == "Foreign transaction fee eclipses reward yield."
            or card_name.casefold() not in lowered
        ):
            continue
        words = re.findall(r"[A-Za-z][A-Za-z'-]*", node.evidence)
        if any(
            " ".join(words[start : start + size]).casefold() in lowered
            for size in range(4, min(9, len(words) + 1))
            for start in range(len(words) - size + 1)
        ):
            return (
                bool(text.strip())
                and not any(character.isdigit() for character in text)
                and "$" not in text
                and "%" not in text
            )
    return False
