import re
from typing import Optional

from src.models import (
    LocationContext,
    TransactionEvaluationRequest,
    TransactionEvaluationResponse,
)


HIGH_VALUE_THRESHOLD_USD = 2500.0
LOCATION_CONFIDENCE_THRESHOLD = 0.70


def approval_reason(
    request: TransactionEvaluationRequest,
    location: LocationContext,
    has_recommendations: bool,
) -> Optional[str]:
    if not has_recommendations:
        return "No eligible wallet card remains after ToT constraint evaluation."
    if request.amount > HIGH_VALUE_THRESHOLD_USD:
        return (
            f"Transaction amount (${request.amount}) exceeds "
            f"high-value threshold (${HIGH_VALUE_THRESHOLD_USD:,.0f})."
        )
    if (
        location.confidence_score < LOCATION_CONFIDENCE_THRESHOLD
        or location.is_ambiguous
    ):
        return (
            "Merchant category confidence is low or ambiguous "
            f"({location.confidence_score})."
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
