from dataclasses import dataclass
from typing import Callable, List, Tuple

from src.models import (
    CardRecommendation,
    CreditCardProfile,
    ThoughtNode,
    TransactionEvaluationRequest,
    UserWallet,
)

HIGH_RISK_CATEGORIES = {"travel", "electronics", "jewelry"}
LOW_VALUE_THRESHOLD_USD = 50.0


@dataclass
class _CardBranch:
    card: CreditCardProfile
    expected_yield_cents: float
    valuation_key: str
    point_value_cents: float
    multiplier: float
    perk_value_cents: float = 0.0
    evidence: str = ""


def _point_valuation_key(card: CreditCardProfile) -> str:
    if "chase" in card.card_id:
        return "chase_ur"
    if "amex" in card.card_id:
        return "amex_mr"
    if "capital_one" in card.card_id:
        return "capital_one_miles"
    return "cash"


def _recommendation(
    card: CreditCardProfile,
    multiplier: float,
    expected_yield_cents: float,
    perk_value_cents: float,
    penalty_fee_cents: float,
    category: str,
    valuation_key: str,
    point_value_cents: float,
    evidence: str = "",
) -> CardRecommendation:
    rationale = (
        f"Earns {multiplier:g}x on {category} "
        f"({valuation_key} valued at {point_value_cents}¢/pt)."
    )
    if perk_value_cents:
        rationale += f" Estimated eligible perks add {perk_value_cents:.1f}¢."
    if penalty_fee_cents:
        rationale += (
            f" Incurs {card.foreign_transaction_fee}% FTF penalty "
            f"(-{penalty_fee_cents:.1f}¢)."
        )
    if evidence:
        rationale += f" Benefit evidence: {evidence}"

    return CardRecommendation(
        card_id=card.card_id,
        card_name=card.card_name,
        expected_yield_cents=round(expected_yield_cents, 2),
        net_value_cents=round(
            expected_yield_cents + perk_value_cents - penalty_fee_cents, 2
        ),
        rationale=rationale,
        has_purchase_protection=card.has_purchase_protection,
        has_extended_warranty=card.has_extended_warranty,
        foreign_transaction_fee=card.foreign_transaction_fee,
    )


def evaluate_wallet_tree(
    request: TransactionEvaluationRequest,
    wallet: UserWallet,
    category: str,
    qualifies_for_supermarket_multipliers: bool,
    query_benefits: Callable[[str, str], str],
) -> Tuple[List[CardRecommendation], List[ThoughtNode], List[str]]:
    """Evaluate every wallet branch breadth-first across yield, perks, and fees."""
    audit: List[str] = []
    nodes: List[ThoughtNode] = []
    branches: List[_CardBranch] = []

    for card in wallet.cards:
        if card.is_frozen or card.is_expired:
            nodes.append(
                ThoughtNode(
                    depth=1,
                    card_id=card.card_id,
                    status="pruned",
                    evidence="Card is frozen or expired.",
                )
            )
            audit.append(f"Pruned {card.card_name}: card is frozen or expired.")
            continue

        multiplier = card.base_multipliers.get(
            category, card.base_multipliers.get("default", 1.0)
        )
        if category == "groceries" and not qualifies_for_supermarket_multipliers:
            multiplier = card.base_multipliers.get("default", 1.0)
            audit.append(
                f"Card {card.card_name}: Supermarket exclusion applies; "
                f"fallback to {multiplier}x."
            )

        valuation_key = _point_valuation_key(card)
        point_value_cents = wallet.point_valuation_cents.get(valuation_key, 1.0)
        expected_yield_cents = request.amount * multiplier * point_value_cents
        branches.append(
            _CardBranch(
                card=card,
                expected_yield_cents=expected_yield_cents,
                valuation_key=valuation_key,
                point_value_cents=point_value_cents,
                multiplier=multiplier,
            )
        )
        nodes.append(
            ThoughtNode(
                depth=1,
                card_id=card.card_id,
                status="evaluated",
                expected_yield_cents=round(expected_yield_cents, 2),
                net_value_cents=round(expected_yield_cents, 2),
            )
        )

    if not branches:
        return [], nodes, audit

    early_exit = (
        request.amount < LOW_VALUE_THRESHOLD_USD
        and not request.is_international
        and category not in HIGH_RISK_CATEGORIES
    )
    if early_exit:
        audit.append(
            f"ToT early exit: local transaction under "
            f"${LOW_VALUE_THRESHOLD_USD:.0f}; evaluated base yield only."
        )
        recommendations = [
            _recommendation(
                branch.card,
                branch.multiplier,
                branch.expected_yield_cents,
                0.0,
                0.0,
                category,
                branch.valuation_key,
                branch.point_value_cents,
            )
            for branch in branches
        ]
        return _sort_recommendations(recommendations, wallet), nodes, audit

    for branch in branches:
        card = branch.card
        evidence = query_benefits(
            card.card_id,
            f"{category} purchase protection and extended warranty",
        )
        evidence_lower = evidence.lower()
        perk_rate = 0.0
        if card.has_purchase_protection and "purchase protection" in evidence_lower:
            perk_rate += card.purchase_protection_value_rate
        if card.has_extended_warranty and "extended warranty" in evidence_lower:
            perk_rate += card.extended_warranty_value_rate
        branch.evidence = evidence
        branch.perk_value_cents = request.amount * perk_rate * 100.0
        nodes.append(
            ThoughtNode(
                depth=2,
                card_id=card.card_id,
                status="evaluated",
                expected_yield_cents=round(branch.expected_yield_cents, 2),
                perk_value_cents=round(branch.perk_value_cents, 2),
                net_value_cents=round(
                    branch.expected_yield_cents + branch.perk_value_cents, 2
                ),
                evidence=evidence,
            )
        )

    viable_branches: List[_CardBranch] = []
    for branch in branches:
        card = branch.card
        fee = (
            request.amount * card.foreign_transaction_fee
            if request.is_international
            else 0.0
        )
        if request.is_international and fee >= branch.expected_yield_cents:
            nodes.append(
                ThoughtNode(
                    depth=3,
                    card_id=card.card_id,
                    status="pruned",
                    expected_yield_cents=round(branch.expected_yield_cents, 2),
                    perk_value_cents=round(branch.perk_value_cents, 2),
                    penalty_fee_cents=round(fee, 2),
                    net_value_cents=round(
                        branch.expected_yield_cents + branch.perk_value_cents - fee, 2
                    ),
                    evidence="Foreign transaction fee eclipses reward yield.",
                )
            )
            audit.append(
                f"Pruned {card.card_name}: foreign transaction fee "
                f"({fee:.1f}¢) meets or exceeds reward yield "
                f"({branch.expected_yield_cents:.1f}¢)."
            )
            continue

        nodes.append(
            ThoughtNode(
                depth=3,
                card_id=card.card_id,
                status="evaluated",
                expected_yield_cents=round(branch.expected_yield_cents, 2),
                perk_value_cents=round(branch.perk_value_cents, 2),
                penalty_fee_cents=round(fee, 2),
                net_value_cents=round(
                    branch.expected_yield_cents + branch.perk_value_cents - fee, 2
                ),
            )
        )
        viable_branches.append(branch)

    recommendations = [
        _recommendation(
            branch.card,
            branch.multiplier,
            branch.expected_yield_cents,
            branch.perk_value_cents,
            request.amount * branch.card.foreign_transaction_fee
            if request.is_international
            else 0.0,
            category,
            branch.valuation_key,
            branch.point_value_cents,
            branch.evidence,
        )
        for branch in viable_branches
    ]
    return _sort_recommendations(recommendations, wallet), nodes, audit


def _sort_recommendations(
    recommendations: List[CardRecommendation], wallet: UserWallet
) -> List[CardRecommendation]:
    def preference_rank(recommendation: CardRecommendation) -> int:
        if wallet.tie_break_preference == "cash_back":
            preferred = (
                "cash" in recommendation.card_id
                or "double_cash" in recommendation.card_id
            )
        else:
            preferred = any(
                issuer in recommendation.card_id for issuer in ("chase", "amex", "citi")
            )
        return 1 if preferred else 0

    return sorted(
        recommendations,
        key=lambda recommendation: (
            recommendation.net_value_cents,
            preference_rank(recommendation),
        ),
        reverse=True,
    )
