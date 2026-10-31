from src.models import (
    LocationContext,
    TransactionEvaluationRequest,
    TransactionEvaluationResponse,
)
from src.safety.guardrails import approval_reason, is_grounded_summary


def test_missing_recommendation_requires_approval_even_for_high_value_transaction():
    reason = approval_reason(
        TransactionEvaluationRequest(merchant_name="Unknown", amount=3000.0),
        LocationContext(
            venue_name="Unknown",
            inferred_category="miscellaneous",
            is_ambiguous=False,
            confidence_score=1.0,
        ),
        has_recommendations=False,
    )

    assert reason == "No eligible wallet card remains after ToT constraint evaluation."


def test_unverified_merchant_category_requires_approval():
    reason = approval_reason(
        TransactionEvaluationRequest(merchant_name="Unknown Merchant", amount=100.0),
        LocationContext(
            venue_name="Unknown Merchant",
            inferred_category="dining",
            is_ambiguous=False,
            confidence_score=0.95,
        ),
        has_recommendations=True,
        merchant_is_ambiguous=True,
    )

    assert reason == "Merchant category could not be verified from the local directory."


def test_summary_grounding_rejects_unverified_numeric_claims():
    response = TransactionEvaluationResponse(
        merchant_canonical_name="Trader Joe's",
        mcc_code="5411",
        primary_category="groceries",
        recommended_card=None,
        alternative_cards=[],
    )

    assert not is_grounded_summary("This gives 5% back.", response)
