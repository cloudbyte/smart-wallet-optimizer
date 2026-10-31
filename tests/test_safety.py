from src.models import (
    LocationContext,
    MerchantDetails,
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


def test_unresolved_merchant_confirmation_explains_what_user_should_check():
    reason = approval_reason(
        TransactionEvaluationRequest(merchant_name="Unknown Merchant", amount=100.0),
        LocationContext(
            venue_name="Unknown Merchant",
            inferred_category="dining",
            is_ambiguous=False,
            confidence_score=0.95,
        ),
        has_recommendations=True,
        merchant=MerchantDetails(
            canonical_name="Unknown Merchant",
            mcc_code="5999",
            primary_category="miscellaneous",
            qualifies_for_supermarket_multipliers=False,
            resolution_method="unresolved",
            match_confidence=0.0,
            requires_confirmation=True,
        ),
    )

    assert "couldn't match" in reason
    assert "confirm the merchant and its category" in reason


def test_known_merchant_confirmation_message_does_not_suggest_match_failed():
    reason = approval_reason(
        TransactionEvaluationRequest(merchant_name="Walmart", amount=100.0),
        LocationContext(
            venue_name="Walmart Supercenter",
            inferred_category="wholesale_department",
            is_ambiguous=True,
            confidence_score=0.75,
        ),
        has_recommendations=True,
        merchant=MerchantDetails(
            canonical_name="Walmart",
            mcc_code="5311",
            primary_category="wholesale_department",
            qualifies_for_supermarket_multipliers=False,
            requires_confirmation=True,
        ),
    )

    assert reason.startswith("Walmart was recognized")
    assert "may vary by department" in reason
    assert "We couldn't match" not in reason


def test_ai_match_confirmation_names_suggested_merchant():
    reason = approval_reason(
        TransactionEvaluationRequest(merchant_name="TJ grocery", amount=100.0),
        LocationContext(
            venue_name="Trader Joe's",
            inferred_category="groceries",
            is_ambiguous=False,
            confidence_score=0.98,
        ),
        has_recommendations=True,
        merchant=MerchantDetails(
            canonical_name="Trader Joe's",
            mcc_code="5411",
            primary_category="groceries",
            qualifies_for_supermarket_multipliers=True,
            resolution_method="ai_candidate",
            match_confidence=0.92,
            requires_confirmation=True,
        ),
    )

    assert "matched" in reason
    assert "Trader Joe's" in reason
    assert "confirm this is the correct merchant" in reason


def test_summary_grounding_rejects_unverified_numeric_claims():
    response = TransactionEvaluationResponse(
        merchant_canonical_name="Trader Joe's",
        mcc_code="5411",
        primary_category="groceries",
        recommended_card=None,
        alternative_cards=[],
    )

    assert not is_grounded_summary("This gives 5% back.", response)
