from types import SimpleNamespace
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.models import (
    CardRecommendation,
    TransactionEvaluationRequest,
    TransactionEvaluationResponse,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_APP_PATH = PROJECT_ROOT / "src" / "web_app.py"


def test_streamlit_app_loads_form_without_loading_model():
    app = AppTest.from_file(str(WEB_APP_PATH)).run()

    assert not app.exception
    assert any("Smart Wallet Rewards Optimizer" in item.value for item in app.markdown)
    assert any("Synthetic demo" in item.value for item in app.caption)
    assert app.text_input[0].label == "Merchant"
    assert app.button[0].label == "Find my best card →"


def test_streamlit_app_renders_structured_recommendation():
    app = AppTest.from_file(str(WEB_APP_PATH)).run()
    app.session_state["evaluation_response"] = TransactionEvaluationResponse(
        merchant_canonical_name="Trader Joe's",
        mcc_code="5411",
        primary_category="groceries",
        recommended_card=CardRecommendation(
            card_id="amex_gold",
            card_name="American Express Gold Card",
            expected_yield_cents=680.0,
            net_value_cents=680.0,
            rationale="Earns 4x on groceries.",
            has_purchase_protection=True,
            has_extended_warranty=False,
            foreign_transaction_fee=0.0,
        ),
        alternative_cards=[],
    )

    app.run()

    assert not app.exception
    assert any("American Express Gold Card" in item.value for item in app.markdown)


def test_streamlit_app_handles_response_cached_before_blocked_reason_field():
    app = AppTest.from_file(str(WEB_APP_PATH)).run()
    app.session_state["evaluation_response"] = SimpleNamespace(
        merchant_canonical_name="Trader Joe's",
        mcc_code="5411",
        primary_category="groceries",
        recommended_card=None,
        alternative_cards=[],
        assistant_message=None,
        tree_assessment=None,
        requires_approval=False,
        human_reviewed=False,
        audit_trail=[],
        reasoning_tree=[],
    )

    app.run()

    assert not app.exception
    assert any("No eligible card recommendation" in item.value for item in app.error)


def test_streamlit_allows_over_limit_amount_but_guardrail_blocks_recommendation():
    app = AppTest.from_file(str(WEB_APP_PATH)).run()

    app.number_input[0].set_value(50_000.01)
    app.button[0].click().run()

    assert not app.exception
    response = app.session_state["evaluation_response"]
    assert response.blocked_reason is not None
    assert response.recommended_card is None
    warning = next(item.value for item in app.warning)
    assert "above the 50,000 USD limit" in warning
    assert "Lower it to get a recommendation." in warning
    assert "$" not in warning
    assert any("Transaction blocked" in item.value for item in app.markdown)
    assert not any("Recommended card" in item.value for item in app.markdown)


def test_streamlit_refreshes_cached_blocked_message_from_transaction():
    app = AppTest.from_file(str(WEB_APP_PATH)).run()
    request = TransactionEvaluationRequest(
        merchant_name="Trader Joe's",
        amount=56_446.0,
    )
    app.session_state["transaction_request"] = request
    app.session_state["evaluation_response"] = TransactionEvaluationResponse(
        merchant_canonical_name="Trader Joe's",
        mcc_code="N/A",
        primary_category="unverified",
        recommended_card=None,
        alternative_cards=[],
        blocked_reason=(
            "Recommendations are unavailable for purchases over $50,000. "
            "Enter an amount at or below $50,000 to continue."
        ),
    )

    app.run()

    assert not app.exception
    warning = next(item.value for item in app.warning)
    assert "above the 50,000 USD limit" in warning
    assert "unavailable for purchases" not in warning


def test_streamlit_app_can_record_human_review():
    app = AppTest.from_file(str(WEB_APP_PATH)).run()
    app.session_state["transaction_request"] = TransactionEvaluationRequest(
        merchant_name="Trader Joe's",
        amount=3000.0,
    )
    app.session_state["evaluation_response"] = TransactionEvaluationResponse(
        merchant_canonical_name="Trader Joe's",
        mcc_code="5411",
        primary_category="groceries",
        recommended_card=CardRecommendation(
            card_id="amex_gold",
            card_name="American Express Gold Card",
            expected_yield_cents=680.0,
            net_value_cents=680.0,
            rationale="Earns 4x on groceries.",
            has_purchase_protection=True,
            has_extended_warranty=False,
            foreign_transaction_fee=0.0,
        ),
        alternative_cards=[],
        requires_approval=True,
        approval_reason="High-value purchase requires review.",
    )
    app.run()

    assert not app.exception
    assert any(item.label == "Confirmed merchant" for item in app.text_input)
    review_checkbox = next(
        item
        for item in app.checkbox
        if item.label.startswith("I reviewed these details")
    )
    review_checkbox.check().run()
    submit_button = next(item for item in app.button if item.label == "Submit review")
    submit_button.click().run()

    assert not app.exception
    assert app.session_state["evaluation_response"].human_reviewed
    assert any(
        "no payment was executed" in item
        for item in app.session_state["evaluation_response"].audit_trail
    )
