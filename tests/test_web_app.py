from streamlit.testing.v1 import AppTest

from src.models import CardRecommendation, TransactionEvaluationResponse


def test_streamlit_app_loads_form_without_loading_model():
    app = AppTest.from_file("src/web_app.py").run()

    assert not app.exception
    assert app.title[0].value == "Smart Wallet Rewards Optimizer"
    assert any("Demo using synthetic wallet data." in item.value for item in app.caption)
    assert app.text_input[0].label == "Merchant"
    assert app.button[0].label == "Recommend a card"


def test_streamlit_app_renders_structured_recommendation():
    app = AppTest.from_file("src/web_app.py").run()
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
