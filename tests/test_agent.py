import pytest
from src.agent import SmartWalletAgent
from src.models import TransactionEvaluationRequest
from src.tools.location import resolve_location, LocationInput
from src.tools.merchant import lookup_merchant, MerchantLookupInput
from src.tools.wallet import get_user_wallet

def test_location_resolver():
    ctx = resolve_location(LocationInput(venue_override="Trader Joe's"))
    assert ctx.venue_name == "Trader Joe's"
    assert ctx.inferred_category == "groceries"
    assert ctx.is_ambiguous is False

def test_merchant_lookup():
    details = lookup_merchant(MerchantLookupInput(raw_merchant_name="Walmart Supercenter"))
    assert details.canonical_name == "Walmart"
    assert details.qualifies_for_supermarket_multipliers is False

def test_wallet_retrieval():
    wallet = get_user_wallet("user_alex_01")
    assert wallet.user_id == "user_alex_01"
    assert len(wallet.cards) >= 3

def test_agent_evaluation_groceries():
    agent = SmartWalletAgent()
    req = TransactionEvaluationRequest(merchant_name="Trader Joe's", amount=100.0)
    res = agent.evaluate_transaction(req)
    # Amex Gold gives 4x on groceries, valued at 1.7¢/pt = 680¢
    assert res.recommended_card.card_id == "amex_gold"
    assert res.recommended_card.expected_yield_cents == 680.0

def test_agent_evaluation_travel():
    agent = SmartWalletAgent()
    req = TransactionEvaluationRequest(merchant_name="Delta Air Lines", amount=500.0)
    res = agent.evaluate_transaction(req)
    # Chase Sapphire Reserve gives 3x on travel, valued at 1.8¢/pt = 2700¢
    assert res.recommended_card.card_id == "chase_sapphire_reserve"
    assert res.recommended_card.expected_yield_cents == 2700.0
