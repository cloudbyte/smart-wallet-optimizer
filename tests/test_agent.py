import pytest
from src.agent import SmartWalletAgent
from src.models import TransactionEvaluationRequest, UserWallet
from src.tools.location import resolve_location, LocationInput
from src.tools.merchant import lookup_merchant, MerchantLookupInput
from src.tools.wallet import get_user_wallet

def test_location_resolver():
    ctx = resolve_location(LocationInput(venue_override="Trader Joe's"))
    assert ctx.venue_name == "Trader Joe's"
    assert ctx.inferred_category == "groceries"
    assert ctx.is_ambiguous is False

def test_location_resolver_lat_lon():
    # Lat/Lon for Starbucks or arbitrary venue via OpenStreetMap reverse geocoding
    ctx = resolve_location(LocationInput(latitude=37.7749, longitude=-122.4194))
    assert ctx.venue_name is not None
    assert ctx.confidence_score > 0.0

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

def test_agent_evaluates_tree_breadth_first_and_prunes_fee():
    agent = SmartWalletAgent()
    req = TransactionEvaluationRequest(
        merchant_name="Delta Air Lines",
        amount=100.0,
        is_international=True,
    )

    res = agent.evaluate_transaction(req)

    assert res.recommended_card is not None
    assert res.recommended_card.card_id == "chase_sapphire_reserve"
    assert [node.depth for node in res.reasoning_tree] == sorted(
        node.depth for node in res.reasoning_tree
    )
    citi_fee_node = next(
        node
        for node in res.reasoning_tree
        if node.depth == 3 and node.card_id == "citi_double_cash"
    )
    assert citi_fee_node.status == "pruned"
    assert any("foreign transaction fee" in item for item in res.audit_trail)

def test_agent_uses_early_exit_for_small_local_transaction(monkeypatch):
    agent = SmartWalletAgent()
    monkeypatch.setattr(
        agent.rag,
        "query_benefits",
        lambda *_: (_ for _ in ()).throw(AssertionError("RAG should be bypassed")),
    )
    req = TransactionEvaluationRequest(merchant_name="Trader Joe's", amount=10.0)

    res = agent.evaluate_transaction(req)

    assert res.recommended_card is not None
    assert {node.depth for node in res.reasoning_tree} == {1}
    assert any("early exit" in item for item in res.audit_trail)

def test_agent_includes_mock_perk_estimate_in_tree():
    agent = SmartWalletAgent()
    req = TransactionEvaluationRequest(merchant_name="Trader Joe's", amount=100.0)

    res = agent.evaluate_transaction(req)

    amex_perk_node = next(
        node
        for node in res.reasoning_tree
        if node.depth == 2 and node.card_id == "amex_gold"
    )
    assert amex_perk_node.perk_value_cents == 200.0

def test_agent_requests_approval_when_all_branches_are_pruned(monkeypatch):
    wallet = get_user_wallet("user_alex_01")
    citi_only_wallet = UserWallet(
        user_id=wallet.user_id,
        point_valuation_cents=wallet.point_valuation_cents,
        cards=[wallet.cards[2]],
    )
    monkeypatch.setattr(
        "src.agent.get_user_wallet", lambda _user_id: citi_only_wallet
    )
    agent = SmartWalletAgent()
    req = TransactionEvaluationRequest(
        merchant_name="Delta Air Lines",
        amount=1.0,
        is_international=True,
    )

    res = agent.evaluate_transaction(req)

    assert res.recommended_card is None
    assert res.requires_approval is True
    assert "No eligible wallet card" in res.approval_reason
