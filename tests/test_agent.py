import json

import pytest
import requests
from src.agent import SmartWalletAgent
from src import main as cli
from src.models import LocationContext, TransactionEvaluationRequest, UserWallet
from src.tools.location import resolve_location, LocationInput
from src.tools.merchant import lookup_merchant, MerchantLookupInput
from src.tools.wallet import get_user_wallet


class FixtureBenefits:
    def query_benefits(self, card_id, _query):
        if card_id == "amex_gold":
            return (
                "[Source: test fixture, page 1] Purchase Protection covers "
                "eligible purchases against accidental damage or theft."
            )
        if card_id == "chase_sapphire_reserve":
            return (
                "[Source: test fixture, page 1] Purchase Protection covers "
                "eligible purchases. Extended Warranty Protection adds an "
                "additional year to the eligible manufacturer's warranty."
            )
        return ""


def make_agent(**kwargs):
    return SmartWalletAgent(rag=FixtureBenefits(), **kwargs)


def test_location_resolver():
    ctx = resolve_location(LocationInput(venue_override="Trader Joe's"))
    assert ctx.venue_name == "Trader Joe's"
    assert ctx.inferred_category == "groceries"
    assert ctx.is_ambiguous is False

def test_location_resolver_lat_lon(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"type": "cafe", "name": "Sample Cafe"}

    monkeypatch.setattr(
        "src.tools.location.requests.get", lambda *_args, **_kwargs: FakeResponse()
    )
    ctx = resolve_location(LocationInput(latitude=37.7749, longitude=-122.4194))
    assert ctx.venue_name == "Sample Cafe"
    assert ctx.inferred_category == "dining"
    assert ctx.confidence_score == 0.90


def test_location_failure_returns_ambiguous_context_and_logs(monkeypatch, caplog):
    def fail_request(*_args, **_kwargs):
        raise requests.ConnectionError("network unavailable")

    monkeypatch.setattr("src.tools.location.requests.get", fail_request)

    context = resolve_location(LocationInput(venue_override="Unknown Cafe"))

    assert context.is_ambiguous
    assert context.confidence_score == 0.0
    assert context.inferred_category == "miscellaneous"
    assert "Location lookup failed" in caplog.text


def test_location_input_rejects_out_of_range_coordinates():
    with pytest.raises(ValueError):
        LocationInput(latitude=91.0, longitude=0.0)


def test_merchant_lookup():
    details = lookup_merchant(MerchantLookupInput(raw_merchant_name="Walmart Supercenter"))
    assert details.canonical_name == "Walmart"
    assert details.qualifies_for_supermarket_multipliers is False


def test_unknown_short_merchant_query_does_not_match_arbitrary_name():
    details = lookup_merchant(MerchantLookupInput(raw_merchant_name="art"))

    assert details.primary_category == "miscellaneous"
    assert details.mcc_code == "5999"

def test_wallet_retrieval():
    wallet = get_user_wallet("user_alex_01")
    assert wallet.user_id == "user_alex_01"
    assert len(wallet.cards) >= 3


def test_wallet_retrieval_rejects_unconfigured_user_id():
    with pytest.raises(ValueError, match="No synthetic wallet is configured"):
        get_user_wallet("another-user")


def test_transaction_request_enforces_documented_amount_limit():
    with pytest.raises(ValueError):
        TransactionEvaluationRequest(merchant_name="Trader Joe's", amount=50_000.01)


def test_transaction_request_rejects_empty_merchant_and_non_finite_amounts():
    with pytest.raises(ValueError, match="Merchant name cannot be empty"):
        TransactionEvaluationRequest(merchant_name="  ", amount=10.0)
    with pytest.raises(ValueError):
        TransactionEvaluationRequest(merchant_name="Trader Joe's", amount=float("nan"))


def test_cli_rejects_invalid_amount_without_starting_agent(monkeypatch, capsys):
    inputs = iter(["Trader Joe's", "not-a-number"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr(
        cli,
        "SmartWalletAgent",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("agent must not start for invalid input")
        ),
    )

    cli.main()

    assert "Invalid amount" in capsys.readouterr().out


def test_cli_rejects_amount_over_limit_without_starting_agent(monkeypatch, capsys):
    inputs = iter(["Trader Joe's", "50001", "n"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr(
        cli,
        "SmartWalletAgent",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("agent must not start for invalid input")
        ),
    )

    cli.main()

    assert "Invalid transaction" in capsys.readouterr().out

def test_agent_evaluation_groceries():
    agent = make_agent()
    req = TransactionEvaluationRequest(merchant_name="Trader Joe's", amount=100.0)
    res = agent.evaluate_transaction(req)
    # Amex Gold gives 4x on groceries, valued at 1.7¢/pt = 680¢
    assert res.recommended_card.card_id == "amex_gold"
    assert res.recommended_card.expected_yield_cents == 680.0

def test_agent_evaluation_travel():
    agent = make_agent()
    req = TransactionEvaluationRequest(merchant_name="Delta Air Lines", amount=500.0)
    res = agent.evaluate_transaction(req)
    # Chase Sapphire Reserve gives 3x on travel, valued at 1.8¢/pt = 2700¢
    assert res.recommended_card.card_id == "chase_sapphire_reserve"
    assert res.recommended_card.expected_yield_cents == 2700.0


def test_unknown_merchant_requires_approval_even_with_confident_location(monkeypatch):
    monkeypatch.setattr(
        "src.agent.nodes.resolve_location",
        lambda _input: LocationContext(
            venue_name="Unknown Merchant",
            inferred_category="dining",
            is_ambiguous=False,
            confidence_score=0.95,
        ),
    )

    response = make_agent().evaluate_transaction(
        TransactionEvaluationRequest(merchant_name="Unknown Merchant", amount=100.0)
    )

    assert response.requires_approval
    assert "could not be verified" in response.approval_reason


def test_agent_uses_explainer_with_structured_decision_facts():
    class StubExplainer:
        def __init__(self):
            self.facts = None

        def explain(self, facts):
            self.facts = json.loads(facts)
            return (
                "The American Express Gold Card is the best match "
                "for this grocery purchase."
            )

    explainer = StubExplainer()
    agent = make_agent(explainer=explainer)
    req = TransactionEvaluationRequest(merchant_name="Trader Joe's", amount=100.0)

    res = agent.evaluate_transaction(req)

    assert res.assistant_message == (
        "The American Express Gold Card is the best match "
        "for this grocery purchase."
    )
    assert explainer.facts["decision"]["recommended_card"]["card_id"] == "amex_gold"
    assert explainer.facts["transaction"]["amount_usd"] == 100.0
    assert any("Generated recommendation explanation" in item for item in res.audit_trail)


def test_agent_replaces_unsupported_model_explanation():
    class UnsupportedExplainer:
        def explain(self, _facts):
            return (
                "The American Express Gold Card earns 680 cents "
                "and requires no approval."
            )

    agent = make_agent(explainer=UnsupportedExplainer())
    req = TransactionEvaluationRequest(merchant_name="Trader Joe's", amount=100.0)

    res = agent.evaluate_transaction(req)

    assert res.assistant_message == (
        "The deterministic evaluator recommends American Express Gold Card for "
        "Trader Joe's in the groceries category."
    )
    assert any("Replaced the model explanation" in item for item in res.audit_trail)


def test_agent_uses_react_tool_selection_and_grounded_tot_critic():
    class ReActModel:
        def __init__(self):
            self.actions = []
            self.selected_actions = []

        def choose_tool(self, context, available_actions):
            self.actions.append((json.loads(context), list(available_actions)))
            selected = available_actions[-1]
            self.selected_actions.append(selected)
            return selected

        def critique_tree(self, facts):
            return (
                "The American Express Gold Card is supported by retrieved text: "
                "Purchase Protection covers eligible purchases."
            )

        def explain(self, _facts):
            return "The American Express Gold Card matches this grocery purchase."

    model = ReActModel()
    agent = make_agent(explainer=model)
    req = TransactionEvaluationRequest(merchant_name="Trader Joe's", amount=100.0)

    res = agent.evaluate_transaction(req)

    assert model.actions
    assert all(
        selected in available
        for (_, available), selected in zip(model.actions, model.selected_actions)
    )
    assert len(model.actions) == 7
    assert any("ReAct planner selected tool:" in item for item in res.audit_trail)
    assert res.tree_assessment == (
        "The American Express Gold Card is supported by retrieved text: "
        "Purchase Protection covers eligible purchases."
    )
    assert any("ToT critic produced an evidence-grounded assessment" in item
               for item in res.audit_trail)


def test_agent_rejects_critic_evidence_from_another_card_branch():
    class CrossCardCritic:
        def choose_tool(self, _context, available_actions):
            return available_actions[0]

        def critique_tree(self, _facts):
            return (
                "The American Express Gold Card is supported by retrieved text: "
                "Extended Warranty Protection adds an additional year."
            )

        def explain(self, _facts):
            return "The American Express Gold Card matches this grocery purchase."

    res = make_agent(explainer=CrossCardCritic()).evaluate_transaction(
        TransactionEvaluationRequest(merchant_name="Trader Joe's", amount=100.0)
    )

    assert res.tree_assessment is None
    assert any("could not be verified" in item for item in res.audit_trail)


def test_agent_evaluates_tree_breadth_first_and_prunes_fee():
    agent = make_agent()
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
    assert citi_fee_node.status == "evaluated"
    assert citi_fee_node.penalty_fee_cents == 300.0
    assert citi_fee_node.net_value_cents == 20.0


def test_foreign_transaction_fee_percentage_is_converted_to_cents(monkeypatch):
    wallet = get_user_wallet("user_alex_01")
    citi_only_wallet = UserWallet(
        user_id=wallet.user_id,
        point_valuation_cents=wallet.point_valuation_cents,
        cards=[card for card in wallet.cards if card.card_id == "citi_double_cash"],
    )
    agent = make_agent()
    agent.nodes.user_id = citi_only_wallet.user_id

    monkeypatch.setattr(
        "src.agent.nodes.get_user_wallet", lambda _user_id: citi_only_wallet
    )
    response = agent.evaluate_transaction(
        TransactionEvaluationRequest(
            merchant_name="Delta Air Lines",
            amount=100.0,
            is_international=True,
        )
    )

    citi_fee_node = next(
        node
        for node in response.reasoning_tree
        if node.depth == 3 and node.card_id == "citi_double_cash"
    )
    assert citi_fee_node.penalty_fee_cents == 300.0
    assert citi_fee_node.net_value_cents == 20.0


def test_agent_uses_early_exit_for_small_local_transaction(monkeypatch):
    agent = make_agent()
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
    agent = make_agent()
    req = TransactionEvaluationRequest(merchant_name="Trader Joe's", amount=100.0)

    res = agent.evaluate_transaction(req)

    amex_perk_node = next(
        node
        for node in res.reasoning_tree
        if node.depth == 2 and node.card_id == "amex_gold"
    )
    assert amex_perk_node.perk_value_cents == 200.0


def test_rag_citation_title_alone_does_not_count_as_benefit_evidence():
    class HeaderOnlyBenefits:
        def query_benefits(self, _card_id, _query):
            return (
                "[Source: Purchase Protection Policy]\n"
                "The policy applies to eligible purchases."
            )

    agent = SmartWalletAgent(rag=HeaderOnlyBenefits())
    response = agent.evaluate_transaction(
        TransactionEvaluationRequest(merchant_name="Trader Joe's", amount=100.0)
    )
    amex_perk_node = next(
        node
        for node in response.reasoning_tree
        if node.depth == 2 and node.card_id == "amex_gold"
    )

    assert amex_perk_node.perk_value_cents == 0.0

def test_agent_requests_approval_when_all_branches_are_pruned(monkeypatch):
    wallet = get_user_wallet("user_alex_01")
    citi_only_wallet = UserWallet(
        user_id=wallet.user_id,
        point_valuation_cents=wallet.point_valuation_cents,
        cards=[
            card.model_copy(update={"foreign_transaction_fee": 4.0})
            for card in wallet.cards
            if card.card_id == "citi_double_cash"
        ],
    )
    monkeypatch.setattr(
        "src.agent.nodes.get_user_wallet", lambda _user_id: citi_only_wallet
    )
    agent = make_agent()
    req = TransactionEvaluationRequest(
        merchant_name="Delta Air Lines",
        amount=1.0,
        is_international=True,
    )

    res = agent.evaluate_transaction(req)

    assert res.recommended_card is None
    assert res.requires_approval is True
    assert "No eligible wallet card" in res.approval_reason
