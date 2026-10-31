import json
from typing import Dict, List, Optional

from src.models import (
    LocationContext,
    LocationInput,
    MerchantDetails,
    ThoughtNode,
    TransactionEvaluationRequest,
    TransactionEvaluationResponse,
    UserWallet,
)
from src.rag.retriever import SmartWalletRAG
from src.safety.guardrails import (
    approval_reason,
    is_grounded_critique,
    is_grounded_summary,
)
from src.tools.location import resolve_location
from src.tools.merchant import MerchantLookupInput, lookup_merchant
from src.tools.wallet import get_user_wallet
from src.reasoning.tot import (
    HIGH_RISK_CATEGORIES,
    LOW_VALUE_THRESHOLD_USD,
    evaluate_wallet_tree,
)
from src.agent.state import AgentLanguageModel, AgentState


class TransactionAgentNodes:
    def __init__(
        self,
        user_id: str = "user_alex_01",
        explainer: Optional[AgentLanguageModel] = None,
        rag: Optional[SmartWalletRAG] = None,
    ):
        self.user_id = user_id
        self.language_model = explainer
        embedder = getattr(explainer, "embed", None)
        self.rag = rag or SmartWalletRAG(
            embedder=embedder if callable(embedder) else None
        )

    def _initialize(self, state: AgentState) -> AgentState:
        return {
            "phase": "context",
            "pending_actions": [
                "resolve_location",
                "lookup_merchant",
                "get_wallet",
            ],
            "benefit_evidence": {},
            "audit_trail": [
                "Started ReAct agent loop; the language model selects among "
                "the available data tools."
            ],
        }

    def _plan_action(self, state: AgentState) -> AgentState:
        available = state["pending_actions"]
        if not available:
            raise RuntimeError("ReAct planner was called with no available actions.")

        selected = None
        choose_tool = getattr(self.language_model, "choose_tool", None)
        if callable(choose_tool):
            context = {
                "request": state["request"].model_dump(),
                "phase": state["phase"],
                "location": (
                    state["location_context"].model_dump()
                    if "location_context" in state
                    else None
                ),
                "merchant": (
                    state["merchant"].model_dump() if "merchant" in state else None
                ),
                "wallet_cards": (
                    [
                        {
                            "card_id": card.card_id,
                            "card_name": card.card_name,
                            "is_frozen": card.is_frozen,
                            "is_expired": card.is_expired,
                        }
                        for card in state["wallet"].cards
                    ]
                    if "wallet" in state
                    else None
                ),
                "retrieved_benefit_evidence": state.get("benefit_evidence", {}),
                "completed_actions": [
                    action
                    for action in [
                        "resolve_location",
                        "lookup_merchant",
                        "get_wallet",
                    ]
                    if action not in available
                ],
            }
            selected = choose_tool(json.dumps(context), list(available))

        audit = state["audit_trail"]
        if selected not in available:
            selected = available[0]
            if callable(choose_tool):
                audit.append(
                    "ReAct planner returned an unavailable action; executing the "
                    "next required tool instead."
                )
        else:
            audit.append(f"ReAct planner selected tool: {selected}.")
        return {"next_action": selected, "audit_trail": audit}

    def _execute_tool(self, state: AgentState) -> AgentState:
        action = state["next_action"]
        request = state["request"]
        update: AgentState = {}
        audit = state["audit_trail"]
        evidence = state["benefit_evidence"]

        if action == "resolve_location":
            location = resolve_location(
                LocationInput(venue_override=request.merchant_name)
            )
            update["location_context"] = location
            audit.append(
                f"Tool resolve_location observation: {location.venue_name}; "
                f"category={location.inferred_category}; "
                f"confidence={location.confidence_score}; "
                f"ambiguous={location.is_ambiguous}."
            )
        elif action == "lookup_merchant":
            resolver = getattr(self.language_model, "resolve_merchant", None)
            merchant = lookup_merchant(
                MerchantLookupInput(raw_merchant_name=request.merchant_name),
                resolver=resolver if callable(resolver) else None,
            )
            update["merchant"] = merchant
            audit.append(
                f"Tool lookup_merchant observation: {merchant.canonical_name}; "
                f"MCC={merchant.mcc_code}; category={merchant.primary_category}; "
                f"resolution={merchant.resolution_method}; "
                f"confidence={merchant.match_confidence}; "
                f"confirmation_required={merchant.requires_confirmation}; "
                f"notes={merchant.notes or 'none'}."
            )
        elif action == "get_wallet":
            wallet = get_user_wallet(self.user_id)
            update["wallet"] = wallet
            active_count = sum(
                not card.is_frozen and not card.is_expired for card in wallet.cards
            )
            audit.append(
                f"Tool get_wallet observation: {active_count} eligible cards "
                f"for user {wallet.user_id}."
            )
        elif action.startswith("query_benefits:"):
            card_id = action.partition(":")[2]
            wallet = state.get("wallet")
            if wallet is None or card_id not in {card.card_id for card in wallet.cards}:
                raise ValueError(f"ReAct planner selected unknown card: {card_id}")
            category = state["merchant"].primary_category
            query = (
                f"{category} purchase protection and extended warranty"
            )
            retrieved = self.rag.query_benefits(card_id, query)
            evidence[card_id] = retrieved
            audit.append(
                f"Tool query_benefits observation for {card_id}: "
                f"{retrieved or 'no matching benefit evidence found'}."
            )
            update["benefit_evidence"] = evidence
        else:
            raise ValueError(f"Unsupported ReAct tool action: {action}")

        pending = [item for item in state["pending_actions"] if item != action]
        if len(pending) == len(state["pending_actions"]):
            raise RuntimeError(f"ReAct action was not pending: {action}")
        update["pending_actions"] = pending
        update["audit_trail"] = audit
        return update

    def _after_tool(self, state: AgentState) -> str:
        if state["pending_actions"]:
            return "plan_action"
        if state["phase"] == "context":
            return "prepare_benefit_queries"
        return "evaluate_cards"

    def _prepare_benefit_queries(self, state: AgentState) -> AgentState:
        request = state["request"]
        category = state["merchant"].primary_category
        early_exit = (
            request.amount < LOW_VALUE_THRESHOLD_USD
            and not request.is_international
            and category not in HIGH_RISK_CATEGORIES
        )
        if early_exit:
            actions = []
            state["audit_trail"].append(
                "ReAct skipped benefit retrieval because this is a low-value, "
                "local, non-high-risk transaction."
            )
        else:
            actions = [
                f"query_benefits:{card.card_id}"
                for card in state["wallet"].cards
                if not card.is_frozen and not card.is_expired
            ]
        state["phase"] = "benefits"
        state["pending_actions"] = actions
        return state

    def _after_benefit_planning(self, state: AgentState) -> str:
        return "plan_action" if state["pending_actions"] else "evaluate_cards"

    def _evaluate_cards(self, state: AgentState) -> AgentState:
        request = state["request"]
        location = state["location_context"]
        merchant = state["merchant"]
        wallet = state["wallet"]
        audit = state["audit_trail"]
        category = merchant.primary_category
        recommendations, reasoning_tree, tree_audit = evaluate_wallet_tree(
            request=request,
            wallet=wallet,
            category=category,
            qualifies_for_supermarket_multipliers=(
                merchant.qualifies_for_supermarket_multipliers
            ),
            query_benefits=lambda card_id, _query: state["benefit_evidence"].get(
                card_id, ""
            ),
        )
        audit.extend(tree_audit)
        audit.append(
            "Tree-of-Thought card evaluation complete: "
            f"{len(reasoning_tree)} nodes across "
            f"{len({node.depth for node in reasoning_tree})} depth(s)."
        )

        approval_reason_text = approval_reason(
            request=request,
            location=location,
            has_recommendations=bool(recommendations),
            merchant=merchant,
            requested_category=request.category_hint,
        )
        if approval_reason_text:
            audit.append(f"HITL Triggered: {approval_reason_text}")

        result = TransactionEvaluationResponse(
            merchant_canonical_name=merchant.canonical_name,
            mcc_code=merchant.mcc_code,
            primary_category=category,
            recommended_card=recommendations[0] if recommendations else None,
            alternative_cards=recommendations[1:],
            requires_approval=approval_reason_text is not None,
            approval_reason=approval_reason_text,
            audit_trail=audit,
            reasoning_tree=reasoning_tree,
        )
        return {"response": result}

    def _critique_tree(self, state: AgentState) -> AgentState:
        critique_tree = getattr(self.language_model, "critique_tree", None)
        if not callable(critique_tree):
            return {}

        response = state["response"]
        facts = {
            "request": state["request"].model_dump(),
            "recommendations": [
                card.model_dump()
                for card in [
                    response.recommended_card,
                    *response.alternative_cards,
                ]
                if card is not None
            ],
            "reasoning_tree": [
                node.model_dump() for node in response.reasoning_tree
            ],
        }
        generated = critique_tree(json.dumps(facts, indent=2))
        if is_grounded_critique(generated, response):
            response.tree_assessment = generated
            response.audit_trail.append(
                f"ToT critic produced an evidence-grounded assessment with "
                f"{type(self.language_model).__name__}."
            )
        else:
            response.audit_trail.append(
                "ToT critic output was omitted because it could not be verified "
                "against the retrieved evidence."
            )
        return {"response": response}

    def _generate_explanation(self, state: AgentState) -> AgentState:
        if self.language_model is None:
            return {}

        response = state["response"]
        facts = {
            "transaction": {
                "merchant": response.merchant_canonical_name,
                "category": response.primary_category,
                "amount_usd": state["request"].amount,
                "international": state["request"].is_international,
            },
            "decision": {
                "recommended_card": (
                    response.recommended_card.model_dump()
                    if response.recommended_card
                    else None
                ),
                "alternatives": [
                    card.model_dump() for card in response.alternative_cards
                ],
                "requires_approval": response.requires_approval,
                "approval_reason": response.approval_reason,
                "tree_assessment": response.tree_assessment,
            },
        }
        generated = self.language_model.explain(json.dumps(facts, indent=2))
        if is_grounded_summary(generated, response):
            response.assistant_message = generated
            response.audit_trail.append(
                f"Generated recommendation explanation with "
                f"{type(self.language_model).__name__}."
            )
        else:
            card = response.recommended_card
            response.assistant_message = (
                f"Based on the available information, {card.card_name} is the "
                f"recommended card for {response.merchant_canonical_name} in the "
                f"{response.primary_category} category."
                if card
                else (
                    "No eligible card recommendation is available for this "
                    "transaction."
                )
            )
            response.audit_trail.append(
                "Replaced the model explanation with a verified summary because "
                "it included unsupported details."
            )
        return {"response": response}
