import json
import logging
from typing import Dict, List, Optional

from src.models import (
    BenefitsRiskAssessment,
    ContextRewardsAssessment,
    LocationInput,
    MerchantDetails,
    TransactionEvaluationRequest,
    TransactionEvaluationResponse,
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
    calculate_reward_yields,
    evaluate_wallet_tree,
)
from src.agent.state import AgentLanguageModel, AgentState

LOGGER = logging.getLogger(__name__)


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
            "audit_trail": ["Started coordinated multi-agent evaluation."],
        }

    def _resolve_location(self, state: AgentState) -> AgentState:
        request = state["request"]
        location = resolve_location(
            LocationInput(venue_override=request.merchant_name)
        )
        return {
            "location_context": location,
            "location_audit": (
                f"Location resolved: {location.venue_name}; "
                f"category={location.inferred_category}; "
                f"confidence={location.confidence_score}; "
                f"ambiguous={location.is_ambiguous}."
            ),
        }

    def _lookup_merchant(self, state: AgentState) -> AgentState:
        request = state["request"]
        resolver = getattr(self.language_model, "resolve_merchant", None)
        merchant = lookup_merchant(
            MerchantLookupInput(raw_merchant_name=request.merchant_name),
            resolver=resolver if callable(resolver) else None,
        )
        return {
            "merchant": merchant,
            "merchant_audit": (
                f"Merchant resolved: {merchant.canonical_name}; "
                f"MCC={merchant.mcc_code}; category={merchant.primary_category}; "
                f"resolution={merchant.resolution_method}; "
                f"confidence={merchant.match_confidence}; "
                f"confirmation_required={merchant.requires_confirmation}; "
                f"notes={merchant.notes or 'none'}."
            ),
        }

    def _load_wallet(self, _state: AgentState) -> AgentState:
        wallet = get_user_wallet(self.user_id)
        active_count = sum(
            not card.is_frozen and not card.is_expired for card in wallet.cards
        )
        return {
            "wallet": wallet,
            "wallet_audit": (
                f"Loaded synthetic wallet for {wallet.user_id} with "
                f"{active_count} eligible cards."
            ),
        }

    def _context_rewards_specialist(self, state: AgentState) -> AgentState:
        request = state["request"]
        merchant = state["merchant"]
        wallet = state["wallet"]
        reward_yields = calculate_reward_yields(
            request=request,
            wallet=wallet,
            category=merchant.primary_category,
            qualifies_for_supermarket_multipliers=(
                merchant.qualifies_for_supermarket_multipliers
            ),
        )
        audit = [
            *state["audit_trail"],
            state["location_audit"],
            state["merchant_audit"],
            state["wallet_audit"],
            (
                "Agent 1 (Context & Rewards Specialist) identified "
                f"{len(reward_yields)} candidate cards and calculated reward yields."
            ),
        ]
        candidate_card_ids = list(reward_yields)
        return {
            "context_assessment": ContextRewardsAssessment(
                location=state["location_context"],
                merchant=merchant,
                wallet=wallet,
                candidate_card_ids=candidate_card_ids,
                reward_yields_cents=reward_yields,
            ),
            "candidate_cards": candidate_card_ids,
            "audit_trail": audit,
        }

    def _benefits_risk_specialist(self, state: AgentState) -> AgentState:
        request = state["request"]
        context = state["context_assessment"]
        category = context.merchant.primary_category
        audit = state["audit_trail"]
        early_exit = (
            request.amount < LOW_VALUE_THRESHOLD_USD
            and not request.is_international
            and category not in HIGH_RISK_CATEGORIES
        )
        evidence_by_card: Dict[str, str] = {}
        retrieval_errors: Dict[str, str] = {}
        if early_exit:
            audit.append(
                "Agent 2 (Benefits & Risk Specialist) early exit: skipped "
                "policy retrieval "
                "for a low-value, local, non-high-risk transaction."
            )
        else:
            for card_id in context.candidate_card_ids:
                query = f"{category} purchase protection and extended warranty"
                try:
                    retrieved = self.rag.query_benefits(card_id, query)
                except (OSError, RuntimeError, ValueError) as error:
                    retrieval_errors[card_id] = type(error).__name__
                    LOGGER.warning(
                        "Benefit retrieval failed for card %s (%s).",
                        card_id,
                        type(error).__name__,
                    )
                    audit.append(
                        f"Policy retrieval failed for {card_id} "
                        f"({type(error).__name__}); benefit claims are withheld."
                    )
                else:
                    evidence_by_card[card_id] = retrieved
                    audit.append(
                        f"Policy retrieval for {card_id}: "
                        f"{retrieved or 'no matching benefit evidence found'}."
                    )
            audit.append(
                "Agent 2 (Benefits & Risk Specialist) returned retrieved evidence "
                "and wallet fee data."
            )
        card_by_id = {card.card_id: card for card in context.wallet.cards}
        return {
            "benefits_assessment": BenefitsRiskAssessment(
                evidence_by_card=evidence_by_card,
                foreign_transaction_fees={
                    card_id: card_by_id[card_id].foreign_transaction_fee
                    for card_id in context.candidate_card_ids
                },
                retrieval_errors=retrieval_errors,
            ),
            "audit_trail": audit,
        }

    def _decision_orchestrator(self, state: AgentState) -> AgentState:
        request = state["request"]
        context = state["context_assessment"]
        benefits = state["benefits_assessment"]
        location = context.location
        merchant = context.merchant
        wallet = context.wallet
        audit = state["audit_trail"]
        category = merchant.primary_category
        recommendations, reasoning_tree, tree_audit = evaluate_wallet_tree(
            request=request,
            wallet=wallet,
            category=category,
            qualifies_for_supermarket_multipliers=(
                merchant.qualifies_for_supermarket_multipliers
            ),
            query_benefits=lambda card_id, _query: benefits.evidence_by_card.get(
                card_id, ""
            ),
            reward_yields_cents=context.reward_yields_cents,
            foreign_transaction_fees=benefits.foreign_transaction_fees,
        )
        audit.extend(tree_audit)
        audit.append(
            "Tree-of-Thought card evaluation complete: "
            f"{len(reasoning_tree)} nodes across "
            f"{len({node.depth for node in reasoning_tree})} depth(s)."
        )
        audit.append(
            "Agent 3 (Decision Orchestrator) applied Tree-of-Thought evaluation "
            "and hard constraints."
        )

        approval_reason_text = approval_reason(
            request=request,
            location=location,
            has_recommendations=bool(recommendations),
            merchant=merchant,
            requested_category=request.category_hint,
            benefit_retrieval_failed=bool(benefits.retrieval_errors),
        )
        if approval_reason_text:
            audit.append(f"HITL Triggered: {approval_reason_text}")

        recommended_card = recommendations[0] if recommendations else None
        result = TransactionEvaluationResponse(
            merchant_canonical_name=merchant.canonical_name,
            mcc_code=merchant.mcc_code,
            primary_category=category,
            recommended_card=recommended_card,
            alternative_cards=recommendations[1:],
            requires_approval=approval_reason_text is not None,
            approval_reason=approval_reason_text,
            audit_trail=audit,
            reasoning_tree=reasoning_tree,
        )
        return {
            "recommended_card": recommended_card,
            "reasoning_tree": reasoning_tree,
            "response": result,
        }

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
        try:
            generated = critique_tree(json.dumps(facts, indent=2))
        except (OSError, RuntimeError, ValueError) as error:
            LOGGER.warning(
                "Tree critique failed (%s); omitting optional critique.",
                type(error).__name__,
            )
            response.audit_trail.append(
                "ToT critic failed; omitted the optional critique "
                f"({type(error).__name__})."
            )
            return {"response": response}
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
        return {
            "tree_assessment": response.tree_assessment,
            "response": response,
        }

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
        try:
            generated = self.language_model.explain(json.dumps(facts, indent=2))
        except (OSError, RuntimeError, ValueError) as error:
            LOGGER.warning(
                "Recommendation explanation failed (%s); using factual fallback.",
                type(error).__name__,
            )
            generated = ""
            response.audit_trail.append(
                "Model explanation failed; used a factual fallback "
                f"({type(error).__name__})."
            )
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
        return {
            "assistant_message": response.assistant_message,
            "response": response,
        }
