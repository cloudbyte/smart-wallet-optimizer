from src.models import (
    LocationInput,
    TransactionEvaluationRequest,
    TransactionEvaluationResponse,
)
from src.rag.retriever import SmartWalletRAG
from src.tools.location import resolve_location
from src.tools.merchant import MerchantLookupInput, lookup_merchant
from src.tools.wallet import get_user_wallet
from src.tot import evaluate_wallet_tree


class SmartWalletAgent:
    def __init__(self, user_id: str = "user_alex_01"):
        self.user_id = user_id
        self.rag = SmartWalletRAG()

    def evaluate_transaction(
        self, request: TransactionEvaluationRequest
    ) -> TransactionEvaluationResponse:
        audit = []

        # 1. Resolve Location & Merchant Context
        loc_ctx = resolve_location(LocationInput(venue_override=request.merchant_name))
        audit.append(
            f"Resolved location context: {loc_ctx.venue_name} "
            f"(Category: {loc_ctx.inferred_category}, "
            f"Confidence: {loc_ctx.confidence_score})"
        )

        merch_details = lookup_merchant(
            MerchantLookupInput(raw_merchant_name=request.merchant_name)
        )
        audit.append(
            f"Resolved MCC details: {merch_details.canonical_name} "
            f"(MCC: {merch_details.mcc_code}, "
            f"Category: {merch_details.primary_category})"
        )

        # 2. Retrieve wallet and evaluate card branches breadth-first.
        wallet = get_user_wallet(self.user_id)
        active_cards = [
            card for card in wallet.cards if not card.is_frozen and not card.is_expired
        ]
        audit.append(
            f"Retrieved wallet for user {wallet.user_id} with "
            f"{len(active_cards)} eligible cards."
        )

        effective_category = request.category_hint or merch_details.primary_category
        recommendations, reasoning_tree, tree_audit = evaluate_wallet_tree(
            request=request,
            wallet=wallet,
            category=effective_category,
            qualifies_for_supermarket_multipliers=(
                merch_details.qualifies_for_supermarket_multipliers
            ),
            query_benefits=self.rag.query_benefits,
        )
        audit.extend(tree_audit)
        audit.append(
            "ToT evaluation complete: "
            f"{len(reasoning_tree)} reasoning nodes across "
            f"{len({node.depth for node in reasoning_tree})} depth(s)."
        )

        # 3. Guardrails & HITL checks
        requires_approval = False
        approval_reason = None
        if request.amount > 2500.0:
            requires_approval = True
            approval_reason = (
                f"Transaction amount (${request.amount}) exceeds "
                "high-value threshold ($2,500)."
            )
            audit.append(f"HITL Triggered: {approval_reason}")
        elif loc_ctx.confidence_score < 0.70 or loc_ctx.is_ambiguous:
            requires_approval = True
            approval_reason = (
                "Merchant category confidence is low or ambiguous "
                f"({loc_ctx.confidence_score})."
            )
            audit.append(f"HITL Triggered: {approval_reason}")

        if not recommendations:
            requires_approval = True
            approval_reason = (
                "No eligible wallet card remains after ToT constraint evaluation."
            )
            audit.append(f"HITL Triggered: {approval_reason}")

        return TransactionEvaluationResponse(
            merchant_canonical_name=merch_details.canonical_name,
            mcc_code=merch_details.mcc_code,
            primary_category=effective_category,
            recommended_card=recommendations[0] if recommendations else None,
            alternative_cards=recommendations[1:],
            requires_approval=requires_approval,
            approval_reason=approval_reason,
            audit_trail=audit,
            reasoning_tree=reasoning_tree,
        )
