from typing import List
from src.models import (
    TransactionEvaluationRequest,
    TransactionEvaluationResponse,
    CardRecommendation,
    LocationInput
)
from src.tools.location import resolve_location
from src.tools.merchant import lookup_merchant, MerchantLookupInput
from src.tools.wallet import get_user_wallet

class SmartWalletAgent:
    def __init__(self, user_id: str = "user_alex_01"):
        self.user_id = user_id

    def evaluate_transaction(self, request: TransactionEvaluationRequest) -> TransactionEvaluationResponse:
        audit = []
        
        # 1. Resolve Location & Merchant Context
        loc_ctx = resolve_location(LocationInput(venue_override=request.merchant_name))
        audit.append(f"Resolved location context: {loc_ctx.venue_name} (Category: {loc_ctx.inferred_category}, Confidence: {loc_ctx.confidence_score})")
        
        merch_details = lookup_merchant(MerchantLookupInput(raw_merchant_name=request.merchant_name))
        audit.append(f"Resolved MCC details: {merch_details.canonical_name} (MCC: {merch_details.mcc_code}, Category: {merch_details.primary_category})")
        
        # 2. Retrieve User Wallet & Valuations
        wallet = get_user_wallet(self.user_id)
        audit.append(f"Retrieved wallet for user {wallet.user_id} with {len(wallet.cards)} active cards.")
        
        effective_category = request.category_hint or merch_details.primary_category
        
        recommendations: List[CardRecommendation] = []
        
        for card in wallet.cards:
            # Check foreign transaction fee impact
            fx_fee_penalty = (request.amount * (card.foreign_transaction_fee / 100.0)) * 100.0 if request.is_international else 0.0
            
            # Determine multiplier
            multiplier = card.base_multipliers.get(effective_category, card.base_multipliers.get("default", 1.0))
            
            # Special check for supermarket exclusion
            if effective_category == "groceries" and not merch_details.qualifies_for_supermarket_multipliers:
                multiplier = card.base_multipliers.get("default", 1.0)
                audit.append(f"Card {card.card_name}: Supermarket exclusion applies for {merch_details.canonical_name}. Fallback to {multiplier}x.")
            
            # Point valuation (assume chase_ur / amex_mr / cash)
            valuation_key = "chase_ur" if "chase" in card.card_id else ("amex_mr" if "amex" in card.card_id else "cash")
            point_value_cents = wallet.point_valuation_cents.get(valuation_key, 1.0)
            
            expected_yield_cents = request.amount * multiplier * point_value_cents
            net_value_cents = expected_yield_cents - fx_fee_penalty
            
            rationale = f"Earns {multiplier}x on {effective_category} ({valuation_key} valued at {point_value_cents}¢/pt)."
            if fx_fee_penalty > 0:
                rationale += f" Incurs {card.foreign_transaction_fee}% FTF penalty (-{fx_fee_penalty:.1f}¢)."
            
            recommendations.append(CardRecommendation(
                card_id=card.card_id,
                card_name=card.card_name,
                expected_yield_cents=round(expected_yield_cents, 2),
                net_value_cents=round(net_value_cents, 2),
                rationale=rationale,
                has_purchase_protection=card.has_purchase_protection,
                has_extended_warranty=card.has_extended_warranty,
                foreign_transaction_fee=card.foreign_transaction_fee
            ))
        
        # Sort by net value descending
        recommendations.sort(key=lambda x: x.net_value_cents, reverse=True)
        
        best_card = recommendations[0]
        alt_cards = recommendations[1:]
        
        # 3. Guardrails & HITL checks
        requires_approval = False
        approval_reason = None
        if request.amount > 2500.0:
            requires_approval = True
            approval_reason = f"Transaction amount (${request.amount}) exceeds high-value threshold ($2,500)."
            audit.append(f"HITL Triggered: {approval_reason}")
        elif loc_ctx.confidence_score < 0.70 or loc_ctx.is_ambiguous:
            requires_approval = True
            approval_reason = f"Merchant category confidence is low or ambiguous ({loc_ctx.confidence_score})."
            audit.append(f"HITL Triggered: {approval_reason}")
            
        return TransactionEvaluationResponse(
            merchant_canonical_name=merch_details.canonical_name,
            mcc_code=merch_details.mcc_code,
            primary_category=effective_category,
            recommended_card=best_card,
            alternative_cards=alt_cards,
            requires_approval=requires_approval,
            approval_reason=approval_reason,
            audit_trail=audit
        )
