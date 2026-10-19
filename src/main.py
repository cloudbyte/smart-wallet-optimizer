import sys
from src.agent import SmartWalletAgent
from src.models import TransactionEvaluationRequest

def main():
    print("=== Smart Wallet Rewards Optimizer (Mock Evaluation) ===")
    merchant = input("Enter merchant name (e.g., Trader Joe's, Delta Air Lines, Walmart): ").strip() or "Trader Joe's"
    try:
        amount = float(input("Enter transaction amount ($): ").strip() or "75.50")
    except ValueError:
        amount = 75.50
        
    is_intl_input = input("Is this an international transaction? (y/N): ").strip().lower()
    is_international = is_intl_input in ["y", "yes"]

    agent = SmartWalletAgent()
    req = TransactionEvaluationRequest(
        merchant_name=merchant,
        amount=amount,
        is_international=is_international
    )

    res = agent.evaluate_transaction(req)

    print("\n--- EVALUATION RESULTS ---")
    print(f"Merchant: {res.merchant_canonical_name} (MCC: {res.mcc_code}, Category: {res.primary_category})")
    print(f"\n✨ Recommended Card: {res.recommended_card.card_name}")
    print(f"   Net Value: {res.recommended_card.net_value_cents:.1f}¢ ($ {res.recommended_card.net_value_cents/100:.2f})")
    print(f"   Rationale: {res.recommended_card.rationale}")
    print(f"   Perks: Purchase Protection: {'Yes' if res.recommended_card.has_purchase_protection else 'No'} | Extended Warranty: {'Yes' if res.recommended_card.has_extended_warranty else 'No'}")

    if res.alternative_cards:
        print("\n💳 Alternatives:")
        for alt in res.alternative_cards:
            print(f"   - {alt.card_name}: {alt.net_value_cents:.1f}¢ net value ({alt.rationale})")

    if res.requires_approval:
        print(f"\n⚠️  APPROVAL REQUIRED: {res.approval_reason}")

    print("\n📋 Audit Trail:")
    for step in res.audit_trail:
        print(f"   • {step}")

if __name__ == "__main__":
    main()
