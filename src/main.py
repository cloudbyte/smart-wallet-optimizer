from pydantic import ValidationError

from src.agent import SmartWalletAgent
from src.llm import Qwen3AgentModel
from src.models import TransactionEvaluationRequest

def main():
    print("=== Smart Wallet Rewards Optimizer (Qwen3 LLM Agent) ===")
    merchant = input("Enter merchant name (e.g., Trader Joe's, Delta Air Lines, Walmart): ").strip() or "Trader Joe's"
    try:
        amount = float(input("Enter transaction amount ($): ").strip() or "75.50")
    except ValueError:
        print("Invalid amount: enter a positive number.")
        return
        
    is_intl_input = input("Is this an international transaction? (y/N): ").strip().lower()
    is_international = is_intl_input in ["y", "yes"]

    try:
        req = TransactionEvaluationRequest(
            merchant_name=merchant,
            amount=amount,
            is_international=is_international,
        )
    except ValidationError as error:
        print(f"Invalid transaction: {error}")
        return

    agent = SmartWalletAgent(explainer=Qwen3AgentModel())

    res = agent.evaluate_transaction(req)

    print("\n--- EVALUATION RESULTS ---")
    if res.blocked_reason:
        print(f"\n⛔ {res.blocked_reason}")
        for step in res.audit_trail:
            print(f"   • {step}")
        return
    if res.assistant_message:
        print(f"\n🤖 AI-generated summary (informational): {res.assistant_message}")
    if res.tree_assessment:
        print(f"\n🧠 LLM ToT critic: {res.tree_assessment}")
    print(f"Merchant: {res.merchant_canonical_name} (MCC: {res.mcc_code}, Category: {res.primary_category})")
    if res.recommended_card:
        print(f"\n✨ Recommended Card: {res.recommended_card.card_name}")
        print(f"   Net Value: {res.recommended_card.net_value_cents:.1f}¢ ($ {res.recommended_card.net_value_cents/100:.2f})")
        print(f"   Rationale: {res.recommended_card.rationale}")
        print(f"   Perks: Purchase Protection: {'Yes' if res.recommended_card.has_purchase_protection else 'No'} | Extended Warranty: {'Yes' if res.recommended_card.has_extended_warranty else 'No'}")
    else:
        print("\n⚠️  No eligible card recommendation is available.")

    if res.alternative_cards:
        print("\n💳 Alternatives:")
        for alt in res.alternative_cards:
            print(f"   - {alt.card_name}: {alt.net_value_cents:.1f}¢ net value ({alt.rationale})")

    if res.requires_approval:
        print(f"\n⚠️  APPROVAL REQUIRED: {res.approval_reason}")

    print("\n🌳 ToT Evaluation:")
    for node in res.reasoning_tree:
        net_value = (
            f", net {node.net_value_cents:.1f}¢"
            if node.net_value_cents is not None
            else ""
        )
        print(
            f"   - Depth {node.depth} | {node.card_id} | {node.status}"
            f"{net_value}"
        )

    print("\n📋 Audit Trail:")
    for step in res.audit_trail:
        print(f"   • {step}")

if __name__ == "__main__":
    main()
