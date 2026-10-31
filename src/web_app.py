import streamlit as st
from pydantic import ValidationError

from src.agent import SmartWalletAgent
from src.llm import Qwen3AgentModel, default_model_id
from src.models import TransactionEvaluationRequest, TransactionEvaluationResponse


@st.cache_resource
def get_agent() -> SmartWalletAgent:
    return SmartWalletAgent(explainer=Qwen3AgentModel())


def render_result(response: TransactionEvaluationResponse) -> None:
    if response.requires_approval:
        st.warning(
            f"**Confirmation required**  \n{response.approval_reason}",
            icon="⚠️",
        )

    recommendation = response.recommended_card
    if recommendation is None:
        st.error("No eligible card recommendation is available.")
    else:
        with st.container(border=True):
            st.markdown("#### ✨ Recommended card")
            st.markdown(f"## {recommendation.card_name}")
            st.caption(recommendation.rationale)
            first, second, third = st.columns(3)
            first.metric("Estimated net value", f"{recommendation.net_value_cents:.1f}¢")
            second.metric("Estimated rewards", f"{recommendation.expected_yield_cents:.1f}¢")
            third.metric("Foreign fee", f"{recommendation.foreign_transaction_fee:.1f}%")

    st.markdown("#### Purchase details")
    merchant, category, mcc = st.columns(3)
    merchant.metric("Merchant", response.merchant_canonical_name)
    category.metric("Category", response.primary_category.replace("_", " ").title())
    mcc.metric("Merchant category code", response.mcc_code)

    if response.assistant_message:
        st.info(response.assistant_message)
    if response.tree_assessment:
        with st.container(border=True):
            st.markdown("#### Model's evidence-grounded assessment")
            st.write(response.tree_assessment)

    if response.alternative_cards:
        st.markdown("#### Other cards evaluated")
        st.dataframe(
            [
                {
                    "Card": card.card_name,
                    "Net value (¢)": round(card.net_value_cents, 1),
                    "Rewards (¢)": round(card.expected_yield_cents, 1),
                }
                for card in response.alternative_cards
            ],
            hide_index=True,
            use_container_width=True,
        )

    with st.expander("Reasoning and audit trail"):
        if response.reasoning_tree:
            st.dataframe(
                [
                    {
                        "Depth": node.depth,
                        "Card": node.card_id,
                        "Status": node.status,
                        "Net value (¢)": node.net_value_cents,
                        "Evidence": node.evidence,
                    }
                    for node in response.reasoning_tree
                ],
                hide_index=True,
                use_container_width=True,
            )
        for item in response.audit_trail:
            st.write(f"- {item}")


def main() -> None:
    st.set_page_config(
        page_title="Smart Wallet Rewards Optimizer",
        page_icon="💳",
        layout="wide",
    )
    st.title("💳 Smart Wallet")
    st.markdown("### Rewards, made easier.")
    st.caption(
        "Compare your wallet's cards for a purchase and see why one may fit best."
    )

    entry, explainer = st.columns([1.35, 0.9], gap="large")
    with entry:
        st.markdown("#### Tell us about your purchase")
        with st.form("transaction"):
            merchant_name = st.text_input(
                "Merchant",
                value="Trader Joe's",
                help="Enter a merchant from the catalog or a name for AI-assisted matching.",
            )
            amount = st.number_input(
                "Purchase amount",
                min_value=0.01,
                max_value=50_000.0,
                value=75.50,
                step=1.0,
                format="%.2f",
                help="Amount in US dollars.",
            )
            is_international = st.checkbox("This is an international purchase")
            submitted = st.form_submit_button(
                "Find my best card  →",
                type="primary",
                use_container_width=True,
            )

    with explainer:
        with st.container(border=True):
            st.markdown("#### What you'll get")
            st.markdown(
                """
                - **A card suggestion** based on your wallet
                - **An estimated value** from rewards and applicable fees
                - **A clear explanation** of the result
                """
            )
            st.caption(f"Powered by {default_model_id()}")
            st.caption(
                "Demo with synthetic wallet data. Estimates are illustrative, "
                "not financial advice."
            )

    if submitted:
        try:
            request = TransactionEvaluationRequest(
                merchant_name=merchant_name,
                amount=amount,
                is_international=is_international,
            )
        except ValidationError as error:
            st.error(f"Check the transaction details: {error}")
            return

        with st.spinner("Evaluating merchant, wallet, and card benefits..."):
            response = get_agent().evaluate_transaction(request)
        st.session_state["evaluation_response"] = response

    response = st.session_state.get("evaluation_response")
    if response is not None:
        st.divider()
        st.markdown("### Your recommendation")
        render_result(response)


if __name__ == "__main__":
    main()
