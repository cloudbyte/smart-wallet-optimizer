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
        st.warning(f"Confirmation required: {response.approval_reason}")

    st.subheader("Merchant")
    st.write(
        f"**{response.merchant_canonical_name}** · "
        f"{response.primary_category} · MCC {response.mcc_code}"
    )

    recommendation = response.recommended_card
    st.subheader("Recommendation")
    if recommendation is None:
        st.error("No eligible card recommendation is available.")
    else:
        st.markdown(f"### {recommendation.card_name}")
        first, second = st.columns(2)
        first.metric("Estimated net value", f"{recommendation.net_value_cents:.1f}¢")
        second.metric("Estimated rewards", f"{recommendation.expected_yield_cents:.1f}¢")
        st.write(recommendation.rationale)

    if response.assistant_message:
        st.info(response.assistant_message)
    if response.tree_assessment:
        st.markdown(f"**Model's evidence-grounded assessment:** {response.tree_assessment}")

    if response.alternative_cards:
        st.subheader("Other cards evaluated")
        st.dataframe(
            [
                {
                    "Card": card.card_name,
                    "Estimated net value (¢)": card.net_value_cents,
                    "Estimated rewards (¢)": card.expected_yield_cents,
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
        layout="centered",
    )
    st.title("Smart Wallet Rewards Optimizer")
    st.caption(
        "Demo using synthetic wallet data. Recommendations are illustrative "
        "and are not financial advice."
    )
    st.caption(f"Language model: `{default_model_id()}`")

    with st.form("transaction"):
        merchant_name = st.text_input(
            "Merchant",
            value="Trader Joe's",
            help="Enter a merchant from the catalog or a name for AI-assisted matching.",
        )
        amount = st.number_input(
            "Transaction amount (USD)",
            min_value=0.01,
            max_value=50_000.0,
            value=75.50,
            step=1.0,
        )
        is_international = st.checkbox("International transaction")
        submitted = st.form_submit_button("Recommend a card", type="primary")

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
        render_result(response)


if __name__ == "__main__":
    main()
