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

    st.caption(
        f"Purchase · **{response.merchant_canonical_name}** · "
        f"{response.primary_category.replace('_', ' ').title()} · "
        f"MCC {response.mcc_code}"
    )

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
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(ellipse at 12% 0%, rgba(80, 95, 232, .11), transparent 34rem),
                #f7f8fc;
        }
        .block-container {
            max-width: 1040px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }
        .page-heading {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin: 0 0 1.2rem;
        }
        .page-heading p {
            margin: 0 0 .2rem;
            color: #68769b;
            font-size: .68rem;
            font-weight: 700;
            letter-spacing: .08em;
            text-transform: uppercase;
        }
        .page-heading h1 {
            margin: 0;
            color: #27324a;
            font-size: clamp(1.35rem, 2.5vw, 1.8rem);
            line-height: 1.15;
        }
        .demo-pill {
            flex: 0 0 auto;
            padding: .3rem .6rem;
            color: #58658a;
            background: #eef0f8;
            border-radius: 999px;
            font-size: .68rem;
            font-weight: 700;
            letter-spacing: .04em;
        }
        [data-testid="stForm"] {
            padding: 1rem 1.2rem;
            background: #ffffff;
            border: 1px solid #e4e8f2;
            border-radius: 15px;
            box-shadow: 0 5px 18px rgba(36, 48, 90, .045);
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff;
            border-color: #e4e8f2;
            border-radius: 15px;
            box-shadow: 0 5px 18px rgba(36, 48, 90, .045);
        }
        [data-testid="stMetric"] {
            padding: .8rem .9rem;
            background: #f8f9fc;
            border: 1px solid #e4e8f2;
            border-radius: 14px;
        }
        [data-testid="stMetricLabel"] p { color: #526078; }
        [data-testid="stMetricValue"] { color: #202c47; }
        [data-testid="stDataFrame"] { border-radius: 12px; }
        div.stFormSubmitButton > button {
            min-height: 2.9rem;
            border-radius: 11px;
            font-weight: 650;
        }
        @media (max-width: 700px) {
            .block-container { padding: 1.25rem 1rem 2.5rem; }
            .page-heading { align-items: flex-start; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <header class="page-heading">
          <div>
            <p>Smart Wallet · Rewards assistant</p>
            <h1>Smart Wallet Rewards Optimizer</h1>
          </div>
          <span class="demo-pill">DEMO</span>
        </header>
        """,
        unsafe_allow_html=True,
    )

    with st.form("transaction"):
        merchant_col, amount_col = st.columns([1.6, 1], gap="medium")
        with merchant_col:
            merchant_name = st.text_input(
                "Merchant",
                value="Trader Joe's",
                help="Enter a merchant from the catalog or a name for AI-assisted matching.",
            )
        with amount_col:
            amount = st.number_input(
                "Purchase amount",
                min_value=0.01,
                max_value=50_000.0,
                value=75.50,
                step=1.0,
                format="%.2f",
                help="Amount in US dollars.",
            )
        confirm_col, submit_col = st.columns([1.6, 1], gap="medium")
        with confirm_col:
            is_international = st.checkbox("International purchase")
        with submit_col:
            submitted = st.form_submit_button(
                "Find my best card →",
                type="primary",
                use_container_width=True,
            )
    st.caption(
        f"Synthetic demo · Illustrative estimates, not financial advice · "
        f"Model: {default_model_id()}"
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
        st.markdown("### Recommendation")
        render_result(response)


if __name__ == "__main__":
    main()
