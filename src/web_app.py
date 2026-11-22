from typing import Optional

import streamlit as st
from pydantic import ValidationError

from src.agent import SmartWalletAgent
from src.llm import (
    CLAUDE_MODEL_IDS,
    LOCAL_MODEL_ID,
    AnthropicClaudeModel,
    Qwen3AgentModel,
    default_model_id,
)
from src.models import TransactionEvaluationRequest, TransactionEvaluationResponse
from src.safety.guardrails import transaction_block_reason


LOCAL_MODEL_LABEL = "Qwen3 0.6B (local)"
MODEL_OPTIONS = {
    LOCAL_MODEL_LABEL: None,
    **CLAUDE_MODEL_IDS,
}


@st.cache_resource
def get_agent() -> SmartWalletAgent:
    return SmartWalletAgent(explainer=Qwen3AgentModel())


def clear_previous_evaluation() -> None:
    st.session_state.pop("transaction_request", None)
    st.session_state.pop("evaluation_response", None)
    st.session_state.pop("selected_model", None)
    st.session_state.pop("anthropic_api_key", None)
    st.session_state.pop("review_anthropic_api_key", None)


def build_agent(model_label: str, api_key: str = "") -> SmartWalletAgent:
    if model_label not in MODEL_OPTIONS:
        raise ValueError(f"Unsupported AI model: {model_label}")
    model_id = MODEL_OPTIONS.get(model_label)
    if model_id is None:
        return get_agent()
    if not api_key.strip():
        raise ValueError("Enter your Anthropic API key to use a Claude model.")
    return SmartWalletAgent(
        explainer=AnthropicClaudeModel(model_id=model_id, api_key=api_key)
    )


def evaluate_with_progress(
    agent: SmartWalletAgent,
    request: TransactionEvaluationRequest,
) -> TransactionEvaluationResponse:
    with st.status("Starting transaction evaluation...", expanded=True) as status:
        response = agent.evaluate_transaction(
            request,
            progress_callback=lambda label: status.update(label=label),
        )
        if getattr(response, "blocked_reason", None) is not None:
            status.update(
                label="Transaction blocked by the safety limit.",
                state="error",
                expanded=True,
            )
        else:
            status.update(
                label="Evaluation complete.",
                state="complete",
                expanded=False,
            )
    return response


def render_evaluation_trace(response: TransactionEvaluationResponse) -> None:
    st.markdown("### 🌳 ToT Evaluation:")
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
    else:
        st.caption("No Tree-of-Thought branches were generated for this evaluation.")

    st.markdown("### 📋 Audit Trail:")
    if response.audit_trail:
        for item in response.audit_trail:
            st.markdown(f"- {item}")
    else:
        st.caption("No audit events were recorded.")


def render_result(
    response: TransactionEvaluationResponse,
    request: Optional[TransactionEvaluationRequest] = None,
    model_label: str = LOCAL_MODEL_LABEL,
) -> None:
    blocked_reason = (
        transaction_block_reason(request)
        if request is not None
        else getattr(response, "blocked_reason", None)
    )
    if blocked_reason is not None:
        st.warning(blocked_reason, icon="⚠️")
        render_evaluation_trace(response)
        return

    if response.requires_approval and response.human_reviewed:
        st.success(
            "You reviewed this recommendation. The app has not made or authorized "
            "a payment."
        )
    elif response.requires_approval:
        st.warning(
            f"**Confirmation required**  \n{response.approval_reason}",
            icon="⚠️",
        )

    if response.requires_approval and request is not None:
        category_options = [
            "dining",
            "groceries",
            "gas",
            "travel",
            "wholesale_department",
            "retail",
            "pharmacy",
            "subscriptions",
            "entertainment",
            "miscellaneous",
        ]
        if response.primary_category not in category_options:
            category_options.append(response.primary_category)
        default_category_index = category_options.index(response.primary_category)
        with st.form("human_review", clear_on_submit=True):
            st.markdown("#### Review or correct the transaction details")
            reviewed_merchant = st.text_input(
                "Confirmed merchant", value=response.merchant_canonical_name
            )
            reviewed_category = st.selectbox(
                "Confirmed category",
                options=category_options,
                index=default_category_index,
            )
            reviewer_confirmed = st.checkbox(
                "I reviewed these details and understand this app does not make payments."
            )
            review_api_key = ""
            if model_label != LOCAL_MODEL_LABEL:
                review_api_key = st.text_input(
                    "Anthropic API key",
                    type="password",
                    key="review_anthropic_api_key",
                    help=(
                        "Used only for this re-evaluation and cleared when you "
                        "submit the review."
                    ),
                )
            review_submitted = st.form_submit_button("Submit review")

        if review_submitted:
            if not reviewer_confirmed:
                st.error("Confirm that you reviewed the details before submitting.")
            elif (
                reviewed_merchant.strip() != response.merchant_canonical_name
                or reviewed_category != response.primary_category
                or (
                    request.category_hint is not None
                    and request.category_hint != reviewed_category
                )
            ):
                if model_label != LOCAL_MODEL_LABEL and not review_api_key.strip():
                    st.error("Enter your Anthropic API key to re-evaluate.")
                    return
                try:
                    corrected_request = TransactionEvaluationRequest(
                        merchant_name=reviewed_merchant,
                        amount=request.amount,
                        category_hint=reviewed_category,
                        is_international=request.is_international,
                    )
                except ValidationError as error:
                    st.error(f"Check the corrected details: {error}")
                    return
                corrected_response = evaluate_with_progress(
                    build_agent(model_label, review_api_key),
                    corrected_request,
                )
                corrected_response.audit_trail.append(
                    "Human review corrected transaction context: "
                    f"merchant={corrected_request.merchant_name}; "
                    f"category={reviewed_category}."
                )
                st.session_state["transaction_request"] = corrected_request
                st.session_state["evaluation_response"] = corrected_response
                st.rerun()
            else:
                reviewed_response = response.model_copy(
                    update={"human_reviewed": True}
                )
                reviewed_response.audit_trail.append(
                    "User confirmed review of the transaction context and "
                    "recommendation; no payment was executed."
                )
                st.session_state["evaluation_response"] = reviewed_response
                st.rerun()

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

    render_evaluation_trace(response)


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

    model_label = st.selectbox(
        "AI model",
        options=list(MODEL_OPTIONS),
        key="model_choice",
        on_change=clear_previous_evaluation,
        help=(
            "Choose a local model or a Claude model. Claude requests are sent "
            "to Anthropic's API."
        ),
    )
    previous_request = st.session_state.get("transaction_request")
    with st.form("transaction", clear_on_submit=True):
        merchant_col, amount_col = st.columns([1.6, 1], gap="medium")
        with merchant_col:
            merchant_name = st.text_input(
                "Merchant",
                value=(
                    previous_request.merchant_name
                    if previous_request is not None
                    else "Trader Joe's"
                ),
                help="Enter a merchant from the catalog or a name for AI-assisted matching.",
            )
        with amount_col:
            amount = st.number_input(
                "Purchase amount",
                min_value=0.01,
                value=(
                    previous_request.amount
                    if previous_request is not None
                    else 75.50
                ),
                step=1.0,
                format="%.2f",
                help=(
                    "Amount in US dollars. Recommendations are blocked for "
                    "amounts over $50,000."
                ),
            )
        api_key = ""
        if model_label != LOCAL_MODEL_LABEL:
            api_key = st.text_input(
                "Anthropic API key",
                type="password",
                key="anthropic_api_key",
                help=(
                    "Sent directly to Anthropic for this evaluation and cleared "
                    "when you submit."
                ),
            )
        confirm_col, submit_col = st.columns([1.6, 1], gap="medium")
        with confirm_col:
            is_international = st.checkbox(
                "International purchase",
                value=(
                    previous_request.is_international
                    if previous_request is not None
                    else False
                ),
            )
        with submit_col:
            submitted = st.form_submit_button(
                "Find my best card →",
                type="primary",
                use_container_width=True,
            )
    st.caption(
        f"Synthetic demo · Illustrative estimates, not financial advice · "
        f"Model: {default_model_id() if model_label == LOCAL_MODEL_LABEL else MODEL_OPTIONS[model_label]}"
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

        st.session_state["transaction_request"] = request
        st.session_state["selected_model"] = model_label
        if model_label != LOCAL_MODEL_LABEL and not api_key.strip():
            st.error("Enter your Anthropic API key to use a Claude model.")
            return

        try:
            agent = build_agent(model_label, api_key)
        except ValueError as error:
            st.error(str(error))
            return
        response = evaluate_with_progress(agent, request)
        st.session_state["evaluation_response"] = response

    response = st.session_state.get("evaluation_response")
    if response is not None:
        st.divider()
        st.markdown("## --- EVALUATION RESULTS ---")
        request = st.session_state.get("transaction_request")
        blocked_reason = (
            transaction_block_reason(request)
            if request is not None
            else getattr(response, "blocked_reason", None)
        )
        heading = (
            "Transaction blocked" if blocked_reason is not None else "Recommendation"
        )
        st.markdown(f"### {heading}")
        render_result(
            response,
            request,
            st.session_state.get("selected_model", model_label),
        )


if __name__ == "__main__":
    main()
