from time import perf_counter
from typing import Callable, Optional

from langgraph.graph import END, StateGraph

from src.agent.nodes import TransactionAgentNodes
from src.agent.state import AgentLanguageModel, AgentState
from src.models import TransactionEvaluationRequest, TransactionEvaluationResponse
from src.rag.retriever import SmartWalletRAG
from src.safety.guardrails import transaction_block_reason


class SmartWalletAgent:
    def __init__(
        self,
        user_id: str = "user_alex_01",
        explainer: Optional[AgentLanguageModel] = None,
        rag: Optional[SmartWalletRAG] = None,
    ):
        self.nodes = TransactionAgentNodes(
            user_id=user_id,
            explainer=explainer,
            rag=rag,
        )
        self.rag = self.nodes.rag
        graph = StateGraph(AgentState)
        graph.add_node("initialize", self.nodes._initialize)
        graph.add_node("resolve_location", self.nodes._resolve_location)
        graph.add_node("lookup_merchant", self.nodes._lookup_merchant)
        graph.add_node("load_wallet", self.nodes._load_wallet)
        graph.add_node(
            "context_rewards_specialist",
            self.nodes._context_rewards_specialist,
        )
        graph.add_node(
            "benefits_risk_specialist",
            self.nodes._benefits_risk_specialist,
        )
        graph.add_node(
            "decision_orchestrator",
            self.nodes._decision_orchestrator,
        )
        graph.add_node("critique_tree", self.nodes._critique_tree)
        graph.add_node("generate_explanation", self.nodes._generate_explanation)
        graph.set_entry_point("initialize")
        graph.add_edge("initialize", "resolve_location")
        graph.add_edge("initialize", "lookup_merchant")
        graph.add_edge("initialize", "load_wallet")
        graph.add_edge(
            ["resolve_location", "lookup_merchant", "load_wallet"],
            "context_rewards_specialist",
        )
        graph.add_edge("context_rewards_specialist", "benefits_risk_specialist")
        graph.add_edge("benefits_risk_specialist", "decision_orchestrator")
        graph.add_edge("decision_orchestrator", "critique_tree")
        graph.add_edge("critique_tree", "generate_explanation")
        graph.add_edge("generate_explanation", END)
        self.graph = graph.compile()

    def evaluate_transaction(
        self,
        request: TransactionEvaluationRequest,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> TransactionEvaluationResponse:
        started_at = perf_counter()
        block_reason = transaction_block_reason(request)
        if block_reason is not None:
            if progress_callback is not None:
                progress_callback("Checking transaction safety limits...")
            response = TransactionEvaluationResponse(
                merchant_canonical_name=request.merchant_name,
                mcc_code="N/A",
                primary_category="unverified",
                recommended_card=None,
                alternative_cards=[],
                blocked_reason=block_reason,
                audit_trail=[
                    "Transaction blocked by the amount safety guardrail; "
                    "no recommendation was generated."
                ],
            )
            elapsed_ms = (perf_counter() - started_at) * 1000
            response.audit_trail.append(
                f"End-to-end evaluation latency: {elapsed_ms:.1f} ms."
            )
            if progress_callback is not None:
                progress_callback("Transaction blocked by the amount safety limit.")
            return response

        progress_labels = {
            "initialize": "Starting transaction evaluation...",
            "resolve_location": "Resolving merchant location context...",
            "lookup_merchant": "Matching merchant and spending category...",
            "load_wallet": "Loading wallet and eligible cards...",
            "context_rewards_specialist": "Calculating card reward estimates...",
            "benefits_risk_specialist": "Checking card benefits and fees...",
            "decision_orchestrator": "Comparing cards and applying safety checks...",
            "critique_tree": "Verifying the recommendation against evidence...",
            "generate_explanation": "Preparing the recommendation...",
        }
        state: AgentState = {}
        for update in self.graph.stream(
            {"request": request},
            stream_mode="updates",
        ):
            for node_name, node_update in update.items():
                if node_update is not None:
                    state.update(node_update)
                if progress_callback is not None:
                    progress_callback(progress_labels.get(node_name, node_name))

        response = state["response"]
        elapsed_ms = (perf_counter() - started_at) * 1000
        response.audit_trail.append(
            f"End-to-end evaluation latency: {elapsed_ms:.1f} ms."
        )
        return response
