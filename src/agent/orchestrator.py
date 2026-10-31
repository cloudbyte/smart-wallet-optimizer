from typing import Optional

from langgraph.graph import END, StateGraph

from src.agent.nodes import TransactionAgentNodes
from src.agent.state import AgentLanguageModel, AgentState
from src.models import TransactionEvaluationRequest, TransactionEvaluationResponse
from src.rag.retriever import SmartWalletRAG


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
        graph.add_node("plan_action", self.nodes._plan_action)
        graph.add_node("execute_tool", self.nodes._execute_tool)
        graph.add_node(
            "prepare_benefit_queries", self.nodes._prepare_benefit_queries
        )
        graph.add_node("evaluate_cards", self.nodes._evaluate_cards)
        graph.add_node("critique_tree", self.nodes._critique_tree)
        graph.add_node("generate_explanation", self.nodes._generate_explanation)
        graph.set_entry_point("initialize")
        graph.add_edge("initialize", "plan_action")
        graph.add_edge("plan_action", "execute_tool")
        graph.add_conditional_edges(
            "execute_tool",
            self.nodes._after_tool,
            {
                "plan_action": "plan_action",
                "prepare_benefit_queries": "prepare_benefit_queries",
                "evaluate_cards": "evaluate_cards",
            },
        )
        graph.add_conditional_edges(
            "prepare_benefit_queries",
            self.nodes._after_benefit_planning,
            {"plan_action": "plan_action", "evaluate_cards": "evaluate_cards"},
        )
        graph.add_edge("evaluate_cards", "critique_tree")
        graph.add_edge("critique_tree", "generate_explanation")
        graph.add_edge("generate_explanation", END)
        self.graph = graph.compile()

    def evaluate_transaction(
        self, request: TransactionEvaluationRequest
    ) -> TransactionEvaluationResponse:
        state = self.graph.invoke({"request": request})
        return state["response"]
