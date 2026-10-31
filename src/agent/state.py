from typing import Dict, List, Optional, Protocol, TypedDict

from src.models import (
    LocationContext,
    MerchantDetails,
    MerchantResolution,
    TransactionEvaluationRequest,
    TransactionEvaluationResponse,
    UserWallet,
)


class AgentLanguageModel(Protocol):
    def explain(self, facts: str) -> str: ...

    def choose_tool(self, context: str, available_actions: List[str]) -> Optional[str]: ...

    def resolve_merchant(
        self, merchant_name: str, candidates: List[str]
    ) -> Optional[MerchantResolution]: ...

    def critique_tree(self, facts: str) -> str: ...

    def embed(self, text: str) -> List[float]: ...


class AgentState(TypedDict, total=False):
    request: TransactionEvaluationRequest
    phase: str
    pending_actions: List[str]
    next_action: str
    location_context: LocationContext
    merchant: MerchantDetails
    wallet: UserWallet
    benefit_evidence: Dict[str, str]
    audit_trail: List[str]
    response: TransactionEvaluationResponse
