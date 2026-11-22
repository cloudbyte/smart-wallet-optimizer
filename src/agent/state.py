from typing import List, Optional, Protocol, TypedDict

from src.models import (
    BenefitsRiskAssessment,
    CardRecommendation,
    ContextRewardsAssessment,
    LocationContext,
    MerchantDetails,
    MerchantResolution,
    ThoughtNode,
    TransactionEvaluationRequest,
    TransactionEvaluationResponse,
    UserWallet,
)


class AgentLanguageModel(Protocol):
    def explain(self, facts: str) -> str: ...

    def resolve_merchant(
        self, merchant_name: str, candidates: List[str]
    ) -> Optional[MerchantResolution]: ...

    def critique_tree(self, facts: str) -> str: ...


class AgentState(TypedDict, total=False):
    request: TransactionEvaluationRequest
    location_context: LocationContext
    location_audit: str
    merchant: MerchantDetails
    merchant_audit: str
    wallet: UserWallet
    wallet_audit: str
    context_assessment: ContextRewardsAssessment
    benefits_assessment: BenefitsRiskAssessment
    candidate_cards: List[str]
    reasoning_tree: List[ThoughtNode]
    recommended_card: Optional[CardRecommendation]
    tree_assessment: Optional[str]
    assistant_message: Optional[str]
    audit_trail: List[str]
    response: TransactionEvaluationResponse
