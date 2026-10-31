import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

class SmartWalletRAG:
    def __init__(self):
        # Placeholder for RAG retrieval of card benefit guides (Purchase Protection, Warranty, etc.)
        self.documents = [
            {"card_id": "chase_sapphire_reserve", "benefit": "Purchase Protection covers new purchases for up to $10,000 per claim ($50,000 per account) against theft or damage for 120 days."},
            {"card_id": "chase_sapphire_reserve", "benefit": "Extended Warranty Protection adds an additional year to the eligible manufacturer's warranty of three years or less."},
            {"card_id": "amex_gold", "benefit": "Purchase Protection covers up to $10,000 per occurrence, up to $50,000 per calendar year, against accidental damage or theft for 90 days."}
        ]

    def query_benefits(self, card_id: str, query: str) -> str:
        matches = [d["benefit"] for d in self.documents if d["card_id"] == card_id]
        if matches:
            return " ".join(matches)
        return "Standard card benefits apply."
