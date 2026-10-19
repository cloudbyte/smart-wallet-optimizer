import json
import os
from src.models import UserWallet

def get_user_wallet(user_id: str = "user_alex_01") -> UserWallet:
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/wallets.json"))
    with open(data_path, "r") as f:
        data = json.load(f)
    return UserWallet(**data)
