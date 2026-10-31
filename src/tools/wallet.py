import json
import os
from src.models import UserWallet

def get_user_wallet(user_id: str = "user_alex_01") -> UserWallet:
    data_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../data/wallets.json")
    )
    with open(data_path, encoding="utf-8") as wallet_file:
        data = json.load(wallet_file)
    wallet = UserWallet(**data)
    if wallet.user_id != user_id:
        raise ValueError(f"No synthetic wallet is configured for user {user_id!r}.")
    return wallet
