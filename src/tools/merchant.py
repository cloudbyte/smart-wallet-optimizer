import json
import os
from typing import Optional
from src.models import MerchantLookupInput, MerchantDetails

def lookup_merchant(input_data: MerchantLookupInput) -> MerchantDetails:
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/merchants.json"))
    try:
        with open(data_path, "r") as f:
            merchants = json.load(f)
        
        query = input_data.raw_merchant_name.lower()
        for m in merchants:
            if query in m["raw_merchant_name"].lower() or query in m["canonical_name"].lower():
                return MerchantDetails(**m)
    except Exception:
        pass
    
    # Default fallback for unknown merchants
    return MerchantDetails(
        canonical_name=input_data.raw_merchant_name,
        mcc_code="5999",
        primary_category="miscellaneous",
        qualifies_for_supermarket_multipliers=False,
        notes="Unrecognized merchant; default MCC assigned"
    )
