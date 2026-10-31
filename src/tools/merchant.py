import json
import logging
import os
from src.models import MerchantDetails, MerchantLookupInput

LOGGER = logging.getLogger(__name__)


def lookup_merchant(input_data: MerchantLookupInput) -> MerchantDetails:
    data_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../data/merchants.json")
    )
    with open(data_path, encoding="utf-8") as merchant_file:
        merchants = json.load(merchant_file)

    query = input_data.raw_merchant_name.strip().casefold()
    if not query:
        raise ValueError("Merchant name cannot be empty.")

    for merchant in merchants:
        if query in {
            merchant["raw_merchant_name"].strip().casefold(),
            merchant["canonical_name"].strip().casefold(),
        }:
            return MerchantDetails(**merchant)

    for merchant in merchants:
        names = (
            merchant["raw_merchant_name"].strip().casefold(),
            merchant["canonical_name"].strip().casefold(),
        )
        if len(query) >= 4 and any(name in query or query in name for name in names):
            return MerchantDetails(**merchant)

    LOGGER.info("Merchant not found in local directory: %s", input_data.raw_merchant_name)
    return MerchantDetails(
        canonical_name=input_data.raw_merchant_name.strip(),
        mcc_code="5999",
        primary_category="miscellaneous",
        qualifies_for_supermarket_multipliers=False,
        notes="Unrecognized merchant; conservative default MCC assigned"
    )
