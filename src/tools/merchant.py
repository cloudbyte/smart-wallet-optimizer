import json
import logging
import os
from typing import Callable, List, Optional

from src.models import MerchantDetails, MerchantLookupInput, MerchantResolution

LOGGER = logging.getLogger(__name__)
MIN_AI_MATCH_CONFIDENCE = 0.85

MerchantResolver = Callable[
    [str, List[str]], Optional[MerchantResolution]
]


def lookup_merchant(
    input_data: MerchantLookupInput,
    resolver: Optional[MerchantResolver] = None,
) -> MerchantDetails:
    data_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../data/merchants.json")
    )
    with open(data_path, encoding="utf-8") as merchant_file:
        merchants = json.load(merchant_file)

    query = input_data.raw_merchant_name.strip().casefold()
    if not query:
        raise ValueError("Merchant name cannot be empty.")

    catalog = [
        (entry, MerchantDetails(**entry))
        for entry in merchants
    ]
    for entry, merchant in catalog:
        known_names = {
            entry["raw_merchant_name"].strip().casefold(),
            merchant.canonical_name.strip().casefold(),
            *(alias.strip().casefold() for alias in merchant.aliases),
        }
        if query in known_names:
            return merchant

    if resolver is not None:
        catalog_names = list(
            dict.fromkeys(
                name
                for entry, merchant in catalog
                for name in (
                    merchant.canonical_name,
                    entry["raw_merchant_name"].strip(),
                    *merchant.aliases,
                )
            )
        )
        try:
            match = resolver(input_data.raw_merchant_name, catalog_names)
        except (OSError, RuntimeError, ValueError) as error:
            LOGGER.warning(
                "AI merchant resolution failed (%s); requiring confirmation.",
                type(error).__name__,
            )
            match = None
        if (
            match is not None
            and match.canonical_name is not None
            and match.confidence >= MIN_AI_MATCH_CONFIDENCE
        ):
            matched = next(
                (
                    merchant
                    for entry, merchant in catalog
                    if match.canonical_name
                    in {
                        merchant.canonical_name,
                        entry["raw_merchant_name"].strip(),
                        *merchant.aliases,
                    }
                ),
                None,
            )
            if matched is not None:
                return matched.model_copy(
                    update={
                        "resolution_method": "ai_candidate",
                        "match_confidence": match.confidence,
                        "requires_confirmation": True,
                        "notes": (
                            f"AI matched the input to {matched.canonical_name}; "
                            "confirm merchant identity before relying on this result."
                        ),
                    }
                )

    LOGGER.info("Merchant not found in local directory: %s", input_data.raw_merchant_name)
    return MerchantDetails(
        canonical_name=input_data.raw_merchant_name.strip(),
        mcc_code="5999",
        primary_category="miscellaneous",
        qualifies_for_supermarket_multipliers=False,
        notes="Unrecognized merchant; conservative default MCC assigned",
        resolution_method="unresolved",
        match_confidence=0.0,
        requires_confirmation=True,
    )
