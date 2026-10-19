from typing import Optional
from src.models import LocationInput, LocationContext

def resolve_location(input_data: LocationInput) -> LocationContext:
    if input_data.venue_override:
        name = input_data.venue_override.lower()
        if "trader" in name:
            return LocationContext(venue_name="Trader Joe's", inferred_category="groceries", is_ambiguous=False, confidence_score=0.98)
        elif "walmart" in name:
            return LocationContext(venue_name="Walmart Supercenter", inferred_category="wholesale_department", is_ambiguous=True, confidence_score=0.75)
        elif "delta" in name:
            return LocationContext(venue_name="Delta Air Lines", inferred_category="travel", is_ambiguous=False, confidence_score=0.99)
        elif "shell" in name:
            return LocationContext(venue_name="Shell Oil", inferred_category="gas", is_ambiguous=False, confidence_score=0.95)
        else:
            return LocationContext(venue_name=input_data.venue_override, inferred_category="dining", is_ambiguous=True, confidence_score=0.70)
    
    return LocationContext(venue_name="Trader Joe's", inferred_category="groceries", is_ambiguous=False, confidence_score=0.95)
