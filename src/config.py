"""
Central configuration: the canonical schema and shared mappings.

Everything else in the pipeline converts TO this standard so the rest of the
code never has to worry about which platform a row came from.
"""

# ---------------------------------------------------------------------------
# The canonical schema = the one official set of columns every row will have.
# Each per-platform "adapter" must output exactly these columns.
# ---------------------------------------------------------------------------
CANONICAL_COLUMNS = [
    "date",           # the day this record is for (real datetime)
    "channel",        # which platform: "google", "bing", or "meta"
    "campaign_id",    # the campaign's id (kept as text to be safe)
    "campaign_name",  # human-readable campaign name
    "campaign_type",  # normalized type, e.g. "search", "performance_max"
    "spend",          # money spent, in dollars
    "revenue",        # money earned, in dollars (NaN when not available)
    "clicks",         # number of clicks
    "impressions",    # number of times the ad was shown
    "conversions",    # number of conversions (purchases/actions)
]

# Numeric columns we always force to be numbers (bad cells become NaN).
NUMERIC_COLUMNS = ["spend", "revenue", "clicks", "impressions", "conversions"]

# ---------------------------------------------------------------------------
# Campaign-type spelling map.
# Each platform writes the same type differently (PERFORMANCE_MAX vs
# PerformanceMax). We squash everything to one agreed spelling.
# Key = lowercased/cleaned raw value, Value = our canonical label.
# ---------------------------------------------------------------------------
CAMPAIGN_TYPE_MAP = {
    "performance_max": "performance_max",
    "performancemax": "performance_max",
    "pmax": "performance_max",
    "search": "search",
    "shopping": "shopping",
    "video": "video",
    "display": "display",
    "demand_gen": "demand_gen",
    "demandgen": "demand_gen",
    "audience": "audience",
}


def normalize_campaign_type(value):
    """Turn any raw type spelling into our one canonical label."""
    if value is None:
        return "unknown"
    cleaned = str(value).strip().lower().replace(" ", "_")
    return CAMPAIGN_TYPE_MAP.get(cleaned, cleaned or "unknown")
