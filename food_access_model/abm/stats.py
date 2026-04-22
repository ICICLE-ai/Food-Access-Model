import math
import logging
import inspect
from typing import Any, Optional

LOW_FOOD_ACCESS_THRESHOLD = 60   # mfai below this → "low food access"

def mean_food_score(households):
    """Mean MFAI score across all households with a valid score."""
    values = [h["Food Access Score"] for h in households if h.get("Food Access Score") is not None]
    return sum(values) / len(values) if values else None

def stddev_food_score(households) -> Optional[float]:
    """
    Population standard deviation of MFAI scores.
    Uses the population formula (divides by N) since households represent
    the full simulated population, not a sample.
    """
    values = [h["Food Access Score"] for h in households if h.get("Food Access Score") is not None]
    if len(values) < 2:
        return None
    mu = sum(values) / len(values)
    variance = sum((v - mu) ** 2 for v in values) / len(values)
    return math.sqrt(variance)

def pct_low_food_access_households(households, threshold=LOW_FOOD_ACCESS_THRESHOLD):
    """
    Percentage of households with an MFAI score below `threshold`.
    Returns a value in [0, 100].
    """
    valid_households = [h for h in households if h.get("Food Access Score") is not None]
    if not valid_households:
        return None

    low_access_count = sum(1 for h in valid_households if h["Food Access Score"] < threshold)
    return 100 * low_access_count / len(valid_households)


def mean_distance_to_nearest_supermarket(households):
    """Mean distance (miles) to the closest food store across all households."""
    distances = [h["Closest Store (Miles)"] for h in households if h.get("Closest Store (Miles)") is not None]
    return sum(distances) / len(distances) if distances else None

def stddev_distance_to_nearest_supermarket(households) -> Optional[float]:
    """
    Population standard deviation of distance (miles) to the closest food store.
    Useful for understanding spatial inequality in food store access.
    """
    distances = [h["Closest Store (Miles)"] for h in households if h.get("Closest Store (Miles)") is not None]
    if len(distances) < 2:
        return None
    mu = sum(distances) / len(distances)
    variance = sum((d - mu) ** 2 for d in distances) / len(distances)
    return math.sqrt(variance)
 

def mean_travel_time_to_nearest_supermarket(households):
    """Mean driving time (minutes) to the nearest food store across all households."""
    times = [h["Driving time"] for h in households if h.get("Driving time") is not None]
    return sum(times) / len(times) if times else None

def compute_all_stats(households) -> dict:
    """
    Runs every registered stat against the provided household list.
    Individual failures are caught and returned as error entries rather
    than crashing the entire response.
    """
    results = {}
    for key, compute_fn in STAT_REGISTRY.items():
        try:
            results[key] = compute_fn(households)
        except Exception as e:
            logging.error(f"Error computing stat '{key}': {e}")
            results[key] = {"stat": key, "error": str(e)}
    return results

def _make_stat(key: str, label: str, unit: str, fn, **kwargs):
    """
    Wraps a stat function so it returns a standardised response dict:
        {
            "stat":  <key>,
            "label": <human-readable label>,
            "value": <float | None>,
            "unit":  <unit string>,
            **kwargs  # any extra metadata, e.g. threshold
        }
    kwargs are forwarded to the underlying function only if it accepts them,
    and are also included in the response dict for transparency.
    """
    def compute(households) -> dict:
        # Only forward kwargs the function actually declares as parameters
        sig = inspect.signature(fn)
        accepted_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        raw_value = fn(households, **accepted_kwargs)
 
        result = {
            "stat":  key,
            "label": label,
            "value": round(raw_value, 4) if raw_value is not None else None,
            "unit":  unit,
        }
        # Attach any extra metadata (e.g. threshold) to the response
        result.update(kwargs)
        return result
 
    return compute

STAT_REGISTRY: dict[str, Any] = {
    "mean_food_score": _make_stat(
        key="mean_food_score",
        label="Mean Food Access Score (MFAI)",
        unit="score",
        fn=mean_food_score,
    ),
    "stddev_food_score": _make_stat(
        key="stddev_food_score",
        label="Std Dev — Food Access Score (MFAI)",
        unit="score",
        fn=stddev_food_score,
    ),
    "pct_low_food_access": _make_stat(
        key="pct_low_food_access",
        label="% Low Food Access Households",
        unit="%",
        fn=pct_low_food_access_households,
        threshold=LOW_FOOD_ACCESS_THRESHOLD,
    ),
    "mean_distance_to_nearest_supermarket": _make_stat(
        key="mean_distance_to_nearest_supermarket",
        label="Mean Distance to Nearest Supermarket",
        unit="miles",
        fn=mean_distance_to_nearest_supermarket,
    ),
    "stddev_distance_to_nearest_supermarket": _make_stat(
        key="stddev_distance_to_nearest_supermarket",
        label="Std Dev — Distance to Nearest Supermarket",
        unit="miles",
        fn=stddev_distance_to_nearest_supermarket,
    ),
    "mean_travel_time_to_nearest_supermarket": _make_stat(
        key="mean_travel_time_to_nearest_supermarket",
        label="Mean Driving Time to Nearest Supermarket",
        unit="minutes",
        fn=mean_travel_time_to_nearest_supermarket,
    ),
}