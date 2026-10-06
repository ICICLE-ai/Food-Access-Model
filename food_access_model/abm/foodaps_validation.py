"""
Compare a simulated area to FoodAPS-1 national benchmarks.

Usage (after at least one model.step()):
    from food_access_model.abm.foodaps_validation import compare_to_foodaps
    print(compare_to_foodaps(model))

FoodAPS targets are national 2012-13 averages, so a single county should be close to
but not identical with them. Large gaps (for example a supermarket trip share far from
0.75) usually point to store-classification problems, such as superstores tagged as
something other than "supermarket" in OpenStreetMap.
"""
from statistics import median

from food_access_model.abm.foodaps_calibration import VALIDATION_TARGETS


def _mean(values):
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None


def compare_to_foodaps(model) -> dict:
    households = [a for a in model.schedule.agents if getattr(a, "type", None) == "household"]
    with_vehicle = [h for h in households if h.has_vehicles]
    without_vehicle = [h for h in households if not h.has_vehicles]

    spm_d, cspm_d = [], []
    for h in households:
        spm, sd = h.get_closest_spm()
        cspm, cd = h.get_closest_cspm()
        if spm is not None:
            spm_d.append(sd)
        if cspm is not None:
            cspm_d.append(cd)

    simulated = {
        "spm_trip_share_all": _mean(h.spm_trip_prob for h in households),
        "spm_trip_share_vehicle": _mean(h.spm_trip_prob for h in with_vehicle),
        "spm_trip_share_no_vehicle": _mean(h.spm_trip_prob for h in without_vehicle),
        "food_insecure_share": _mean(h.food_insecurity_prob for h in households),
        "low_assets_share": _mean(h.prob_low_assets for h in households),
        "median_dist_nearest_spm_mi": median(spm_d) if spm_d else None,
        "median_dist_nearest_cspm_mi": median(cspm_d) if cspm_d else None,
    }
    return {
        key: {"simulated": simulated.get(key), "foodaps_2012": target}
        for key, target in VALIDATION_TARGETS.items()
        if key in simulated
    }
