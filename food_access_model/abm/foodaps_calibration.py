"""
Household behavior parameters calibrated to USDA's FoodAPS-1 survey.

Source data
    USDA ERS National Household Food Acquisition and Purchase Survey (FoodAPS-1),
    public-use files, fielded April 2012 - January 2013; 4,826 households; one week
    of food acquisitions per household. Distances in the FoodAPS access file are
    straight-line miles, matching this model's Euclidean distances.

Estimation
    scripts/foodaps/estimate_abm_params.py. All models are survey-weighted (HHWGT);
    standard errors use the 57 jackknife replicate weights (FoodAPS User Guide,
    Appendix D). Only covariates that household agents carry are used.

    SPM  = supermarket or superstore (SNAP STARS types SM/SS; FoodAPS place types 121/122)
    CSPM = nearest other food store (combination grocery, convenience, small-to-large grocery)

The module is pure Python so it can be imported and tested without mesa or a database.
"""
import math
import os

# --------------------------------------------------------------------------------------
# HHS poverty guidelines, 48 contiguous states + DC: (one person, each additional person)
# Sources: ASPE annual poverty guidelines (2026: $15,960 + $5,680; 2025: $15,650 + $5,500).
POVERTY_GUIDELINES = {
    2022: (13590, 4720),
    2023: (14580, 5140),
    2024: (15060, 5380),
    2025: (15650, 5500),
    2026: (15960, 5680),
}

# Household is "low income" below 130% of the poverty guideline (SNAP gross-income limit).
LOW_INCOME_POVERTY_RATIO = 1.30

# Model A. Trip-level logit: P(a food-store trip is made at an SPM rather than a CSPM).
#   n = 14,774 trips; jackknife SE in comments.
TRIP_CHOICE_LOGIT = {
    "const": 0.4450,        # (0.149)
    "log_spm_dist": -0.2711,  # (0.067)  log miles to nearest SPM
    "log_cspm_dist": 0.1461,  # (0.055)  log miles to nearest CSPM
    "has_vehicle": 0.6239,    # (0.121)
    "log_poverty_ratio": 0.0870,  # (0.047)
    "household_size": 0.0378,     # (0.026)
}

# Model B. Food-store acquisition trips per month (weekly FoodAPS events x 30.44/7),
#   survey-weighted means by (has_vehicle, low_income). Differences are small.
MONTHLY_TRIPS = {
    (True, False): 13,   # 12.86 (SE 0.31)
    (True, True): 13,    # 12.83 (SE 0.50)
    (False, False): 12,  # 12.06 (SE 0.91)
    (False, True): 12,   # 11.90 (SE 0.70)
}

# Model C. Household logit: P(liquid assets < $2,000). Agents have no asset data, so the
#   probability is imputed from attributes they do have. n = 4,731.
LOW_ASSETS_LOGIT = {
    "const": 1.7935,             # (0.210)
    "log_poverty_ratio": -1.6701,  # (0.128)
    "household_size": -0.0132,     # (0.046)
    "has_vehicle": -0.7195,        # (0.208)
    "workers": 0.3474,             # (0.094)
}

# Model D. Household logit: P(low or very low adult food security). n = 4,731.
#   Store-access terms are deliberately excluded: in FoodAPS, supermarket distance has no
#   association with food insecurity once income and assets are controlled, and the
#   convenience-store association is not credibly causal. Adding or removing stores
#   therefore does not change this probability; it changes only with household resources.
FOOD_INSECURITY_LOGIT = {
    "const": -2.4524,            # (0.185)
    "log_poverty_ratio": -0.7442,  # (0.066)
    "low_assets": 2.3166,          # (0.171)
    "has_vehicle": -0.3386,        # (0.150)
    "household_size": 0.0100,      # (0.040)
    "workers": -0.0199,            # (0.079)
}

# Survey-weighted FoodAPS benchmarks (national, 2012-13) for validation.
VALIDATION_TARGETS = {
    "spm_trip_share_all": 0.747,
    "spm_trip_share_vehicle": 0.763,
    "spm_trip_share_no_vehicle": 0.611,
    "food_insecure_share": 0.159,
    "low_assets_share": 0.458,
    "share_not_using_nearest_spm_as_primary": 0.717,
    "median_dist_nearest_spm_mi": 0.81,
    "median_dist_nearest_cspm_mi": 0.40,
}

_MIN_DIST = 0.05   # miles; FoodAPS distances were clipped the same way before logging
_POV_BOUNDS = (0.1, 10.0)


def _logistic(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def model_year() -> int:
    """Simulation year from the YEAR environment variable (default 2022, as in preprocessing)."""
    try:
        return int(os.getenv("YEAR", 2022))
    except ValueError:
        return 2022


def poverty_guideline(household_size: int, year: int = None) -> float:
    """Annual HHS poverty guideline for a household of the given size."""
    year = model_year() if year is None else year
    if year not in POVERTY_GUIDELINES:
        year = min(POVERTY_GUIDELINES, key=lambda y: abs(y - year))
    base, extra = POVERTY_GUIDELINES[year]
    size = max(1, int(household_size or 1))
    return base + extra * (size - 1)


def poverty_ratio(income: float, household_size: int, year: int = None) -> float:
    """Household income as a multiple of the poverty guideline, bounded to [0.1, 10]."""
    ratio = float(income or 0) / poverty_guideline(household_size, year)
    return min(max(ratio, _POV_BOUNDS[0]), _POV_BOUNDS[1])


def is_low_income(pov_ratio: float) -> bool:
    return pov_ratio < LOW_INCOME_POVERTY_RATIO


def prob_spm_trip(spm_dist: float, cspm_dist: float, has_vehicle: bool,
                  pov_ratio: float, household_size: int) -> float:
    """Probability that a food-store trip goes to the supermarket rather than the other store."""
    spm_dist, cspm_dist, household_size = float(spm_dist), float(cspm_dist), float(household_size)
    b = TRIP_CHOICE_LOGIT
    z = (b["const"]
         + b["log_spm_dist"] * math.log(max(spm_dist, _MIN_DIST))
         + b["log_cspm_dist"] * math.log(max(cspm_dist, _MIN_DIST))
         + b["has_vehicle"] * float(has_vehicle)
         + b["log_poverty_ratio"] * math.log(pov_ratio)
         + b["household_size"] * household_size)
    return _logistic(z)


def monthly_trips(has_vehicle: bool, pov_ratio: float) -> int:
    return MONTHLY_TRIPS[(bool(has_vehicle), is_low_income(pov_ratio))]


def prob_low_assets(pov_ratio: float, household_size: int, has_vehicle: bool,
                    workers: int) -> float:
    """P(liquid assets < $2,000) imputed from income, size, vehicle and workers."""
    household_size, workers = float(household_size), float(workers or 0)
    b = LOW_ASSETS_LOGIT
    z = (b["const"]
         + b["log_poverty_ratio"] * math.log(pov_ratio)
         + b["household_size"] * household_size
         + b["has_vehicle"] * float(has_vehicle)
         + b["workers"] * workers)
    return _logistic(z)


def prob_food_insecure(pov_ratio: float, household_size: int, has_vehicle: bool,
                       workers: int) -> float:
    """P(low/very low adult food security), integrating over imputed asset status.

    Returns an expected value rather than a random draw so the result is stable across
    steps (household agents are rebuilt from stored records each step).
    """
    household_size, workers = float(household_size), float(workers or 0)
    p_low = prob_low_assets(pov_ratio, household_size, has_vehicle, workers)
    b = FOOD_INSECURITY_LOGIT
    base = (b["const"]
            + b["log_poverty_ratio"] * math.log(pov_ratio)
            + b["has_vehicle"] * float(has_vehicle)
            + b["household_size"] * household_size
            + b["workers"] * workers)
    return p_low * _logistic(base + b["low_assets"]) + (1 - p_low) * _logistic(base)
