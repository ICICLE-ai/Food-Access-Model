"""
Estimate FoodAPS-1 parameters for the FEAST / Food-Access-Model household agents.

Only covariates the ABM agents carry (or can derive) are used:
  income (-> income-to-poverty ratio), household size, vehicles (any), workers,
  straight-line distance to nearest supermarket/superstore (SPM) and to the nearest
  other food store (CSPM = min distance to combination grocery, convenience,
  medium/large grocery).

Models (survey-weighted; jackknife SEs from 57 replicate weights, FoodAPS UG App. D):
  A. Trip-level logit: P(food-store trip is at a SPM)
  B. Monthly food-store trips by vehicle x low-income (<130% FPL)
  C. Household logit: P(liquid assets < $2,000)
  D. Household logit: P(low/very low adult food security)

Usage: python estimate_abm_params.py <CSV_data_files dir> <out dir>
Writes foodaps_abm_params.json and foodaps_abm_estimates.csv
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")
SRC, OUT = sys.argv[1], sys.argv[2]
REPW = [f"hhwgt{i}" for i in range(1, 58)]
JK = np.array([0.5] * 20 + [2 / 3] * 3 + [0.5] * 16 + [0.875] * 8 + [0.5] * 10)
DF = 32
LOW_INCOME_CUTOFF = 1.30  # SNAP gross-income limit, share of poverty guideline


def build():
    h = pd.read_csv(f"{SRC}/faps_household_puf.csv", low_memory=False)
    a = pd.read_csv(f"{SRC}/faps_access_puf.csv")
    w = pd.read_csv(f"{SRC}/faps_hhweights.csv")[["hhnum"] + REPW]
    ind = pd.read_csv(f"{SRC}/faps_individual_puf.csv", low_memory=False)
    ev = pd.read_csv(f"{SRC}/faps_fahevent_puf.csv", low_memory=False)
    ind.columns = ind.columns.str.lower()
    for c in ["employment", "age_r"]:
        ind[c] = pd.to_numeric(ind[c], errors="coerce")
    ind = ind[ind.guest.astype(str) != "1"]

    d = h.merge(a, on="hhnum").merge(w, on="hhnum")
    d = d[d.anyvehicle.isin([0, 1])].copy()
    d["has_vehicle"] = d.anyvehicle.astype(float)
    d["workers"] = d.hhnum.map(ind[ind.employment.isin([1, 2])].groupby("hhnum").size()).fillna(0)
    d["pov_ratio"] = (d.pctpovguidehh_r / 100).clip(0.1, 10)
    d["log_pov"] = np.log(d.pov_ratio)
    d["low_income"] = (d.pov_ratio < LOW_INCOME_CUTOFF).astype(float)
    d["spm_dist"] = d.nearsmss_dist.clip(lower=0.05)
    d["cspm_dist"] = d[["dist_co", "dist_cs", "dist_mlg"]].min(axis=1).clip(lower=0.05)
    d["log_spm"] = np.log(d.spm_dist)
    d["log_cspm"] = np.log(d.cspm_dist)
    d["spm_far5"] = (d.spm_dist > 5).astype(float)
    d["veh_x_log_spm"] = d.has_vehicle * d.log_spm
    d["low_assets"] = (d.liqassets == 1).astype(float).where(d.liqassets > 0)
    d["food_insecure"] = d.adltfscat.isin([3, 4]).astype(float).where(d.adltfscat > 0)

    ev = ev[(ev.placecateg == 1) & ev.totalpaid.notna()].copy()
    ev["spm_trip"] = (ev.placesnaptype.isin(["SM", "SS"]) | ev.placetype.isin([121, 122])).astype(float)
    d["trips_wk"] = d.hhnum.map(ev.groupby("hhnum").size()).fillna(0)
    d["trips_mo"] = d.trips_wk * 30.44 / 7
    trips = ev[["hhnum", "spm_trip"]].merge(d, on="hhnum")
    return d, trips


def jk_logit(df, y, xs, name):
    s = df[[y] + xs + ["hhwgt"] + REPW].dropna()
    X = sm.add_constant(s[xs])

    def fit(wcol):
        return sm.GLM(s[y], X, family=sm.families.Binomial(), var_weights=s[wcol]).fit().params

    b = fit("hhwgt")
    reps = np.array([fit(r).values for r in REPW])
    se = np.sqrt((JK[:, None] * (reps - b.values) ** 2).sum(0))
    p = 2 * stats.t.sf(np.abs(b.values / se), DF)
    tab = pd.DataFrame({"model": name, "term": b.index, "coef": b.values, "se_jackknife": se,
                        "p_value": p, "n": len(s)})
    return b, tab


def main():
    d, trips = build()
    out, tabs = {}, []

    # A. trip-level store-type choice (ABM spec) + richer robustness spec
    xa = ["log_spm", "log_cspm", "has_vehicle", "log_pov", "hhsize"]
    b, t = jk_logit(trips, "spm_trip", xa, "A_trip_choice")
    tabs.append(t)
    out["trip_choice_logit"] = b.to_dict()
    xa2 = xa + ["spm_far5", "veh_x_log_spm"]
    _, t = jk_logit(trips, "spm_trip", xa2, "A2_trip_choice_robust")
    tabs.append(t)

    # B. monthly trips (weighted means, jackknife SE)
    cells = {}
    for veh in (0.0, 1.0):
        for low in (0.0, 1.0):
            g = d[(d.has_vehicle == veh) & (d.low_income == low)]
            m = np.average(g.trips_mo, weights=g.hhwgt)
            reps = np.array([np.average(g.trips_mo, weights=g[r]) if g[r].sum() > 0 else m for r in REPW])
            se = np.sqrt((JK * (reps - m) ** 2).sum())
            key = f"vehicle={int(veh)},low_income={int(low)}"
            cells[key] = round(float(m), 2)
            tabs.append(pd.DataFrame([{"model": "B_monthly_trips", "term": key, "coef": m,
                                       "se_jackknife": se, "p_value": np.nan, "n": len(g)}]))
    out["monthly_trips"] = cells

    # C. low liquid assets
    xc = ["log_pov", "hhsize", "has_vehicle", "workers"]
    b, t = jk_logit(d, "low_assets", xc, "C_low_assets")
    tabs.append(t)
    out["low_assets_logit"] = b.to_dict()

    # D. food insecurity. ABM spec excludes access terms: their associations are not
    # credibly causal (convenience-store proximity proxies for neighborhood poverty), and
    # including them would make adding a store mechanically change simulated food insecurity.
    xd = ["log_pov", "low_assets", "has_vehicle", "hhsize", "workers"]
    b, t = jk_logit(d, "food_insecure", xd, "D_food_insecurity")
    tabs.append(t)
    out["food_insecurity_logit"] = b.to_dict()
    _, t = jk_logit(d, "food_insecure", xd + ["log_spm", "log_cspm"], "D2_food_insecurity_with_access")
    tabs.append(t)

    # Validation targets (weighted)
    def wavg(x, w):
        return float(np.average(x, weights=w))
    tr = trips.dropna(subset=["spm_trip"])
    out["validation_targets"] = {
        "spm_trip_share_all": wavg(tr.spm_trip, tr.hhwgt),
        "spm_trip_share_no_vehicle": wavg(tr.spm_trip[tr.has_vehicle == 0], tr.hhwgt[tr.has_vehicle == 0]),
        "spm_trip_share_vehicle": wavg(tr.spm_trip[tr.has_vehicle == 1], tr.hhwgt[tr.has_vehicle == 1]),
        "food_insecure_share": wavg(d.food_insecure.dropna(), d.hhwgt[d.food_insecure.notna()]),
        "low_assets_share": wavg(d.low_assets.dropna(), d.hhwgt[d.low_assets.notna()]),
        "median_dist_nearest_spm_mi": float(d.spm_dist.median()),
        "median_dist_nearest_cspm_mi": float(d.cspm_dist.median()),
    }
    out["meta"] = {
        "source": "USDA ERS FoodAPS-1 public-use files (2012-13), N=4,826 households",
        "distance": "straight-line miles",
        "low_income_cutoff_pov_ratio": LOW_INCOME_CUTOFF,
        "variance": "jackknife, 57 replicate weights, coefficients per FoodAPS User Guide Table D1",
    }
    json.dump(out, open(f"{OUT}/foodaps_abm_params.json", "w"), indent=2)
    est = pd.concat(tabs)
    est.to_csv(f"{OUT}/foodaps_abm_estimates.csv", index=False)
    pd.set_option("display.width", 200)
    print(est.round(4).to_string(index=False))
    print(json.dumps(out["validation_targets"], indent=1))


if __name__ == "__main__":
    main()
