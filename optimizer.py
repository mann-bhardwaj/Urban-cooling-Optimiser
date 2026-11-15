"""Steps 3 and 4 - feasible planting sites, cooling benefit, and the budget optimizer.

The budget is an INPUT: optimise(budget_rupees) works for any amount.

Decision: for every feasible 100 m cell i choose one planting level k (none, 1/3, 2/3 or all of
its tree capacity).
    maximise   sum_i,k  benefit[i,k] * x[i,k]                (exposure reduction, person-degC)
    subject to sum_i,k  cost[i,k] * x[i,k] <= budget
               sum_k x[i,k] <= 1                             (one level per cell)
               spend in priority (most vulnerable) cells >= equity_share * total spend
               spend in any one zone <= zone_cap * budget
               x binary
"""
import re, numpy as np, pandas as pd, pulp
from config import *

# ---------------------------------------------------------------- budget text
def parse_budget(text):
    """'50 lakh', '5 cr', '1.5 crore', '2,50,00,000', 1e7 -> rupees (float)."""
    if isinstance(text, (int, float)): return float(text)
    s = str(text).lower().replace(",", "").replace("₹", "").replace("rs.", "").replace("rs", "").strip()
    m = re.match(r"^([0-9]*\.?[0-9]+(?:e[0-9]+)?)\s*([a-z]*)$", s)
    if not m: raise ValueError(f"cannot read budget: {text!r}")
    unit = {"": 1, "k": 1e3, "thousand": 1e3, "l": 1e5, "lac": 1e5, "lakh": 1e5, "lakhs": 1e5,
            "cr": 1e7, "crore": 1e7, "crores": 1e7}.get(m.group(2))
    if unit is None: raise ValueError(f"unknown unit in budget: {text!r}")
    return float(m.group(1)) * unit

def fmt_rs(x):
    return f"₹{x/1e7:.2f} Cr" if x >= 1e7 else f"₹{x/1e5:.2f} lakh"

# ---------------------------------------------------------------- feasible sites
def feasible_sites(df):
    """GIS exclusion rules -> candidate cells with a tree capacity and a cost per tree (ASSUMPTIONS)."""
    plantable_m2 = (0.35*df["open"] + 0.10*df["road"]) * CELL_M**2      # usable open land + road verges
    ok = (df["water"] < 0.30) & (df["built"] < 0.75) & (df["canopy"] < 0.50) & (plantable_m2 >= 800) & (df["pop"] >= 10)
    s = df[ok].copy()
    s["trees_cap"] = np.floor(plantable_m2[ok] / PARAMS["spacing_m2"]).astype(int)
    s["cost_per_tree"] = 1800*(1 + 0.9*s["built"]) + 600*(s["road"] > 0.2)   # Rs incl. guard + 3 yr care
    cut = df.loc[df["pop"] > 20, "vuln"].quantile(0.70)
    s["priority"] = (s["vuln"] >= cut).astype(int)
    return s[s["trees_cap"] > 0].reset_index(drop=True)

def options(s, p=PARAMS):
    """cost[n,3], benefit[n,3], trees[n,3], cooling[n,3] for the three planting levels."""
    cap = s["trees_cap"].to_numpy()[:, None]
    trees = np.maximum(np.round(cap * np.array(LEVELS)[None, :]), 1)
    c0 = s["canopy"].to_numpy()[:, None]
    c1 = np.minimum(c0 + trees * p["survival"] * p["crown_m2"] / CELL_M**2, 0.95)
    th, bo = p["threshold"], p["threshold_boost"]
    gain = (c1 - c0) + bo*(np.clip(c1 - th, 0, None) - np.clip(c0 - th, 0, None))
    cool = np.minimum(p["cooling_per_10pp"] * gain / 0.10, s["heat_excess"].to_numpy()[:, None])
    ben = cool * (s["pop"] * s["vuln_weight"]).to_numpy()[:, None]
    cost = p["cost_mult"] * (p["site_fixed_cost"] + trees * s["cost_per_tree"].to_numpy()[:, None])
    return cost, ben, trees, cool

# ---------------------------------------------------------------- exact ILP
def shortlist(cost, ben, pri, zone, budget, factor):
    """Keep only cells that could plausibly enter the plan: the best benefit-per-rupee cells worth
    `factor` x the budget overall, and again within priority cells and within each zone.
    (Cells far down the ranking can never be worth choosing; this keeps the ILP small and fast.)"""
    ratio = (ben/cost).max(1); full = cost[:, 2]; keep = np.zeros(len(ratio), bool)
    def top(mask, amount):
        idx = np.where(mask)[0]; idx = idx[np.argsort(-ratio[idx])]
        m = np.searchsorted(np.cumsum(full[idx]), amount) + 1
        keep[idx[:max(m, 300)]] = True
    top(np.ones(len(ratio), bool), factor*budget)
    top(pri == 1, factor*budget)
    for z in range(len(ZONES)): top(zone == z, factor*0.35*budget)
    return np.where(keep & (ratio > 0))[0]

def solve_ilp(s, budget, p=PARAMS, time_limit=90, gap=0.0002, factor=4):
    cost, ben, trees, cool = options(s, p)
    zone_all = s["zone"].to_numpy(); pri_all = s["priority"].to_numpy()
    keep = shortlist(cost, ben, pri_all, zone_all, budget, factor)
    c, b, zone, pri = cost[keep], ben[keep], zone_all[keep], pri_all[keep]
    n = len(keep); scale = b.max() or 1.0
    prob = pulp.LpProblem("urban_cooling", pulp.LpMaximize)
    x = [[pulp.LpVariable(f"x_{i}_{k}", cat="Binary") for k in range(3)] for i in range(n)]
    flat = lambda coef, rows: pulp.LpAffineExpression([(x[i][k], float(coef[i, k])) for i in rows for k in range(3)])
    every = range(n)
    prob += flat(b/scale, every)                                   # objective
    prob += flat(c, every) <= budget                               # budget
    for i in every: prob += x[i][0] + x[i][1] + x[i][2] <= 1       # one planting level per cell
    cp = c * (pri[:, None] - p["equity_share"])                    # equity: priority spend >= share of total spend
    prob += flat(cp, every) >= 0
    for z in range(len(ZONES)):                                    # zone cap
        rows = np.where(zone == z)[0]
        if len(rows): prob += flat(c, rows) <= p["zone_cap"] * budget
    avail = pulp.listSolvers(onlyAvailable=True)          # HiGHS if installed, otherwise bundled CBC
    solver = (pulp.HiGHS if "HiGHS" in avail else pulp.PULP_CBC_CMD)(msg=False, timeLimit=time_limit, gapRel=gap)
    prob.solve(solver)
    level = np.zeros(len(s), dtype=int)
    for i in every:
        for k in range(3):
            v = x[i][k].value()
            if v is not None and v > 0.5: level[keep[i]] = k + 1
    return level, pulp.LpStatus[prob.status]

# ---------------------------------------------------------------- heuristics / baselines
def solve_greedy(s, budget, p=PARAMS):
    """Benefit-per-rupee greedy that respects the same equity and zone rules (fast baseline;
    the dashboard runs this same procedure in the browser)."""
    cost, ben, _, _ = options(s, p)
    n = len(s); zone = s["zone"].to_numpy(); pri = s["priority"].to_numpy()
    ratio = ben / cost; best = ratio.argmax(1); order = np.argsort(-ratio.max(1))
    level = np.zeros(n, dtype=int); zspend = np.zeros(len(ZONES)); spent = 0.0
    def fill(limit, only_priority):
        nonlocal spent
        for i in order:
            if level[i] or (only_priority and not pri[i]): continue
            if ratio[i, best[i]] <= 0: break
            for k in (best[i], 1, 0):                       # best option, then smaller ones
                c = cost[i, k]
                if spent + c <= limit and zspend[zone[i]] + c <= p["zone_cap"] * budget and ben[i, k] > 0:
                    level[i] = k + 1; spent += c; zspend[zone[i]] += c; break
            if limit - spent < cost.min(): break
    fill(p["equity_share"] * budget, True)     # reserve the equity share for priority cells
    fill(budget, False)
    return level

def solve_ranked(s, budget, key, ascending, p=PARAMS):
    """Naive strategies: plant fully in cells ranked by one attribute until money runs out."""
    cost, _, _, _ = options(s, p)
    level = np.zeros(len(s), dtype=int); spent = 0.0
    for i in s[key].to_numpy().argsort()[::1 if ascending else -1]:
        if spent + cost[i, 2] <= budget: level[i] = 3; spent += cost[i, 2]
        elif budget - spent < cost.min(): break
    return level

STRATEGIES = {
    "ILP optimizer": lambda s, b, p: solve_ilp(s, b, p)[0],
    "Greedy (benefit per rupee)": solve_greedy,
    "Hottest areas first": lambda s, b, p: solve_ranked(s, b, "lst_pred", False, p),
    "Lowest canopy first": lambda s, b, p: solve_ranked(s, b, "canopy", True, p),
}

# ---------------------------------------------------------------- reporting
def evaluate(s, level, p=PARAMS, total_exposure=None):
    cost, ben, trees, cool = options(s, p)
    sel = level > 0; k = np.clip(level - 1, 0, 2); r = np.arange(len(s))
    c, b, t = cost[r, k]*sel, ben[r, k]*sel, trees[r, k]*sel
    out = dict(sites=int(sel.sum()), trees=int(t.sum()), spent=float(c.sum()), exposure_reduction=float(b.sum()),
               people_reached=float((s["pop"]*sel).sum()),
               priority_share=float(c[s["priority"].to_numpy() == 1].sum()/max(c.sum(), 1)),
               mean_cooling=float((cool[r, k]*sel).sum()/max(sel.sum(), 1)))
    if total_exposure: out["pct_of_city_exposure"] = 100*out["exposure_reduction"]/total_exposure
    return out

def plan_table(s, level, p=PARAMS):
    cost, ben, trees, cool = options(s, p)
    sel = np.where(level > 0)[0]; k = level[sel] - 1
    t = s.iloc[sel][["cell_id", "lat", "lon", "zone", "priority", "lst_pred", "canopy", "pop"]].copy()
    t["zone"] = t["zone"].map(lambda z: ZONES[z])
    t["trees"] = trees[sel, k].astype(int); t["cost_rs"] = cost[sel, k].round(0)
    t["cooling_degC"] = cool[sel, k].round(3); t["exposure_reduction"] = ben[sel, k].round(1)
    return t.sort_values("exposure_reduction", ascending=False).reset_index(drop=True)

def optimise(budget, p=PARAMS, method="ilp"):
    """One call: any budget in, plan out."""
    df = pd.read_csv(DATA/"datacube_scored.csv"); s = feasible_sites(df)
    b = parse_budget(budget)
    level = solve_ilp(s, b, p)[0] if method == "ilp" else solve_greedy(s, b, p)
    return plan_table(s, level, p), evaluate(s, level, p, df["exposure"].sum()), s, level
