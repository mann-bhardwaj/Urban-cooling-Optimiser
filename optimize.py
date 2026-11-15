"""Run the optimizer for ANY budget.

    python optimize.py --budget "50 lakh"
    python optimize.py --budget "5 cr" --survival 0.8 --cooling high
    python optimize.py --budget 25000000 --method greedy
"""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent/"src"))
from config import PARAMS, COOLING_RANGE, OUT, ZONES
from optimizer import optimise, parse_budget, fmt_rs

ap = argparse.ArgumentParser()
ap.add_argument("--budget", required=True, help='e.g. "50 lakh", "1 cr", "10 crore", 25000000')
ap.add_argument("--method", default="ilp", choices=["ilp", "greedy"])
ap.add_argument("--survival", type=float, default=PARAMS["survival"])
ap.add_argument("--cooling", default="mid", choices=list(COOLING_RANGE))
ap.add_argument("--cost-mult", type=float, default=1.0)
a = ap.parse_args()
p = dict(PARAMS, survival=a.survival, cooling_per_10pp=COOLING_RANGE[a.cooling], cost_mult=a.cost_mult)
b = parse_budget(a.budget); t0 = time.time()
plan, kpi, _, _ = optimise(b, p, a.method)
name = f"plan_{int(b)}.csv"; plan.to_csv(OUT/name, index=False)
print(f"\nBudget {fmt_rs(b)}  ({a.method.upper()}, solved in {time.time()-t0:.1f}s)   [SIMULATED DATA]")
print(f"  spent                {fmt_rs(kpi['spent'])}")
print(f"  sites / trees        {kpi['sites']:,} cells / {kpi['trees']:,} trees")
print(f"  exposure reduction   {kpi['exposure_reduction']:,.0f} person-°C  ({kpi['pct_of_city_exposure']:.3f}% of city total)")
print(f"  people in treated cells {kpi['people_reached']:,.0f};  mean cooling {kpi['mean_cooling']:.2f} °C")
print(f"  share of spend in priority cells {100*kpi['priority_share']:.0f}%")
print("\n  Spend by zone:")
z = plan.groupby("zone").agg(sites=("cell_id", "size"), trees=("trees", "sum"), cost=("cost_rs", "sum")).sort_values("cost", ascending=False)
for name_, r in z.iterrows(): print(f"    {name_:14s} {int(r.sites):5d} sites {int(r.trees):7,d} trees  {fmt_rs(r.cost)}")
print(f"\n  Top sites:\n{plan.head(8).to_string(index=False)}\n\n  Full plan saved to outputs/{name}")
