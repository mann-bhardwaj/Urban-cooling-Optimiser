"""Step 5 - scenario and sensitivity analysis: many budgets, four strategies, changed assumptions."""
import json, time, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from config import *
from optimizer import *

BUDGETS = ["50 lakh", "1 cr", "2 cr", "5 cr", "10 cr", "20 cr", "50 cr"]

def run():
    df = pd.read_csv(DATA/"datacube_scored.csv"); s = feasible_sites(df); tot = df["exposure"].sum()
    rows, base_level = [], None
    for B in BUDGETS:
        b = parse_budget(B)
        for name, fn in STRATEGIES.items():
            t0 = time.time(); lv = fn(s, b, PARAMS); e = evaluate(s, lv, PARAMS, tot)
            rows.append(dict(budget=B, budget_rs=b, strategy=name, seconds=round(time.time()-t0, 1), **e))
            if name == "ILP optimizer" and B == "10 cr": base_level = lv
        r = {x["strategy"]: x["exposure_reduction"] for x in rows if x["budget"] == B}
        print(f"{B:8s} ILP {r['ILP optimizer']:>10,.0f} | greedy {r['Greedy (benefit per rupee)']:>10,.0f} | "
              f"hottest {r['Hottest areas first']:>9,.0f} | low-canopy {r['Lowest canopy first']:>9,.0f}")
    sc = pd.DataFrame(rows); sc.round(3).to_csv(OUT/"scenario_results.csv", index=False)

    # sensitivity at Rs 10 crore: does the plan change when assumptions change?
    b = 1e8; sens = []
    cases = {"Base case": {}, "Survival 40%": dict(survival=0.4), "Survival 80%": dict(survival=0.8),
             "Cooling low (0.3 °C/10pp)": dict(cooling_per_10pp=0.3), "Cooling high (1.0 °C/10pp)": dict(cooling_per_10pp=1.0),
             "Costs +50%": dict(cost_mult=1.5), "Costs -25%": dict(cost_mult=0.75)}
    for name, ch in cases.items():
        p = dict(PARAMS, **ch); lv = solve_ilp(s, b, p)[0]; e = evaluate(s, lv, p, tot)
        both = ((lv > 0) & (base_level > 0)).sum(); either = ((lv > 0) | (base_level > 0)).sum()
        sens.append(dict(case=name, **e, site_overlap_with_base=both/either))
        print(f"  {name:28s} reduction {e['exposure_reduction']:>10,.0f}  trees {e['trees']:>7,d}  overlap {both/either:.0%}")
    se = pd.DataFrame(sens); se.round(3).to_csv(OUT/"sensitivity_results.csv", index=False)
    plan_table(s, base_level).to_csv(OUT/"plan_10cr_ilp.csv", index=False)

    # ---- figures ----------------------------------------------------------
    col = {"ILP optimizer": "#c2410c", "Greedy (benefit per rupee)": "#0f766e",
           "Hottest areas first": "#64748b", "Lowest canopy first": "#a8a29e"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for name in STRATEGIES:
        d = sc[sc.strategy == name]
        ax[0].plot(d.budget_rs/1e7, d.exposure_reduction/1e3, "o-", label=name, color=col[name])
    ax[0].set_xscale("log"); ax[0].set_xlabel("Budget (₹ crore, log scale)"); ax[0].set_ylabel("Exposure reduction (thousand person-°C)")
    ax[0].set_title("Benefit bought by each strategy"); ax[0].legend()
    ax[1].barh(se.case[::-1], se.exposure_reduction[::-1]/1e3, color="#c2410c")
    for y, (v, o) in enumerate(zip(se.exposure_reduction[::-1]/1e3, se.site_overlap_with_base[::-1])):
        ax[1].text(v, y, f"  {o:.0%} same sites", va="center", fontsize=8)
    ax[1].set_xlabel("Exposure reduction at ₹10 Cr (thousand person-°C)"); ax[1].set_title("Sensitivity of the ₹10 Cr plan")
    ax[1].set_xlim(0, se.exposure_reduction.max()/1e3*1.3)
    fig.suptitle("Optimizer scenarios (SIMULATED data)"); fig.tight_layout(); fig.savefig(OUT/"fig_scenarios.png", dpi=130)

    def grid(col_, sub=df): 
        g = np.full((N, N), np.nan); g[sub["row"], sub["col"]] = sub[col_]; return g
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
    for a, (c, title, cm) in zip(ax, [("lst_pred", "Predicted LST (°C)", "inferno"), ("exposure", "Heat-exposure index", "magma")]):
        im = a.imshow(grid(c), cmap=cm, vmax=np.nanpercentile(grid(c), 99)); a.set_title(title); a.axis("off"); fig.colorbar(im, ax=a, shrink=0.7)
    ax[2].imshow(grid("lst_pred"), cmap="Greys", alpha=0.45); ch = s[base_level > 0]
    ax[2].scatter(ch["col"], ch["row"], s=4, color="#15803d"); ax[2].set_title(f"₹10 Cr ILP plan: {len(ch):,} cells"); ax[2].axis("off")
    fig.suptitle("Bengaluru-like study area, 100 m grid (SIMULATED data)"); fig.tight_layout(); fig.savefig(OUT/"fig_maps.png", dpi=130)

if __name__ == "__main__":
    run()
