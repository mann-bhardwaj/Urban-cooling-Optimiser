"""Step 6 - pack the scored grid and candidate sites into one self-contained dashboard.html."""
import json, base64, numpy as np, pandas as pd
from config import *
from optimizer import feasible_sites

def run():
    df = pd.read_csv(DATA/"datacube_scored.csv"); s = feasible_sites(df)
    metrics = json.load(open(OUT/"model_metrics.json"))
    sc = pd.read_csv(OUT/"scenario_results.csv")
    def layer(col, lo=None, hi=None):
        v = df[col].to_numpy(); lo = float(np.percentile(v, 1) if lo is None else lo); hi = float(np.percentile(v, 99) if hi is None else hi)
        g = np.zeros((N, N), np.uint8)                      # 0 = outside study area, 1..255 = value
        g[df["row"], df["col"]] = 1 + np.round(254*np.clip((v-lo)/(hi-lo), 0, 1)).astype(int)
        return dict(lo=lo, hi=hi, b64=base64.b64encode(g.tobytes()).decode())
    zg = np.full((N, N), 255, np.uint8); zg[df["row"], df["col"]] = df["zone"]
    check = []
    for B, g in sc.groupby("budget", sort=False):
        r = g.set_index("strategy")["exposure_reduction"]
        check.append(dict(budget=B, rs=float(g["budget_rs"].iloc[0]), ilp=float(r["ILP optimizer"]), greedy=float(r["Greedy (benefit per rupee)"]),
                          hottest=float(r["Hottest areas first"]), lowcanopy=float(r["Lowest canopy first"])))
    data = dict(
        N=N, lat0=LAT0, lon0=LON0, kmLat=KM_PER_DEG_LAT, kmLon=KM_PER_DEG_LON, zones=ZONES, params=PARAMS,
        coolingRange=COOLING_RANGE, levels=list(LEVELS), totalExposure=float(df["exposure"].sum()),
        totalPop=float(df["pop"].sum()), nCells=len(df), lstRef=metrics["lst_reference"],
        layers=dict(heat=layer("lst_pred"), exposure=layer("exposure", 0), canopy=layer("canopy", 0),
                    pop=layer("pop", 0), vuln=layer("vuln", 0, 1)),
        zoneGrid=base64.b64encode(zg.tobytes()).decode(),
        sites=dict(idx=(s["row"]*N + s["col"]).tolist(), cap=s["trees_cap"].tolist(),
                   can=np.round(s["canopy"]*1e4).astype(int).tolist(), w=np.round(s["pop"]*s["vuln_weight"]*10).astype(int).tolist(),
                   h=np.round(s["heat_excess"]*1e3).astype(int).tolist(), cpt=np.round(s["cost_per_tree"]).astype(int).tolist(),
                   zone=s["zone"].tolist(), pri=s["priority"].tolist(), pop=np.round(s["pop"]).astype(int).tolist()),
        metrics=metrics["metrics"], importance=metrics["importance"], check=check)
    html = open(ROOT/"src"/"dashboard_template.html", encoding="utf-8").read().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    open(OUT/"dashboard.html", "w", encoding="utf-8").write(html)
    print(f"dashboard.html written ({len(html)/1e6:.1f} MB, {len(s):,} candidate sites)")

if __name__ == "__main__":
    run()
