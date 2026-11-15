"""Step 2 - land-surface-temperature model with spatial block cross-validation,
then the heat-exposure index  E = H x P x V."""
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
from config import *

def models():
    return {
        "Linear Regression": lambda: LinearRegression(),
        "Random Forest": lambda: RandomForestRegressor(n_estimators=80, max_depth=14, min_samples_leaf=5,
                                                       max_samples=0.3, n_jobs=-1, random_state=SEED),
        "XGBoost": lambda: XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.06, subsample=0.8,
                                        colsample_bytree=0.8, tree_method="hist", n_jobs=2, random_state=SEED),
    }

def cv_scores(make, X, y, splits):
    oof = np.zeros(len(y))
    for tr, te in splits:
        m = make(); m.fit(X[tr], y[tr]); oof[te] = m.predict(X[te])
    return r2_score(y, oof), float(np.sqrt(mean_squared_error(y, oof))), oof

def run():
    df = pd.read_csv(DATA/"datacube.csv")
    X, y = df[FEATURES].to_numpy(), df["lst"].to_numpy()
    random_splits = list(KFold(N_FOLDS, shuffle=True, random_state=SEED).split(X))
    spatial_splits = list(GroupKFold(N_FOLDS).split(X, y, groups=df["block"]))
    metrics, oof_xgb = [], None
    for name, make in models().items():
        r2r, rmser, _ = cv_scores(make, X, y, random_splits)
        r2s, rmses, oof = cv_scores(make, X, y, spatial_splits)
        metrics.append(dict(model=name, r2_random=r2r, rmse_random=rmser, r2_spatial=r2s, rmse_spatial=rmses))
        print(f"{name:18s} random CV R2={r2r:.3f} RMSE={rmser:.2f} | spatial CV R2={r2s:.3f} RMSE={rmses:.2f}")
        if name == "XGBoost": oof_xgb = oof
    final = models()["XGBoost"](); final.fit(X, y)
    df["lst_pred"] = final.predict(X)
    imp = dict(sorted(zip(FEATURES, map(float, final.feature_importances_)), key=lambda t: -t[1]))

    # ---- heat-exposure index: E = H x P x V -------------------------------
    base = float(np.percentile(df["lst_pred"], 10))        # "cool" reference temperature
    df["heat_excess"] = np.clip(df["lst_pred"] - base, 0, None)      # H, deg C above reference
    df["vuln_weight"] = 1 + df["vuln"]                               # V, between 1 and 2
    df["exposure"] = df["heat_excess"] * df["pop"] * df["vuln_weight"]   # weighted person-degC
    df.round(5).to_csv(DATA/"datacube_scored.csv", index=False)
    json.dump(dict(metrics=metrics, importance=imp, lst_reference=base, n_cells=len(df),
                   total_exposure=float(df["exposure"].sum())), open(OUT/"model_metrics.json", "w"), indent=1)
    pd.DataFrame(metrics).round(3).to_csv(OUT/"model_metrics.csv", index=False)

    # ---- figures ----------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    names = [m["model"] for m in metrics]; xs = np.arange(len(names))
    ax[0].bar(xs-0.2, [m["r2_random"] for m in metrics], 0.4, label="Random split CV", color="#9aa5b1")
    ax[0].bar(xs+0.2, [m["r2_spatial"] for m in metrics], 0.4, label="Spatial block CV", color="#c2410c")
    ax[0].set_xticks(xs, names); ax[0].set_ylabel("R²"); ax[0].set_title("Random splits overstate accuracy"); ax[0].legend()
    s = np.random.default_rng(0).choice(len(y), 6000, replace=False)
    ax[1].scatter(y[s], oof_xgb[s], s=3, alpha=0.3, color="#c2410c"); lim = [y.min(), y.max()]
    ax[1].plot(lim, lim, "k--", lw=1); ax[1].set_xlabel("Observed LST (°C)"); ax[1].set_ylabel("Predicted (spatial CV)")
    ax[1].set_title("XGBoost, held-out spatial blocks")
    k = list(imp)[:8][::-1]; ax[2].barh(k, [imp[i] for i in k], color="#0f766e"); ax[2].set_title("XGBoost feature importance")
    fig.suptitle("Heat model (SIMULATED data)"); fig.tight_layout(); fig.savefig(OUT/"fig_heat_model.png", dpi=130)
    return df

if __name__ == "__main__":
    run()
