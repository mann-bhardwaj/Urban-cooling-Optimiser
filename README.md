# Bengaluru Urban Cooling Optimizer (prototype)

A geospatial ML and optimization pipeline that answers: **given any plantation budget, where should
trees go to remove the most human heat exposure?**

> **The data in this prototype is simulated.** Earth Engine exports were not available, so
> `src/datacube.py` generates a synthetic city laid out like Bengaluru (approximate positions of real
> lakes, parks, dense cores and industrial areas). The accuracy figures, maps and rupee allocations
> show that the method works end to end. They are not findings about the real city. Cost, survival
> and cooling values in `src/config.py` are placeholder assumptions.

**Live dashboard:** https://mann-bhardwaj.github.io/Urban-cooling-Optimiser/

## Quick start

    pip install -r requirements.txt

    # open the dashboard (no install needed, works offline)
    outputs/dashboard.html

    # run the exact ILP for any budget
    python optimize.py --budget "50 lakh"
    python optimize.py --budget "5 cr" --survival 0.8 --cooling high
    python optimize.py --budget 25000000 --method greedy

    # rebuild everything from scratch (about 10 minutes)
    python run_all.py

    # or with Docker
    docker build -t cooling-optimizer .
    docker run --rm cooling-optimizer --budget "50 lakh"

## Pipeline

| Step | File | What it does |
|---|---|---|
| 1 | `src/datacube.py` | 100 m grid (65,318 cells, 653 km²): NDVI, NDBI, built-up, canopy, water, roads, elevation, population, vulnerability, LST |
| 2 | `src/heat_model.py` | Linear Regression, Random Forest, XGBoost; random-split CV vs spatial block CV (3 km blocks); exposure index E = H × P × V |
| 3 | `src/optimizer.py` | Feasible sites (GIS exclusion rules), cooling benefit, ILP, greedy and naive baselines |
| 4 | `optimize.py` | Command line: any budget in, plan CSV out |
| 5 | `src/scenarios.py` | 7 budgets × 4 strategies, plus sensitivity to survival, cooling and cost |
| 6 | `src/build_dashboard.py` | Packs everything into `outputs/dashboard.html` |

## The optimization model

For each feasible cell *i* choose one planting level *k* (1/3, 2/3 or all of its tree capacity):

    maximise   Σ benefit[i,k] · x[i,k]
    subject to Σ cost[i,k] · x[i,k] ≤ budget                    (budget is an input)
               Σ_k x[i,k] ≤ 1                                   (one level per cell)
               spend in priority cells ≥ 30% of total spend     (equity)
               spend in any zone ≤ 35% of budget                (spread)
               x binary

- benefit = cooling (°C) × population × vulnerability weight, in weighted person-°C
- cooling = 0.6 °C per +10 points of canopy (range 0.3 to 1.0), ×1.5 for canopy gained above 40%,
  capped at the cell's heat excess
- cost = ₹20,000 fixed per cell + trees × cost per tree (about ₹1,800 to ₹3,900)
- Before solving, the ILP shortlists the best benefit-per-rupee cells worth 4× the budget (overall,
  among priority cells, and per zone). Raising this to 10× gave the same ₹10 Cr answer in testing.

## Results on the simulated data

Heat model (5-fold):

| Model | R² random | R² spatial | RMSE random | RMSE spatial |
|---|---|---|---|---|
| Linear Regression | 0.836 | 0.824 | 0.90 °C | 0.94 °C |
| Random Forest | 0.942 | 0.840 | 0.54 °C | 0.89 °C |
| XGBoost | 0.953 | 0.854 | 0.49 °C | 0.85 °C |

Exposure reduction (weighted person-°C):

| Budget | ILP | Greedy | Hottest first | Lowest canopy first |
|---|---|---|---|---|
| ₹50 lakh | 23,052 | 22,901 | 2,224 | 1,668 |
| ₹1 Cr | 43,486 | 43,324 | 4,798 | 7,411 |
| ₹5 Cr | 176,075 | 175,930 | 52,676 | 54,397 |
| ₹10 Cr | 316,333 | 316,125 | 116,097 | 102,517 |
| ₹20 Cr | 551,339 | 550,990 | 229,428 | 208,810 |
| ₹50 Cr | 1,091,282 | 1,091,224 | 537,495 | 480,419 |

Things worth saying when presenting:

- Random splits overstate XGBoost accuracy (R² 0.95 vs 0.85 under spatial CV), as reference [3] warns.
- The optimizer beats "hottest first" by 2.7× at ₹10 Cr and about 10× at ₹50 lakh, because the
  hottest cells are often not where people live or where trees fit.
- The ILP beats the benefit-per-rupee greedy by under 1%. The value comes from the objective and
  constraints, not from the solver.
- ₹10 Cr removes about 0.55% of city-wide exposure here: 30,000 trees is small against a city.
- Survival and cooling assumptions scale the benefit but barely change which sites are chosen
  (99% overlap). Cost assumptions do change the sites (68 to 76% overlap).

## Moving to real data

Replace `build()` in `src/datacube.py` with a loader that returns the same columns from Earth Engine
exports (Sentinel-2 NDVI/NDBI, Landsat LST), WorldPop and OpenStreetMap. Nothing downstream changes.
Known limits to state: the final LST map is fitted in-sample, cooling does not spill into neighbouring
cells, zones are approximate sectors, and the dashboard runs the greedy solver rather than the ILP.
