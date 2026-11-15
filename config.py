"""Shared settings and assumptions for the Bengaluru Urban Cooling Optimizer prototype.

IMPORTANT: the data cube in this prototype is SIMULATED (see datacube.py). Every number
below marked ASSUMPTION is a placeholder to be calibrated with real data / field sensors.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

SEED = 42
N = 300                 # grid is N x N cells
CELL_M = 100            # 100 m cells
LAT0, LON0 = 12.9716, 77.5946   # grid centre (central Bengaluru)
KM_PER_DEG_LAT = 110.57
KM_PER_DEG_LON = 108.50         # at ~13 deg N

ZONES = ["East", "West", "South", "Mahadevapura", "Yelahanka",
         "Dasarahalli", "RR Nagar", "Bommanahalli"]

# ---- Heat model -----------------------------------------------------------
FEATURES = ["ndvi", "ndbi", "built", "canopy", "water", "road", "elev",
            "dist_water_km", "canopy_500m", "built_500m", "ndvi_500m", "pop"]
BLOCK_CELLS = 30        # 3 km blocks for spatial cross-validation
N_FOLDS = 5

# ---- Planting / cooling assumptions (ALL ASSUMPTIONS) ---------------------
PARAMS = dict(
    crown_m2=25.0,          # mature crown area per surviving tree
    spacing_m2=36.0,        # planting space needed per tree (6 m x 6 m)
    survival=0.60,          # share of saplings that survive to maturity
    cooling_per_10pp=0.60,  # deg C of LST reduction per +10 percentage points of canopy
    threshold=0.40,         # canopy share above which cooling strengthens (cf. ref [4])
    threshold_boost=0.50,   # extra cooling (x1.5) for canopy gained above the threshold
    cost_mult=1.0,          # multiplier on all costs
    site_fixed_cost=20000,  # Rs fixed mobilisation cost per 100 m cell that gets any planting
    equity_share=0.30,      # at least this share of spend goes to priority (vulnerable) cells
    zone_cap=0.35,          # no zone receives more than this share of the budget
)
COOLING_RANGE = {"low": 0.30, "mid": 0.60, "high": 1.00}   # deg C per +10 pp canopy
LEVELS = (1/3, 2/3, 1.0)   # planting intensity options per cell (share of capacity)
