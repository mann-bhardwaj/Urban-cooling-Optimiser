"""Step 1 - build the 100 m spatial data cube.

SIMULATED DATA. Earth Engine exports (Sentinel-2 NDVI/NDBI, Landsat LST, WorldPop, OSM)
are not available in this prototype, so this module generates a synthetic city laid out
like Bengaluru (approximate positions of real lakes, parks, dense cores and industrial
areas). To switch to real data, replace build() with a loader that returns a table with
the same columns - nothing downstream needs to change.
"""
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter, uniform_filter, distance_transform_edt
from config import *

def build():
    rng = np.random.default_rng(SEED)
    ii, jj = np.mgrid[0:N, 0:N]
    xk = (jj - N/2 + 0.5) * CELL_M/1000          # km east of centre
    yk = (N/2 - ii - 0.5) * CELL_M/1000          # km north of centre
    r = np.hypot(xk, yk); th = np.arctan2(yk, xk)

    def grf(sigma):                               # smooth random field, unit variance
        f = gaussian_filter(rng.standard_normal((N, N)), sigma, mode="reflect")
        return (f - f.mean())/f.std()
    def km(lat, lon):
        return (lon-LON0)*KM_PER_DEG_LON, (lat-LAT0)*KM_PER_DEG_LAT
    def bump(lat, lon, rx, ry=None):
        ry = ry or rx; cx, cy = km(lat, lon)
        return np.exp(-(((xk-cx)/rx)**2 + ((yk-cy)/ry)**2))
    def blob(lat, lon, rx, ry=None):
        ry = ry or rx; cx, cy = km(lat, lon)
        return ((((xk-cx)/rx)**2 + ((yk-cy)/ry)**2) < 1).astype(float)

    # study area: irregular city boundary
    R = 14.5 + 1.8*np.sin(2*th+0.6) + 1.1*np.sin(3*th-1.0) + 0.7*np.sin(5*th)
    inside = r < R

    # --- water (approximate positions of real lakes) ---
    lakes = [(12.935,77.675,1.3,0.7),(12.945,77.735,1.0,0.5),(12.983,77.620,0.45,0.35),
             (13.046,77.586,0.4,0.3),(13.009,77.573,0.2,0.25),(12.906,77.616,0.6,0.4),
             (12.921,77.641,0.35,0.25),(13.100,77.590,0.45,0.4),(13.045,77.610,0.3,0.3),
             (12.915,77.675,0.25,0.25),(13.085,77.610,0.5,0.35),(13.060,77.640,0.4,0.3),
             (12.890,77.560,0.3,0.25),(12.960,77.510,0.35,0.3),(13.010,77.680,0.3,0.3)]
    wmask = np.zeros((N, N))
    for la, lo, rx, ry in lakes: wmask = np.maximum(wmask, blob(la, lo, rx, ry))
    water = np.clip(gaussian_filter(wmask, 0.7), 0, 1)

    # --- urban structure ---
    cores = bump(12.972,77.575,2.2) + 0.8*bump(12.985,77.605,1.8) + 0.6*bump(12.930,77.580,2.5) \
          + 0.6*bump(12.995,77.555,2.0) + 0.5*bump(12.935,77.625,2.0) + 0.5*bump(13.000,77.640,2.0)
    industrial = np.clip(bump(13.030,77.520,1.8,1.4) + bump(12.985,77.735,1.8,1.3)
                         + bump(12.850,77.665,1.6,1.3), 0, 1)
    u = np.clip(0.75*np.exp(-(r/9.5)**2) + 0.25*np.clip(cores,0,1.5) + 0.12*grf(15), 0, 1)
    built = np.clip(0.12 + 0.62*u + 0.10*grf(6) + 0.30*industrial, 0.02, 0.95)

    parks = [(12.976,77.593,0.7),(12.950,77.585,0.6),(13.021,77.567,1.0),(13.077,77.578,1.6),
             (12.885,77.527,1.0),(13.070,77.530,1.3),(12.998,77.592,0.6),(12.962,77.632,0.9),
             (12.870,77.590,1.2),(12.940,77.530,0.6)]
    park = np.zeros((N, N))
    for la, lo, rr in parks: park = np.maximum(park, bump(la, lo, rr))
    built *= (1 - 0.85*park)
    openland = bump(12.950,77.668,1.5,0.8)                    # old airport: bare open land
    built *= (1 - 0.8*openland)

    # --- vulnerability: dense low-income pockets with little tree cover ---
    pockets = np.clip((grf(4) - 1.1), 0, None); pockets = np.clip(pockets/pockets.max()*2.2, 0, 1) * (built > 0.35)
    vuln = np.clip(0.28 + 0.14*grf(12) + 0.55*pockets + 0.08*(r/14), 0, 1)

    canopy = np.clip(0.05 + 0.28*(1-built)*(0.55+0.45*grf(8)) + 0.62*park + 0.04*(r/14), 0, 0.92)
    canopy *= (1 - 0.6*pockets) * (1 - 0.7*openland)

    # --- roads: rings + radial highways ---
    major = (np.abs(r-4.6) < 0.06) | (np.abs(r-9.4) < 0.07)
    for deg in (133, 82, 12, -52, -88, -112, -148, 172):      # Tumkur, Bellary, OMR, Hosur, ...
        a = np.deg2rad(deg)
        along = xk*np.cos(a) + yk*np.sin(a); across = -xk*np.sin(a) + yk*np.cos(a)
        major |= (np.abs(across) < 0.06) & (along > 0)
    road = np.clip(0.04 + 0.10*built + 0.35*major, 0, 0.5)

    # make land-cover shares consistent: built + canopy + water + road + open = 1
    land = 1 - water
    tot = built + canopy + road
    scale = np.where(tot > 0.97, 0.97/tot, 1.0)
    built, canopy, road = built*scale*land, canopy*scale*land, road*scale*land
    open_ = np.clip(1 - water - built - canopy - road, 0, 1)
    bare = np.clip(0.5 + 0.35*grf(7), 0, 1)                   # share of open land that is bare soil

    # --- spectral indices (what Sentinel-2 would give us) ---
    ndvi = 0.06 + 0.66*canopy + 0.28*open_*(1-bare) - 0.25*water + rng.normal(0, 0.03, (N, N))
    ndbi = -0.22 + 0.50*built + 0.18*open_*bare - 0.30*canopy - 0.25*water + rng.normal(0, 0.03, (N, N))
    elev = 900 + 28*grf(40) + 1.6*(yk - xk)                   # m; higher towards the north-west

    # --- population (people per 1 ha cell), scaled to ~11.5 million ---
    dens = built * (30 + 520*u**1.5) * np.exp(0.30*grf(5)) * (1 - 0.65*industrial) * (1 + 0.8*pockets)
    dens *= (1 - 0.9*park) * inside
    pop = dens * 11.5e6 / dens.sum()

    # --- "observed" land-surface temperature (what Landsat would give us) ---
    k5 = lambda a: uniform_filter(a, 5, mode="nearest")
    cn, bn = k5(canopy), k5(built)
    dist_w = distance_transform_edt(water < 0.5) * CELL_M/1000
    c_eff = 0.5*canopy + 0.5*cn
    lst = (33.0 + 8.5*(0.6*built + 0.4*bn)
           - 7.0*(c_eff + 0.5*np.clip(c_eff - 0.40, 0, None))
           - 6.0*water - 2.2*np.exp(-dist_w/0.4)
           - 0.012*(elev - 900) + 1.5*industrial + 2.2*open_*bare
           + 0.9*grf(10) + rng.normal(0, 0.35, (N, N)))

    # --- administrative zones (approximate) and CV blocks ---
    deg = np.rad2deg(th) % 360
    zone = np.full((N, N), -1)
    core = r < 6.5
    zone[core & ((deg >= 315) | (deg < 90))] = 0
    zone[core & (deg >= 90) & (deg < 210)] = 1
    zone[core & (deg >= 210) & (deg < 315)] = 2
    o = ~core
    zone[o & ((deg >= 340) | (deg < 50))] = 3
    zone[o & (deg >= 50) & (deg < 110)] = 4
    zone[o & (deg >= 110) & (deg < 160)] = 5
    zone[o & (deg >= 160) & (deg < 260)] = 6
    zone[o & (deg >= 260) & (deg < 340)] = 7
    block = (ii//BLOCK_CELLS)*(N//BLOCK_CELLS) + jj//BLOCK_CELLS

    df = pd.DataFrame({
        "row": ii.ravel(), "col": jj.ravel(),
        "lat": (LAT0 + yk/KM_PER_DEG_LAT).ravel(), "lon": (LON0 + xk/KM_PER_DEG_LON).ravel(),
        "zone": zone.ravel(), "block": block.ravel(),
        "ndvi": ndvi.ravel(), "ndbi": ndbi.ravel(), "built": built.ravel(), "canopy": canopy.ravel(),
        "water": water.ravel(), "road": road.ravel(), "open": open_.ravel(), "elev": elev.ravel(),
        "dist_water_km": dist_w.ravel(), "canopy_500m": cn.ravel(), "built_500m": bn.ravel(),
        "ndvi_500m": k5(ndvi).ravel(), "pop": pop.ravel(), "vuln": vuln.ravel(), "lst": lst.ravel(),
    })
    df = df[inside.ravel()].reset_index(drop=True)
    df.insert(0, "cell_id", np.arange(len(df)))
    return df

if __name__ == "__main__":
    df = build()
    df.round(5).to_csv(DATA/"datacube.csv", index=False)
    print(f"data cube: {len(df):,} cells ({len(df)/100:.0f} km2), population {df['pop'].sum()/1e6:.2f} M")
    print(df[["lst","ndvi","built","canopy","pop","vuln"]].describe().round(2).T[["mean","min","max"]])
