"""
Permutation null for the three state definitions, plus a reproduction check.

Two jobs:

  PART 1  Rebuild the three definitions from the raw features and compare the
          result to the pipeline's own interaction_state column, window by
          window. Any mismatch is printed with the values that caused it.

  PART 2  With a fixed seed, shuffle each metric independently within its
          slide, reapply the definitions, and report observed rate, null mean,
          upper bound of the null 95% interval, ratio and an empirical p-value for each state.

Read-only. Writes nothing.

Runtime is a few minutes at N_PERM = 1000. N_PERM = 200 is faster, and
the ratios change only in the second decimal place.
"""

import numpy as np
import pandas as pd
import glob
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import config

STEP5 = config.STEP5_TABLES
N_PERM = 1000
SPLIT_Q = 0.50
SEED = 0

NUM = {
    "int_i2c": "interaction_inflammatory_to_neoplastic",
    "int_c2i": "interaction_neoplastic_to_inflammatory",
    "inf_i2c": "infiltration_inflammatory_to_neoplastic",
    "inf_c2i": "infiltration_neoplastic_to_inflammatory",
    "modular": "global_modularity",
}
CAT = {"sp_i": "immune_spatial_state", "sp_c": "cancer_spatial_state"}
TUMOR_COUNT_COL = "neoplastic_count"
BORDER_MIN_TUMOR = 5
SIGN = {"int_i2c", "int_c2i"}
STATE_COL = "interaction_state"

frames = []
for f in sorted(glob.glob(str(STEP5 / "*_interaction_state.csv"))):
    d = pd.read_csv(f)
    d = d[d.window_class == 4]
    if not len(d):
        continue
    d = d.copy()
    d["slide"] = Path(f).name.replace("_interaction_state.csv", "")
    frames.append(d)

D = pd.concat(frames, ignore_index=True)
N = len(D)
groups = [np.asarray(ix) for ix in D.groupby("slide").indices.values()]
RAWN = {k: D[c].to_numpy(float) for k, c in NUM.items()}
TUMC = D[TUMOR_COUNT_COL].to_numpy(float)
RAWC = {k: D[c].to_numpy(object) for k, c in CAT.items()}
print(f"co-occurring windows: {N}   slides: {len(groups)}\n")


def build(rawn, rawc):
    hi, lo = {}, {}
    for k, v in rawn.items():
        if k in SIGN:
            hi[k] = np.nan_to_num(v, nan=-np.inf) > 0.0
            lo[k] = np.nan_to_num(v, nan=np.inf) < 0.0
            continue
        h = np.zeros(N, dtype=bool)
        filled = np.nan_to_num(v, nan=-np.inf)
        for ix in groups:
            ref = v[ix][np.isfinite(v[ix])]
            if len(ref) == 0:
                continue
            h[ix] = filled[ix] > float(np.quantile(ref, SPLIT_Q))
        # An undefined value is neither high nor low: a window whose
        # infiltration cannot be computed is never counted as low.
        hi[k] = h
        lo[k] = (~h) & np.isfinite(v)
    cl_i = np.isin(rawc["sp_i"], ["aggregate", "broad"])
    cl_c = np.isin(rawc["sp_c"], ["aggregate", "broad"])
    return hi, lo, cl_i, cl_c


def states(rawn, rawc, tumc):
    h, l, cl_i, cl_c = build(rawn, rawc)
    contact_hi = h["int_i2c"] & h["int_c2i"]
    contact_lo = l["int_i2c"] & l["int_c2i"]
    infil_hi = h["inf_i2c"] & h["inf_c2i"]
    return {
        "distant_co_occurrence": contact_lo & cl_i & cl_c & h["modular"],
        "bordering": (
            contact_hi & l["inf_i2c"] & cl_i & h["modular"] & (tumc >= BORDER_MIN_TUMOR)
        ),
        "infiltrated": contact_hi & infil_hi & l["modular"],
    }


# ------------------------------------------------- PART 1 reproduction check
rep = states(RAWN, RAWC, TUMC)
names = list(rep)

print("=" * 78)
print("PART 1   REPRODUCTION CHECK against the pipeline's own column")
print("=" * 78)
if STATE_COL not in D.columns:
    print(f"WARNING: {STATE_COL} not present; skipping check")
else:
    truth = D[STATE_COL].astype(str).to_numpy()
    total_mismatch = 0
    print(f'{"state":26s}{"rebuilt":>9s}{"pipeline":>10s}{"mismatch":>10s}')
    for k in names:
        mine = rep[k]
        theirs = truth == k
        mm = int((mine ^ theirs).sum())
        total_mismatch += mm
        print(f"{k:26s}{int(mine.sum()):9d}{int(theirs.sum()):10d}{mm:10d}")
    if total_mismatch == 0:
        print(
            "\n  Exact match: the rebuilt labels reproduce the pipeline labels in every window."
        )
    else:
        print(f"\n  {total_mismatch} disagreeing window(s). details:\n")
        show = list(NUM.values()) + list(CAT.values())
        for k in names:
            bad = np.where(rep[k] ^ (truth == k))[0]
            for i in bad[:10]:
                print(
                    f"  state={k}  rebuilt={bool(rep[k][i])}  "
                    f"pipeline={truth[i]}  slide={D['slide'].iat[i]}"
                )
                for cname in show:
                    print(f"      {cname:48s}{D[cname].iat[i]}")
                for mk, cname in NUM.items():
                    if mk in SIGN:
                        continue
                    ix = D.index[D.slide == D["slide"].iat[i]]
                    ref = D.loc[ix, cname].to_numpy(float)
                    ref = ref[np.isfinite(ref)]
                    if len(ref):
                        print(
                            f"      within-slide median of {cname:28s}"
                            f"{np.quantile(ref, SPLIT_Q)!r}"
                        )
                print()
print()

# ----------------------------------------------------------- PART 2 the null
obs = {k: float(v.mean()) for k, v in rep.items()}
rng = np.random.default_rng(SEED)
null = {k: [] for k in names}
for p in range(N_PERM):
    if (p + 1) % 200 == 0:
        print(f"  ...{p+1}/{N_PERM} permutations")
    sn, sc = {}, {}
    for k, v in RAWN.items():
        w = v.copy()
        for ix in groups:
            w[ix] = rng.permutation(w[ix])
        sn[k] = w
    for k, v in RAWC.items():
        w = v.copy()
        for ix in groups:
            w[ix] = rng.permutation(w[ix])
        sc[k] = w
    tc = TUMC.copy()
    for ix in groups:
        tc[ix] = rng.permutation(tc[ix])
    q = states(sn, sc, tc)
    for k in names:
        null[k].append(float(q[k].mean()))

print("\n" + "=" * 78)
print(f"PART 2   PERMUTATION NULL   seed={SEED}   permutations={N_PERM}")
print("=" * 78)
print(
    f'{"state":26s}{"obs n":>8s}{"obs":>8s}{"null mean":>11s}'
    f'{"null 95% hi":>13s}{"ratio":>8s}{"p":>10s}'
)
for k in names:
    a = np.array(null[k])
    ratio = obs[k] / a.mean() if a.mean() > 0 else float("nan")
    pval = (1 + int((a >= obs[k]).sum())) / (1 + N_PERM)
    print(
        f"{k:26s}{int(rep[k].sum()):8d}{100*obs[k]:7.2f}%"
        f"{100*a.mean():10.2f}%{100*np.quantile(a,0.975):12.2f}%"
        f"{ratio:8.2f}{pval:10.4f}"
    )
