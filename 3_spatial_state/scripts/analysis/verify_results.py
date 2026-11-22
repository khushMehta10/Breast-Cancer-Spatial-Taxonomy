"""
Pull every number reported in the Results section directly from the
pipeline output.

Read-only. Prints the values reported in the Results section.

The state, niche and dominance column names are detected automatically.
"""

import numpy as np
import pandas as pd
import glob
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import config

STEP5 = config.STEP5_TABLES

files = sorted(glob.glob(str(STEP5 / "*_interaction_state.csv")))
if not files:
    raise SystemExit(f"no Step 5 tables under {STEP5}")

frames = []
for f in files:
    d = pd.read_csv(f)
    d["slide"] = Path(f).name.replace("_interaction_state.csv", "")
    frames.append(d)
D = pd.concat(frames, ignore_index=True)

print(f"Step 5 tables read: {len(files)}")
print(f"rows total:         {len(D)}\n")


def find(*subs):
    """first column whose lowercase name contains all substrings"""
    for c in D.columns:
        lc = c.lower()
        if all(s in lc for s in subs):
            return c
    return None


STATE = find("interaction_state") or find("state")
NICHE = find("niche")
DOM   = find("dominan")

print("detected columns")
print(f"  state:     {STATE}")
print(f"  niche:     {NICHE}")
print(f"  dominance: {DOM}\n")
if STATE is None:
    print("columns present:", list(D.columns))
    raise SystemExit("could not find the state column; inspect the list above")

# ------------------------------------------------------------- denominators
ALL_TISSUE  = len(D)
ALL_SLIDES  = D["slide"].nunique()

if "window_class" in D.columns:
    CO = D[D.window_class == 4].copy()
else:
    CO = D[D[STATE].notna()].copy()
    print("WARNING: no window_class column; co-occurring set inferred from state column\n")

n_co      = len(CO)
slides_co = CO["slide"].nunique()

print("=" * 70)
print("DENOMINATORS")
print("=" * 70)
print(f"tissue windows in Step 5 tables      {ALL_TISSUE:8d}   slides {ALL_SLIDES}")
print(f"co-occurring windows (class 4)       {n_co:8d}   slides {slides_co}")
if ALL_TISSUE:
    print(f"co-occurring as share of tissue      {100*n_co/ALL_TISSUE:8.1f}%")
print()

# ------------------------------------------------------------------ states
print("=" * 70)
print("STATES   (denominator = co-occurring windows)")
print("=" * 70)
print(f'{"state":26s}{"windows":>9s}{"% of co-occ":>13s}{"slides":>8s}{"win/slide":>11s}')
counts = CO[STATE].value_counts(dropna=False)
named = 0
for st in counts.index:
    m = CO[STATE] == st
    n = int(m.sum())
    sl = int(CO.loc[m, "slide"].nunique())
    if str(st).lower() not in ("unlabeled", "unlabelled", "nan", "none", "other"):
        named += n
    print(f"{str(st):26s}{n:9d}{100*n/n_co:12.1f}%{sl:8d}{n/sl if sl else 0:11.1f}")
print("-" * 67)
print(f'{"named total":26s}{named:9d}{100*named/n_co:12.1f}%')
print(f'{"unnamed":26s}{n_co-named:9d}{100*(n_co-named)/n_co:12.1f}%')
print()

# ------------------------------------------------------------------- niche
if NICHE is not None:
    nm = CO[NICHE].astype(str).str.lower().isin(["true", "1", "1.0", "yes"])
    n_n = int(nm.sum())
    sl_n = int(CO.loc[nm, "slide"].nunique())
    print("=" * 70)
    print("NICHE")
    print("=" * 70)
    print(f"niche windows                        {n_n:8d}   slides {sl_n}")
    print(f"as share of co-occurring             {100*n_n/n_co:8.1f}%")
    inf = CO[STATE].astype(str).str.contains("infiltrat", case=False, na=False)
    if inf.sum():
        print(f"as share of infiltrated              {100*n_n/int(inf.sum()):8.1f}%")
        outside = int((nm & ~inf).sum())
        print(f"niche windows NOT infiltrated        {outside:8d}   (should be 0)")
    print()

# --------------------------------------------------------------- dominance
if DOM is not None:
    print("=" * 70)
    print("DOMINANCE within each state")
    print("=" * 70)
    labs = sorted(CO[DOM].dropna().astype(str).unique())
    hdr = f'{"state":26s}{"n":>7s}'
    for L in labs:
        hdr += f"{L:>20s}"
    print(hdr)
    rows = list(counts.index) + ["ALL co-occurring"]
    for st in rows:
        m = (np.ones(n_co, dtype=bool) if st == "ALL co-occurring"
             else (CO[STATE] == st).to_numpy())
        n = int(m.sum())
        if not n:
            continue
        sub = CO.loc[m, DOM].astype(str)
        row = f"{str(st):26s}{n:7d}"
        for L in labs:
            k = int((sub == L).sum())
            row += f"{k:12d}{100*k/n:7.0f}%"
        print(row)
    print()

print("=" * 70)
print("Every number reported in the Results section should match a value above.")
print("=" * 70)
