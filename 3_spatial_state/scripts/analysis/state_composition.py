"""
Median nucleus counts and compartment arrangement, per state, from the
pipeline output.

Read-only.
"""

import numpy as np
import pandas as pd
import glob
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import config

STEP5 = config.STEP5_TABLES

frames = []
for f in sorted(glob.glob(str(STEP5 / "*_interaction_state.csv"))):
    d = pd.read_csv(f)
    d = d[d.window_class == 4]
    if len(d):
        frames.append(d)
D = pd.concat(frames, ignore_index=True)
S = D["interaction_state"].astype(str).to_numpy()

CI, CC = "inflammatory_count", "neoplastic_count"

GROUPS = [("distant_co_occurrence", S == "distant_co_occurrence"),
          ("bordering",             S == "bordering"),
          ("infiltrated",           S == "infiltrated"),
          ("unlabeled",             S == "unlabeled"),
          ("ALL co-occurring",      np.ones(len(D), bool))]

print("=" * 86)
print("MEDIAN NUCLEI PER WINDOW  (with quartiles)")
print("=" * 86)
print(f'{"state":24s}{"n":>7s}{"immune q25":>12s}{"immune med":>12s}'
      f'{"immune q75":>12s}{"tumor q25":>11s}{"tumor med":>11s}{"tumor q75":>11s}')
for lab, m in GROUPS:
    n = int(m.sum())
    if not n:
        continue
    i, c = D.loc[m, CI], D.loc[m, CC]
    print(f"{lab:24s}{n:7d}{i.quantile(.25):12.1f}{i.median():12.1f}"
          f"{i.quantile(.75):12.1f}{c.quantile(.25):11.1f}{c.median():11.1f}"
          f"{c.quantile(.75):11.1f}")

print("\n" + "=" * 86)
print("SHARE OF WINDOWS WITH FEW TUMOR NUCLEI")
print("=" * 86)
print(f'{"state":24s}{"n":>7s}{"<=2":>10s}{"<=5":>10s}{"<=10":>10s}{">=20":>10s}')
for lab, m in GROUPS:
    n = int(m.sum())
    if not n:
        continue
    c = D.loc[m, CC].to_numpy(float)
    print(f"{lab:24s}{n:7d}{100*(c<=2).mean():9.0f}%{100*(c<=5).mean():9.0f}%"
          f"{100*(c<=10).mean():9.0f}%{100*(c>=20).mean():9.0f}%")

print("\n" + "=" * 86)
print("SPATIAL STATE OF EACH COMPARTMENT")
print("=" * 86)
for lab, key in [("immune", "immune_spatial_state"),
                 ("tumor",  "cancer_spatial_state")]:
    print(f"\n{lab}")
    for gl, m in GROUPS[:3]:
        vc = pd.Series(D.loc[m, key].to_numpy(object)).value_counts()
        tot = int(m.sum())
        print(f"  {gl:24s}" + "  ".join(
            f"{k}={v} ({100*v/tot:.0f}%)" for k, v in vc.items()))

print("\ndone")
