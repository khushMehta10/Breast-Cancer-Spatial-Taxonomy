"""
k = 5 nearest-neighbor cell graph for every slide.

  in : 2_athena/output/tables/<SLIDE>_cells.csv
  out: 2_athena/output/graphs/<SLIDE>_knn_k5.npz   (symmetric sparse adjacency)
"""

import glob, os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy import sparse

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import config

TABLE_DIR = str(config.ATHENA_OUT / "tables")
OUT_DIR = str(config.ATHENA_OUT / "graphs")
K = 5
os.makedirs(OUT_DIR, exist_ok=True)

tables = sorted(glob.glob(f"{TABLE_DIR}/*_cells.csv"))
print(f"{len(tables)} tables\n", flush=True)

for i, path in enumerate(tables, 1):
    slide = os.path.basename(path).replace("_cells.csv", "")
    out = f"{OUT_DIR}/{slide}_knn_k{K}.npz"
    if os.path.exists(out):
        print(f"[{i}/{len(tables)}] {slide} - already done, skipping", flush=True)
        continue

    print(f"===== [{i}/{len(tables)}] {slide} =====", flush=True)

    cells_df = pd.read_csv(path)
    coordinates = cells_df[["x", "y"]].to_numpy(dtype=np.float64)
    k = K

    model = NearestNeighbors(
        n_neighbors=k + 1, metric="euclidean", algorithm="kd_tree", n_jobs=-1
    )
    model.fit(coordinates)
    distances, indices = model.kneighbors(coordinates)

    neighbor_distances = distances[:, 1:]
    neighbor_indices = indices[:, 1:]

    number_of_cells = len(cells_df)
    source = np.repeat(np.arange(number_of_cells), k)
    target = neighbor_indices.reshape(-1)

    directed = sparse.coo_matrix(
        (np.ones(len(source), dtype=np.uint8), (source, target)),
        shape=(number_of_cells, number_of_cells),
    ).tocsr()

    adjacency = directed.maximum(directed.T)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()

    degree = np.asarray(adjacency.sum(axis=1)).ravel()

    print("Nodes:", number_of_cells)
    print("Edges:", adjacency.nnz // 2)
    print("Mean degree:", degree.mean())
    print("Is symmetric:", (adjacency != adjacency.T).nnz == 0)
    print("Self-edges:", adjacency.diagonal().sum())
    print("Isolated cells:", np.sum(degree == 0))

    sparse.save_npz(out, adjacency)
    print("wrote", out, "\n", flush=True)
