"""
k = 5 nearest-neighbor cell graph for a single slide (SPA51), with checks.

  in : 2_athena/output/tables/SPA51_cells.csv
  out: 2_athena/output/graphs/SPA51_knn_k5.npz
"""

import os
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from scipy import sparse

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import config

cells_df = pd.read_csv(config.ATHENA_OUT / "tables" / "SPA51_cells.csv")

coordinates = cells_df[["x", "y"]].to_numpy(dtype=np.float64)


k = 5


# Find k nearest neighbors plus the cell itself.
model = NearestNeighbors(
    n_neighbors=k + 1, metric="euclidean", algorithm="kd_tree", n_jobs=-1
)

model.fit(coordinates)

distances, indices = model.kneighbors(coordinates)


# Remove the self-neighbor in the first column.
neighbor_distances = distances[:, 1:]
neighbor_indices = indices[:, 1:]


# Build directed neighbor relationships.
number_of_cells = len(cells_df)

source = np.repeat(np.arange(number_of_cells), k)

target = neighbor_indices.reshape(-1)


directed = sparse.coo_matrix(
    (np.ones(len(source), dtype=np.uint8), (source, target)),
    shape=(number_of_cells, number_of_cells),
).tocsr()


# Convert to an undirected union graph.
adjacency = directed.maximum(directed.T)

adjacency.setdiag(0)
adjacency.eliminate_zeros()


# Graph validation.
degree = np.asarray(adjacency.sum(axis=1)).ravel()


print("Nodes:", number_of_cells)
print("Edges:", adjacency.nnz // 2)
print("Mean degree:", degree.mean())
print("Is symmetric:", (adjacency != adjacency.T).nnz == 0)
print("Self-edges:", adjacency.diagonal().sum())
print("Isolated cells:", np.sum(degree == 0))


# Save graph.
os.makedirs(config.ATHENA_OUT / "graphs", exist_ok=True)
sparse.save_npz(config.ATHENA_OUT / "graphs" / "SPA51_knn_k5.npz", adjacency)
