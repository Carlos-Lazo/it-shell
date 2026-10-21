import pathlib
import numpy as np
from sklearn.decomposition import PCA
import csv

# Define paths (adjust if repository layout changes)
BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
vectors_path = BASE_DIR / "dataset_builder" / "indexer" / "artifacts" / "all-mpnet-base-v2" / "vectors.npz"
aliases_path = BASE_DIR / "dataset_builder" / "indexer" / "artifacts" / "all-mpnet-base-v2" / "aliases.npy"

# Load vectors
vectors_data = np.load(vectors_path)
# Determine key name
if "vectors" in vectors_data:
    vectors = vectors_data["vectors"]
elif "arr_0" in vectors_data:
    vectors = vectors_data["arr_0"]
else:
    vectors = vectors_data[list(vectors_data.files)[0]]

# Load aliases and extract category
aliases = np.load(aliases_path, allow_pickle=True)
categories = []
for a in aliases:
    s = a.decode() if isinstance(a, bytes) else str(a)
    cat = s.split(":")[0]
    categories.append(cat)

# PCA to 2 dimensions
pca = PCA(n_components=2, random_state=42)
proj = pca.fit_transform(vectors)

# Write CSV (PC1, PC2, category)
output_csv = BASE_DIR / "scripts" / "embeddings_pca.csv"
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["pc1", "pc2", "category"])
    for x, y, cat in zip(proj[:, 0], proj[:, 1], categories):
        writer.writerow([x, y, cat])

print(f"PCA projection exported to {output_csv}")

