import pathlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Paths (adjust if needed)
BASE_DIR = pathlib.Path(__file__).resolve().parents[2]  # repository root
vectors_path = BASE_DIR / "dataset_builder" / "indexer" / "artifacts" / "all-mpnet-base-v2" / "vectors.npz"
aliases_path = BASE_DIR / "dataset_builder" / "indexer" / "artifacts" / "all-mpnet-base-v2" / "aliases.npy"

# Load vectors and aliases
vectors_data = np.load(vectors_path)
# Assuming the key is 'arr_0' or 'vectors'
if "vectors" in vectors_data:
    vectors = vectors_data["vectors"]
elif "arr_0" in vectors_data:
    vectors = vectors_data["arr_0"]
else:
    # fallback to first array
    vectors = vectors_data[list(vectors_data.files)[0]]

aliases = np.load(aliases_path, allow_pickle=True)
# aliases are strings like "category:id"
categories = []
for a in aliases:
    # ensure str
    s = a.decode() if isinstance(a, bytes) else str(a)
    cat = s.split(":")[0]
    categories.append(cat)

# Dimensionality reduction to 2D
pca = PCA(n_components=2, random_state=42)
proj = pca.fit_transform(vectors)

# Plot
plt.figure(figsize=(12, 8))
sns.scatterplot(x=proj[:, 0], y=proj[:, 1], hue=categories, palette="tab20", legend="full", s=30, linewidth=0)
plt.title("PCA projection of SBERT embeddings (all‑mpnet‑base‑v2)")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.tight_layout()
# Save figure to artifacts directory
output_path = pathlib.Path("/home/carlos/.gemini/antigravity/brain/86486c30-e529-436a-9b49-a5609f9bc508") / "visualization_embeddings.png"
plt.savefig(output_path, dpi=150)
print(f"Embedding visualization saved to {output_path}")

