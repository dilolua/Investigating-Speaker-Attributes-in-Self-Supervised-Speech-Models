"""
Makes the figure of speaker-averaged embeddings (PCA, 2 components),
coloured by gender, for the first and middle layer of the three models.

"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- adjust ---------------------------------------------------------------
MODELS = {  # label: (folder with embeddings_*_<layer>.npy and metadata_processed.csv, file stem)
    "wav2vec2": ("embeddings/wav2vec2", "facebook_wav2vec2-base"),
    "HuBERT":   ("embeddings/hubert",   "facebook_hubert-base-ls960"),
    "WavLM":    ("embeddings/wavlm",    "microsoft_wavlm-base"),
}
LAYERS = ["first", "middle"]          # add "final" if wanted
OUT = "speaker_pca_gender.pdf"
# --------------------------------------------------------------------------

fig, axes = plt.subplots(len(LAYERS), len(MODELS), figsize=(9, 5.6), squeeze=False)
colors = {"F": "#CC79A7", "M": "#0072B2"}

for j, (label, (folder, stem)) in enumerate(MODELS.items()):
    meta = pd.read_csv(Path(folder) / "metadata_processed.csv")
    for i, layer in enumerate(LAYERS):
        emb = np.load(Path(folder) / f"embeddings_{stem}_{layer}.npy")
        df = pd.DataFrame(emb)
        df["speaker_id"] = meta["speaker_id"].to_numpy()
        spk = df.groupby("speaker_id").mean()               # one point per speaker
        gender = meta.drop_duplicates("speaker_id").set_index("speaker_id")["gender"].loc[spk.index]
        xy = PCA(n_components=2, random_state=42).fit_transform(spk.to_numpy())
        ax = axes[i, j]
        for g, name in [("F", "female"), ("M", "male")]:
            m = (gender == g).to_numpy()
            ax.scatter(xy[m, 0], xy[m, 1], s=12, alpha=0.75, c=colors[g], label=name)
        ax.set_title(f"{label}, {layer} layer", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0 and j == 0:
            ax.legend(fontsize=8, frameon=False)
fig.supxlabel("PC 1", fontsize=8); fig.supylabel("PC 2", fontsize=8)
plt.tight_layout()
plt.savefig(OUT)
print("saved", OUT)
