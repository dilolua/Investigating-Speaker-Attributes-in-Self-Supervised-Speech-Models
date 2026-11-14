"""
Ridge regression for height / weight within gender (Mixer 6), repeated over
several speaker-disjoint 80/20 splits (seeds). Same model (Ridge, alpha=1),
same filtering and same within-gender split logic as probe_height_weight_mixer6.py;
only the split seed changes.

Run from the folder that contains the embeddings_mixer6* folders:
    python probe_height_weight_seeds.py

Change TARGET and MODEL_CHOICE below (3 models x 2 targets = 6 runs).

Output (in regression_results_mixer6/):
    <target>_seeds_<model>.csv          one row per seed, gender and layer
    <target>_seeds_summary_<model>.csv  mean and standard deviation over seeds
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

MODEL_CHOICE = "wav2vec2"   # "wav2vec2", "hubert" or "wavlm"
TARGET = "weight_kg"        # "height_cm" or "weight_kg"
SEEDS = [42, 0, 1, 2, 3, 4, 5, 6, 7, 8]   # 42 = the seed used so far
TEST_SIZE = 0.2
ALPHA = 1.0
LAYERS = ["first", "middle", "final"]

MODEL_CONFIGS = {
    "wav2vec2": {"name": "facebook/wav2vec2-base", "dir": "embeddings_mixer6"},
    "hubert": {"name": "facebook/hubert-base-ls960", "dir": "embeddings_mixer6_hubert"},
    "wavlm": {"name": "microsoft/wavlm-base", "dir": "embeddings_mixer6_wavlm"},
}

MODEL_NAME = MODEL_CONFIGS[MODEL_CHOICE]["name"]
EMB_DIR = Path(MODEL_CONFIGS[MODEL_CHOICE]["dir"])
METADATA_FILE = EMB_DIR / "metadata_processed.csv"
SAFE = MODEL_NAME.replace("/", "_")

OUT_DIR = Path("regression_results_mixer6")
OUT_ALL = OUT_DIR / f"{TARGET}_seeds_{SAFE}.csv"
OUT_SUM = OUT_DIR / f"{TARGET}_seeds_summary_{SAFE}.csv"


def speaker_disjoint_split(metadata, seed):
    speakers = metadata["speaker_id"].unique()
    train_spk, test_spk = train_test_split(speakers, test_size=TEST_SIZE, random_state=seed)
    train_idx = metadata.index[metadata["speaker_id"].isin(train_spk)].to_numpy()
    test_idx = metadata.index[metadata["speaker_id"].isin(test_spk)].to_numpy()
    return train_idx, test_idx


OUT_DIR.mkdir(parents=True, exist_ok=True)

metadata_full = pd.read_csv(METADATA_FILE)
valid = (
    ~metadata_full["gender"].astype(str).str.contains("Unknown", case=False, na=True)
    & metadata_full[TARGET].notna()
    & (metadata_full[TARGET] > 0)     # a recorded value of 0 (e.g. 0 kg) is treated as missing
)
valid_mask = valid.to_numpy()
metadata = metadata_full[valid_mask].reset_index(drop=True)
print(f"{TARGET}: dropped {(~valid_mask).sum()} rows with missing/unknown values or values <= 0")

# load each layer once
emb_full = {
    layer: np.load(EMB_DIR / f"embeddings_{SAFE}_{layer}.npy")[valid_mask]
    for layer in LAYERS
}

rows = []
for gender in sorted(metadata["gender"].unique()):
    g_mask = (metadata["gender"] == gender).to_numpy()
    g_meta = metadata[g_mask].reset_index(drop=True)
    y = g_meta[TARGET]
    emb = {layer: emb_full[layer][g_mask] for layer in LAYERS}

    for seed in SEEDS:
        train_idx, test_idx = speaker_disjoint_split(g_meta, seed)
        n_test_spk = g_meta.iloc[test_idx]["speaker_id"].nunique()
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        for layer in LAYERS:
            model = Ridge(alpha=ALPHA)
            model.fit(emb[layer][train_idx], y_train)
            pred = model.predict(emb[layer][test_idx])
            r = pearsonr(y_test, pred)[0] if y_test.nunique() > 1 else float("nan")
            row = {
                "target": TARGET, "model": MODEL_NAME, "gender": gender,
                "layer": layer, "seed": seed, "n_test_speakers": n_test_spk,
                "r2": r2_score(y_test, pred),
                "mae": mean_absolute_error(y_test, pred),
                "pearson_r": r,
            }
            rows.append(row)
            print(f"{gender} seed {seed:>3} {layer:<6} R2={row['r2']:+.3f} "
                  f"r={r:.3f} (test speakers: {n_test_spk})")

df = pd.DataFrame(rows)
df.to_csv(OUT_ALL, index=False)

summary = (
    df.groupby(["target", "model", "gender", "layer"], sort=False)
      .agg(n_seeds=("seed", "nunique"),
           r2_mean=("r2", "mean"), r2_sd=("r2", "std"),
           r2_min=("r2", "min"), r2_max=("r2", "max"),
           pearson_mean=("pearson_r", "mean"), pearson_sd=("pearson_r", "std"),
           mae_mean=("mae", "mean"))
      .reset_index()
)
summary.to_csv(OUT_SUM, index=False)
print()
print(summary.to_string(index=False))
print(f"\nSaved: {OUT_ALL}\nSaved: {OUT_SUM}")
