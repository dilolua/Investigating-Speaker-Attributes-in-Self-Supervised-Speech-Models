from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr


MODEL_CHOICE = "wav2vec2"
#options: "wav2vec2", "hubert", or "wavlm"

MODEL_CONFIGS = {
    "wav2vec2": {"name": "facebook/wav2vec2-base", "dir": "embeddings_mixer6"},
    "hubert": {"name": "facebook/hubert-base-ls960", "dir": "embeddings_mixer6_hubert"},
    "wavlm": {"name": "microsoft/wavlm-base", "dir": "embeddings_mixer6_wavlm"},
}

MODEL_NAME = MODEL_CONFIGS[MODEL_CHOICE]["name"]
EMBEDDINGS_DIR = Path(MODEL_CONFIGS[MODEL_CHOICE]["dir"])
METADATA_FILE = EMBEDDINGS_DIR / "metadata_processed.csv"

LAYERS = ["first", "middle", "final"]

TARGET = "weight_kg"
#options: "height_cm" or "weight_kg"

TEST_SIZE = 0.2
RANDOM_SEED = 42

OUTPUT_DIR = Path("regression_results_mixer6")
OUTPUT_FILE = OUTPUT_DIR / f"{TARGET}_by_gender_summary_{MODEL_NAME.replace('/', '_')}.csv"


#Load embeddings + metadata


def load_layer_embeddings(layer_name: str) -> np.ndarray:
    safe_model_name = MODEL_NAME.replace("/", "_")
    path = EMBEDDINGS_DIR / f"embeddings_{safe_model_name}_{layer_name}.npy"
    return np.load(path)


def load_metadata() -> pd.DataFrame:
    return pd.read_csv(METADATA_FILE)


def drop_missing(metadata: pd.DataFrame, target: str) -> np.ndarray:
    """Drop rows missing gender or the target variable itself."""
    valid_mask = (
            ~metadata["gender"].astype(str).str.contains("Unknown", case=False, na=True)
            & metadata[target].notna()
            & (metadata[target] > 0)
    )
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        print(f"Dropping {n_dropped} rows with missing gender or {target}")
    return valid_mask.to_numpy()


#Speaker-disjoint split, done WITHIN a single gender at a time

def speaker_disjoint_split(metadata: pd.DataFrame, test_size: float, random_seed: int):
    unique_speakers = metadata["speaker_id"].unique()
    train_speakers, test_speakers = train_test_split(
        unique_speakers, test_size=test_size, random_state=random_seed,
    )
    train_idx = metadata.index[metadata["speaker_id"].isin(train_speakers)].to_numpy()
    test_idx = metadata.index[metadata["speaker_id"].isin(test_speakers)].to_numpy()
    return train_idx, test_idx


#Train + evaluate a regression probe

def run_regression(embeddings: np.ndarray, y: pd.Series,
                    train_idx: np.ndarray, test_idx: np.ndarray) -> dict:
    X_train, X_test = embeddings[train_idx], embeddings[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    corr, _ = pearsonr(y_test, y_pred) if len(set(y_test)) > 1 else (float("nan"), None)

    return {
        "r2": r2,
        "mae": mae,
        "pearson_r": corr,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "target_mean": y.mean(),
        "target_std": y.std(),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Target: {TARGET}")
    print(f"Model: {MODEL_NAME}\n")

    metadata_full = load_metadata()
    valid_mask = drop_missing(metadata_full, TARGET)
    metadata = metadata_full[valid_mask].reset_index(drop=True)

    print(f"Gender distribution:\n{metadata['gender'].value_counts()}\n")

    summary_rows = []

    #NEW: this is the key step -- run the regression SEPARATELY within
    #each gender, never combining them. This is what actually "controls
    #for gender."
    for gender_value in sorted(metadata["gender"].unique()):
        print(f"=== Gender: {gender_value} ===")
        gender_metadata = metadata[metadata["gender"] == gender_value].reset_index(drop=True)
        gender_mask = (metadata["gender"] == gender_value).to_numpy()

        print(f"  {gender_metadata['speaker_id'].nunique()} speakers, "
              f"{len(gender_metadata)} sentences")
        print(f"  {TARGET} range: {gender_metadata[TARGET].min():.1f} - "
              f"{gender_metadata[TARGET].max():.1f} "
              f"(mean {gender_metadata[TARGET].mean():.1f})\n")

        train_idx, test_idx = speaker_disjoint_split(gender_metadata, TEST_SIZE, RANDOM_SEED)

        for layer_name in LAYERS:
            embeddings_full = load_layer_embeddings(layer_name)
            embeddings = embeddings_full[valid_mask][gender_mask]

            result = run_regression(embeddings, gender_metadata[TARGET], train_idx, test_idx)

            print(f"  --- Layer: {layer_name} ---")
            print(f"    R²: {result['r2']:.4f}  (0 = no better than predicting the mean; "
                  f"negative = worse than that)")
            print(f"    MAE: {result['mae']:.2f}  (average error, same units as {TARGET})")
            print(f"    Pearson r: {result['pearson_r']:.4f}\n")

            summary_rows.append({
                "target": TARGET,
                "model": MODEL_NAME,
                "gender": gender_value,
                "layer": layer_name,
                "r2": result["r2"],
                "mae": result["mae"],
                "pearson_r": result["pearson_r"],
                "n_train": result["n_train"],
                "n_test": result["n_test"],
                "target_mean": result["target_mean"],
                "target_std": result["target_std"],
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_FILE, index=False)
    print(summary_df.to_string())
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
