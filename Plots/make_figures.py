"""
Figures for the thesis (except the speaker PCA figure and the Ridge figure,
which have their own scripts: make_speaker_pca_figure.py, make_ridge_figure.py).

Creates:
    probing_gender_dialect.pdf     pooled linear probing, gender and dialect
    gender_three_techniques.pdf    probing / K-means (speaker level) / ABX for gender
    height_weight_hist.pdf         speaker-level height and weight distributions
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

MODELS = [("facebook_wav2vec2-base", "wav2vec2", "#0072B2", "o"),
          ("facebook_hubert-base-ls960", "HuBERT", "#D55E00", "s"),
          ("microsoft_wavlm-base", "WavLM", "#009E73", "^")]
LAYERS = ["first", "middle", "final"]
XLAB = ["First", "Middle", "Final"]
ABX_ERROR = {"wav2vec2": [36.9, 36.2, 45.0],
             "HuBERT": [28.7, 29.1, 31.7],
             "WavLM": [28.8, 22.6, 37.0]}


def by_layer(df, col):
    return [float(df[df.layer == l][col].iloc[0]) for l in LAYERS]


def fig_probing(out="probing_gender_dialect.pdf"):
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8), sharey=True)
    for ax, task, title in zip(axes, ["gender", "dialect"],
                               ["Gender (pooled probe)", "Dialect region (pooled probe)"]):
        for key, label, color, marker in MODELS:
            df = pd.read_csv(f"{task}_probe_summary_mixer6_{key}.csv")
            ax.plot(XLAB, [100 * v for v in by_layer(df, "accuracy")], marker=marker,
                    color=color, lw=1.6, ms=5, label=label)
        base = 100 * float(df["baseline_accuracy"].iloc[0])
        ax.axhline(base, color="gray", ls="--", lw=0.8)
        ax.text(2.0, base + 1.2, "majority baseline", ha="right", fontsize=6, color="gray")
        ax.set_title(title, fontsize=8)
        ax.grid(True, color="#dddddd", lw=0.6)
        ax.set_ylim(45, 102)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(loc="center left", frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")


def fig_three(out="gender_three_techniques.pdf"):
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    titles = ["Linear probing, pooled\n(accuracy, %; higher is better)",
              "K-means, speaker level\n(ARI; higher is better)",
              "ABX, sentence matched\n(error, %; lower is better)"]
    for key, label, color, marker in MODELS:
        acc = pd.read_csv(f"gender_probe_summary_mixer6_{key}.csv")
        ari = pd.read_csv(f"gender_cluster_summary_{key}_standardized_speakerlevel.csv")
        kw = dict(marker=marker, color=color, lw=1.6, ms=5, label=label)
        axes[0].plot(XLAB, [100 * v for v in by_layer(acc, "accuracy")], **kw)
        axes[1].plot(XLAB, by_layer(ari, "ari"), **kw)
        axes[2].plot(XLAB, ABX_ERROR[label], **kw)
    axes[0].set_ylim(90, 100)
    axes[1].set_ylim(-0.02, 1.0)
    axes[2].set_ylim(15, 52)
    axes[2].axhline(50, color="gray", ls="--", lw=0.8)
    axes[2].text(0.02, 48.8, "chance", fontsize=6, color="gray", va="top")
    for ax, t in zip(axes, titles):
        ax.set_title(t, fontsize=8)
        ax.grid(True, color="#dddddd", lw=0.6)
    axes[0].legend(loc="lower left", frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")


def fig_hist(out="height_weight_hist.pdf"):
    demo = pd.read_csv("mixer6_demographics.csv")
    meta = pd.read_csv("metadata_mixer6_sentences_with_clips.csv", usecols=["speaker_id"])
    spk = demo[demo.subjid.isin(meta.speaker_id.unique())].copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for ax, col, xlabel, bins in [(axes[0], "ht_cm", "Height (cm)", 15),
                                  (axes[1], "wt_kg", "Weight (kg)", 15)]:
        d = spk[spk[col] > 0]   # a recorded value of 0 is treated as missing
        for g, color, name in [("F", "#CC79A7", "female"), ("M", "#0072B2", "male")]:
            v = d[d.gender == g][col]
            ax.hist(v, bins=bins, alpha=0.65, color=color, label=f"{name} (n={len(v)})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Speakers")
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.legend(frameon=False, fontsize=7)
        ax.grid(True, color="#dddddd", lw=0.6)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")


if __name__ == "__main__":
    fig_probing()
    fig_three()
    fig_hist()
    print("saved: probing_gender_dialect.pdf, gender_three_techniques.pdf, height_weight_hist.pdf")
