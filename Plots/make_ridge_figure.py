"""
Figure: Ridge R^2 for height and weight within gender, mean over seeds (+-1 SD).

Input : the six files  <target>_seeds_summary_<model>.csv  written by
        probe_height_weight_seeds.py (in regression_results_mixer6/).
Output: images/ridge_r2.pdf

Usage : python make_ridge_figure.py [folder_with_summary_csvs] [output_pdf]
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("regression_results_mixer6")
out = sys.argv[2] if len(sys.argv) > 2 else "ridge_r2.pdf"

MODELS = [("facebook_wav2vec2-base", "wav2vec2", "#0072B2", "o"),
          ("facebook_hubert-base-ls960", "HuBERT", "#D55E00", "s"),
          ("microsoft_wavlm-base", "WavLM", "#009E73", "^")]
LAYERS = ["first", "middle", "final"]
TARGETS = [("height_cm", "Height (Ridge, within gender)"),
           ("weight_kg", "Weight (Ridge, within gender)")]

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)

for ax, (target, title) in zip(axes, TARGETS):
    for gi, (key, label, color, marker) in enumerate(MODELS):
        df = pd.read_csv(folder / f"{target}_seeds_summary_{key}.csv")
        for gender, filled in (("F", True), ("M", False)):
            for li, layer in enumerate(LAYERS):
                row = df[(df.gender == gender) & (df.layer == layer)].iloc[0]
                x = gi * 7 + (0 if gender == "F" else 3.2) + li * 0.9
                ax.errorbar(x, row.r2_mean, yerr=row.r2_sd, fmt="none",
                            ecolor=color, elinewidth=0.8, capsize=1.5, alpha=0.55, zorder=2)
                ax.plot(x, row.r2_mean, marker=marker, ms=5.5, color=color, zorder=3,
                        mfc=color if filled else "white", mew=1.1, ls="none")
    ax.axhline(0, color="black", lw=0.9)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([1.0 + 3.5 * 0 + i * 7 + 1.6 for i in range(3)])
    ax.set_xticklabels([m[1] for m in MODELS])
    ax.grid(True, axis="y", color="#dddddd", lw=0.6, zorder=0)
    ax.set_ylim(-0.8, 0.4)

axes[0].set_ylabel("$R^2$")
axes[0].text(0.03, 0.04, "$R^2=0$: predicting the mean", transform=axes[0].transAxes, fontsize=6.5)
axes[1].plot([], [], "o", color="gray", label="female (filled)")
axes[1].plot([], [], "o", mfc="white", color="gray", label="male (open)")
axes[1].legend(loc="lower right", frameon=False, fontsize=7)
fig.text(0.5, -0.02, "Symbols: mean over 10 speaker-disjoint splits; bars: $\\pm$1 SD across splits. "
         "Within each model: first, middle, final layer.", ha="center", fontsize=6.5)
fig.tight_layout()
fig.savefig(out, bbox_inches="tight")
print("saved", out)
