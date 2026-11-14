# Investigating Speaker Attributes in Self-Supervised Speech Models

Code for the bachelor thesis *Investigating Speaker Attributes in Self-Supervised Speech
Models: A Comparison of Interpretability Techniques* (Diana Merkle, University of Zurich,
Department of Computational Linguistics, 2026). Supervisor: Prof. Dr. Eleanor Chodroff.

This work extends an earlier project on the TIMIT corpus
([wav2vec2-speaker-analysis](https://github.com/dilolua/wav2vec2-speaker-analysis)),
in which wav2vec2-base was probed for gender and dialect on two fixed sentences. The thesis
keeps the probing design of that project (mean-pooled embeddings of the first, middle and
final layer, one probe per sentence to control the spoken content) and applies it to a
larger corpus (Mixer 6), two additional models and three further techniques. The results
of the two projects are not directly comparable, because the corpora, the dialect classes
and the number of speakers differ.

## Overview

The thesis tests whether the representations of wav2vec2-base, HuBERT-base and WavLM-base
contain speaker attributes, and compares four interpretability techniques on the same
embeddings (layers 1, 6 and 12, mean-pooled). The data are 196 speakers and 140,887
sentence clips from the Mixer 6 corpus.

| Attribute | Techniques |
|---|---|
| Gender | linear probing, K-means clustering, simplified ABX test |
| Dialect region (from birth state) | linear probing, K-means clustering |
| Height and weight (within gender) | Ridge regression |

Main questions:

- Is speaker information (gender, dialect region, and within a gender, height and weight) present in the representations?
- Do the different techniques lead to the same conclusion for the same attribute?
- How do the answers depend on the layer and on the model?
- For which kind of attribute is each technique suitable?

## Methods in short

- **Linear probing:** logistic regression (`max_iter = 5000`) on a speaker-disjoint 80/20 split, once with all sentences pooled and once with one probe per sentence (20 sentences). Accuracy and macro-F1 are compared with a majority-class baseline.
- **K-means clustering:** k equal to the number of classes, 10 initialisations, on single clips and on speaker averages (standardised features). Measures: ARI, AMI, homogeneity, completeness and purity.
- **ABX (gender):** A is a male and B a female speaker, X alternates between male and female, and all three are different speakers saying the same sentence. Cosine distance, 5,000 sampled triplets per model and layer.
- **Ridge regression (height, weight):** fitted separately within each gender with `alpha = 1`, repeated over ten speaker-disjoint splits (seeds 42 and 0 to 8). Measures: R2, mean absolute error and Pearson correlation.

## Main results

- Gender is decoded by a linear probe with 96-99% accuracy in every model and layer, also when the sentence is held constant.
- K-means finds gender only after the embeddings are averaged per speaker, mostly at the middle layer (ARI up to 0.94).
- The ABX test finds gender with error rates of 23-45% (chance level 50%).
- None of the techniques recovered dialect region (defined by birth state). This does not show that dialect information is absent.
- Within a gender, neither height nor weight was predicted better than the group mean (R2 at or below 0 in nearly all conditions).
- The techniques give different pictures of the same representations, so conclusions about what a model encodes depend on the technique.

Details, tables and limitations are in the thesis.

## Scripts

| Script | Purpose |
|---|---|
| `build_metadata_from_clean_transcripts.py` | builds the sentence metadata |
| `cut_clips_from_sentence_metadata.py` | cuts the sentence clips |
| `extract_embeddings_mixer6.py` | extracts mean-pooled embeddings for the three models |
| `linear_probing.py` | pooled linear probing (gender, dialect region) |
| `linear_probing_per_sentence.py` | linear probing per sentence |
| `cluster_mixer6.py` | K-means clustering and evaluation measures |
| `abx_mixer6_sentence_matched.py` | sentence-matched ABX test for gender |
| `probe_height_weight_mixer6.py` | Ridge regression for height and weight (seed 42) |
| `probe_height_weight_seeds.py` | Ridge regression over ten speaker-disjoint splits |
| `make_thesis_figures.py` | probing, three-technique and height-weight figures |
| `make_ridge_figure.py` | Ridge regression figure |
| `make_speaker_pca_figure.py` | PCA figure of the speaker averages |

The settings (model, layer, target, seeds) are at the top of each script.

## Requirements

Python 3.12.6 with `torch` 2.2.2, `torchaudio` 2.2.2, `transformers` 4.44.2,
`scikit-learn` 1.9.0, `numpy` 2.5.4, `pandas` 3.0.5, `scipy` 1.17.1 and
`matplotlib` 3.11.2.

```bash
pip install -r requirements.txt
```

## Data

The Mixer 6 corpus is distributed by the Linguistic Data Consortium (LDC) and is not
included in this repository. Audio files, transcripts and the demographics file must be
obtained from the LDC. The pre-trained models are downloaded from Hugging Face:

```
facebook/wav2vec2-base
facebook/hubert-base-ls960
microsoft/wavlm-base
```

## Reproducing the results

1. Prepare the sentence metadata and cut the clips (`build_metadata_from_clean_transcripts.py`, `cut_clips_from_sentence_metadata.py`).
2. Extract the embeddings for the three models (`extract_embeddings_mixer6.py`).
3. Run the analyses (`linear_probing.py`, `linear_probing_per_sentence.py`, `cluster_mixer6.py`, `abx_mixer6_sentence_matched.py`, `probe_height_weight_mixer6.py`, `probe_height_weight_seeds.py`).
4. Create the figures (`make_thesis_figures.py`, `make_ridge_figure.py`, `make_speaker_pca_figure.py`).

All random choices use seed 42. The Ridge regressions are repeated with ten seeds.
