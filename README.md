# Investigating Speaker Attributes in Self-Supervised Speech Models

Code for the bachelor thesis "Investigating Speaker Attributes in Self-Supervised
Speech Models: A Comparison of Interpretability Techniques" (University of Zurich,
Department of Computational Linguistics, 2026). Author: Diana Merkle.
Supervisor: Prof. Dr. Eleanor Chodroff.

## What the project does

The thesis tests whether the representations of wav2vec2-base, HuBERT-base and
WavLM-base contain speaker attributes, and compares four techniques on the same
embeddings (layers 1, 6 and 12, mean-pooled):

| Attribute | Techniques |
|---|---|
| Gender | linear probing, K-means clustering, simplified ABX test |
| Dialect region | linear probing, K-means clustering |
| Height and weight (within gender) | Ridge regression |

The data are 196 speakers and 140,887 sentence clips from the Mixer 6 corpus.

## Repository structure

| Folder | Content |
|---|---|
| `Data/` | preparation of the sentence metadata and clips |
| `Embeddings/` | embedding extraction |
| `Linear probing/` | pooled and per-sentence probing |
| `Cluster/` | K-means clustering and cluster measures |
| `ABX/` | sentence-matched ABX test |
| `Ridge regression/` | height and weight within gender, including the seed runs |
| `figures/` | scripts for the thesis figures |

## Scripts

| Script | Purpose |
|---|---|
| `build_metadata_from_clean_transcripts.py` | builds the sentence metadata |
| `cut_clips_from_sentence_metadata.py` | cuts the sentence clips |
| `extract_embeddings_mixer6.py` | extracts mean-pooled embeddings |
| `linear_probing.py` | pooled linear probing |
| `linear_probing_per_sentence.py` | linear probing per sentence |
| `cluster_mixer6.py` | K-means clustering and evaluation measures |
| `abx_mixer6_sentence_matched.py` | ABX test for gender |
| `probe_height_weight_mixer6.py` | Ridge regression (seed 42) |
| `probe_height_weight_seeds.py` | Ridge regression over ten speaker splits |
| `make_thesis_figures.py`, `make_ridge_figure.py`, `make_speaker_pca_figure.py` | figures |

Each script has its settings (model, layer, target, seeds) at the top of the file.

## Requirements

Python 3.12.6 with `torch` 2.2.2, `torchaudio` 2.2.2, `transformers` 4.44.2,
`scikit-learn` 1.9.0, `numpy` 2.5.4, `pandas` 3.0.5, `scipy` 1.17.1 and
`matplotlib` 3.11.2.

```
pip install -r requirements.txt
```

## Data

The Mixer 6 corpus is distributed by the Linguistic Data Consortium (LDC) and is
not included in this repository. Audio files, transcripts and the demographics file
must be obtained from the LDC. The pre-trained models are downloaded from Hugging
Face (`facebook/wav2vec2-base`, `facebook/hubert-base-ls960`, `microsoft/wavlm-base`).

## Reproducing the results

1. Prepare the metadata and cut the clips (`Data/`).
2. Extract the embeddings for the three models (`Embeddings/`).
3. Run the analyses (`Linear probing/`, `Cluster/`, `ABX/`, `Ridge regression/`).
4. Create the figures with the scripts in `figures/`.

All random choices use seed 42. The Ridge regressions are repeated with ten seeds
(42 and 0 to 8).

## Citation

Merkle, D. (2026). *Investigating Speaker Attributes in Self-Supervised Speech Models:
A Comparison of Interpretability Techniques.* Bachelor's thesis, University of Zurich.

## Licence

[add a licence, for example MIT, or "All rights reserved"]
