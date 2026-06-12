# FOMO26 Evaluation Pipeline

Linear probing and fairness evaluation for the [FOMO26 Challenge](https://fomo26.github.io).

Given precomputed embeddings from a model, this pipeline trains a linear classifier on the class labels and reports classification performance and fairness metrics across the configured grouping variables.

## How it works

1. A model (container) produces a 1-D embedding per scan and saves it to disk as `<ptid>.npy` (one file per subject). We provide an example encoder (AMAES `resenc_unet_b` checkpoint from HF) so you can sanity-check the pipeline end-to-end.
2. Main script `lp_fomo.py` loads the embeddings, joins them to labels from the subjects CSV by ptid (patient's id), splits subjects into train/val/test sets, runs a linear probe, and writes a fairness report.

## Installation

```bash
pip install -r requirements.txt

# Fairness metrics package
pip install git+https://github.com/fomo26/fomo-metrics.git
```

## Configuring fairness variables

Before running the pipeline you must tell it which columns in your CSV hold
the demographic (fairness) variables.  Open `pipeline/config.py` and fill in
the three values:

```python
# First demographic variable column in your subjects CSV.
FEATURE1_CSV_COL = "your_variable_1"

# Second demographic variable column in your subjects CSV.
FEATURE2_CSV_COL = "your_variable_2"

# Bin upper bounds (inclusive) and integer group labels for FEATURE1.
# Set these to match the range of your variable.
# Example: FEATURE1_BINS = ((25, 0), (50, 1), (75, 2), (1000, 3))
FEATURE1_BINS = ((..., 0), (..., 1), (..., 2))
```

`FEATURE1_BINS` is a sequence of `(upper_bound, label)` pairs in ascending
order.  A subject whose value is ≤ the first upper bound gets label 0, ≤ the
second gets label 1, and so on.  Make the last upper bound large enough to
cover all expected values in your data.

## CSV format

Your subjects CSV must contain at minimum the following columns:

| Column | Description |
|---|---|
| `ptid` | Unique subject identifier — must match the `.npy` filename |
| `selected_cohort` | Class label for the linear probe (e.g. diagnosis group) |
| *(your FEATURE1 column)* | First demographic variable |
| *(your FEATURE2 column)* | Second demographic variable |

For embedding extraction with `embed_all.py`, the CSV must also include:

| Column | Description |
|---|---|
| `nifti_path` | Absolute path to the subject's NIfTI scan |

## Usage

Extract embeddings, then train the linear probe and write the fairness report. Please use the following commands:

```bash
# 0. Configure your fairness variables in pipeline/config.py (see above)

# 1. (Optional) Generate dummy embeddings to test the pipeline without a model
#    (one <ptid>.npy per subject, with a learnable class signal)
python pipeline/make_dummy_embeddings.py \
    --csv  /path/to/subjects.csv \
    --out  /path/to/embeddings \
    --dims 512

# 2. Extract embeddings from scans (one <ptid>.npy per subject)
python pipeline/embed_all.py \
    --csv     /path/to/subjects.csv \
    --ckpt    /path/to/backbone.ckpt \
    --out-dir /path/to/embeddings

# 3. Train LP + write fairness report
python pipeline/lp_fomo.py \
    --embeddings-dir /path/to/embeddings \
    --csv            /path/to/subjects.csv \
    --output-dir     results/my_model
```

### Outputs

| File | Contents |
|---|---|
| `predictions.json` | Per-sample label, prediction, and softmax scores |
| `eval_report.json` | Overall metrics + per-variable fairness report |

### Example `eval_report.json`

```json
{
  "feature_dim": 512,
  "n_train": 240, "n_val": 60, "n_test": 300,
  "best_lr": 0.001, "best_val_auroc": 0.91,
  "overall": { "ovr_f1": 0.72, "ovr_auroc": 0.89 },
  "per_variable": {
    "variable_1": { "groups": {"group_a": 146, "group_b": 154}, "disparities": {"ovr_f1": 0.04} },
    "variable_2": { "groups": {"bin_0": 17, "bin_1": 91, "bin_2": 133, "bin_3": 59}, "disparities": {"ovr_f1": 0.07} }
  },
  "fairness_score": {
    "ovr_f1": { "score": 0.94, "variables_used": ["variable_1", "variable_2"] }
  }
}
```

## Dataset split

Subjects are split into train/val/test with a stratified, reproducible split
(per-class, with a configurable number of subjects per split):

| Split | Per class |
|---|---|
| Train | 40 % |
| Val | 10 % |
| Test | 50 % |

Fairness evaluation runs on the **test set only**.  
Split is reproducible with `seed=42`.

## Fairness metrics

Computed by [`fomo-metrics`](https://github.com/fomo26/fomo-metrics).

**Fairness variables:** one or more grouping variables read from the CSV, each
either categorical (grouped as-is) or continuous (bucketed into bins). Configure
them in `pipeline/config.py`.

**Maximum disparity** per variable *v* and metric *M*:

```
D_v(M) = max_g M_g  -  min_g M_g
```

**Fairness score** (higher = more equitable, 1.0 = perfect):

```
FairnessScore(M) = (1 / |V'|) * Σ_{v in V'} (1 - D_v(M))
```

## Embeddings format

The embedding model produces one file per subject:

```
embeddings/
    <ptid_1>.npy
    <ptid_2>.npy
    ...
```

Each `<ptid>.npy` contains a single 1-D `float32` array (embedding). All
subjects must share the same feature dimension `D`.

The file contains **no labels or metadata**. Labels and fairness variables are
pulled back from the subjects CSV by `ptid` filename.

## Repository structure

```
model_weights/                      folder to place your model checkpoint
pipeline/
    embed_all.py                    extract embeddings from scans
    make_dummy_embeddings.py        generate dummy embeddings for testing
    embedding_dataset.py            loads <ptid>.npy embeddings from a directory
    lp_fomo.py                      LP + fairness evaluation
    fairness_report.py              fairness helpers (binning, normalisation, report)
    identity_model.py               passthrough encoder used by the LP module
    config.py                       fairness-variable configuration  ← edit this
requirements.txt
```
