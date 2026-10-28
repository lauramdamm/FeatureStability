# Finger MI Feature Stability — Analysis Code

Analysis code accompanying the manuscript *"Sequence Structure Effects on Feature
Extraction Stability in Finger Motor Imagery"* (Damm, Jiang & Demosthenous,
University College London).

The repository contains two scripts that run in sequence:

| Script | Role |
|---|---|
| `FeatExtractFrontiers.py` | Loads pre-computed EEG features, runs classification, permutation testing and stability analysis, writes a cache of per-subject results |
| `FAF5.py` | Reads that cache and produces the manuscript's statistics, LaTeX tables, validation report and summary figures |

The split exists so that the expensive stage (200 permutations × 5 methods ×
20 subjects) runs once and is cached, while the reporting stage can be re-run
cheaply as the manuscript evolves.

---

## 1. Requirements

Python 3.9+ with:

```
mne
numpy
pandas
scipy
statsmodels
scikit-learn
matplotlib
```

Pin exact versions in `requirements.txt` before archiving — the cache is stored
with `pickle`, which is not guaranteed to load across major library versions
(see [Known issues](#8-known-issues)).

---

## 2. Expected directory layout

Both scripts expect the feature files produced by the upstream extraction step,
organised as:

```
<BASE>/
└── Subject01/
    └── Explore/
        ├── Sequential/
        │   └── AR/
        │       └── Feature Analysis/
        │           └── ar_features.npz
        └── Random/
            └── AR/
                └── Feature Analysis/
                    └── ar_features.npz
```

Notes:

- Subjects are scanned as `Subject01` … `Subject20`.
- Methods are `CWT`, `STFT`, `ST`, `AR`, `FBCSP`.
- Filename pattern is `<method>.lower()_features.npz`, **except** FBCSP, which
  is `csp_features.npz`.
- Each `.npz` must contain two arrays: `features` and `labels`.

Epoch files (`epochs_epo.fif`, `labels.npy`) are read by `LD.loadepo`, but that
helper is not called by the main pipeline — the pipeline operates on the
already-extracted feature files.

---

## 3. Configuration

Paths are currently hard-coded as module-level constants at the top of
`extraction_pipeline.py`:

```python
BASE       = r'C:\...\Subjects\Data Collection'
OUTPUT     = r'C:\...\Dual Analysis'
CACHE_DIR  = Path(r'C:\...\Dual Analysis\CACHE')
```

and again in the `__main__` block of `unified_analysis_pipeline.py`:

```python
CACHE_DIR = Path(r'C:\...\Dual Analysis\CACHE\20261016_175648')
```

**Edit these before running.** Recommended: promote them to a `config.yaml` or
argparse flags so the code is portable and does not publish local paths.

---

## 4. Usage

### Step 1 — Extract and cache results

```bash
python extraction_pipeline.py
```

Creates a timestamped cache directory `CACHE/<YYYYMMDD_HHMMSS>/` containing:

| File | Contents |
|---|---|
| `SubjectNN_<METHOD>_result.pkl` | Full per-subject, per-method result object |
| `all_results_agg.pkl` | List of all 100 result objects — **input to step 2** |
| `extraction_log.csv` | Success/failure log with sample counts and accuracies |
| `subject_method_summary.csv` | Flat summary of accuracies and permutation p-values |
| `finger_stability_patterns.csv` | Per-finger pattern classification (5 fingers × 100 rows) |

Runtime is dominated by the permutation tests (200 permutations × 3-fold CV per
condition per method per subject). Expect this to take hours, not minutes.

### Step 2 — Generate statistics and tables

Point the cache path at the timestamped directory from step 1, then:

```bash
python unified_analysis_pipeline.py
```

Or programmatically:

```python
from unified_analysis_pipeline import run_full_manuscript_analysis

results = run_full_manuscript_analysis(
    cache_dir='path/to/CACHE/20261016_175648',
    validate_against={
        'friedman_chi2_stability': 34.13,
        'ar_core_stable_pct': 100.0,
    },
)
```

Outputs are written to `<cache_dir>/ManuscriptOutput/`.

---

## 5. What each stage computes

### Extraction (`extraction_pipeline.py`)

**Classification.** 5-class one-vs-rest SVM (RBF kernel) with `StandardScaler`,
evaluated by stratified 3-fold cross-validation. Hyperparameters `C ∈ {0.1, 1, 10}`
and `γ ∈ {scale, 0.1, 1.0}` are tuned by an inner 3-fold grid search nested inside
each outer fold. Sequential and random conditions are modelled independently.

**Permutation testing.** 200 label shuffles per condition, each re-running the
full 3-fold CV. The p-value is the proportion of permuted accuracies ≥ the
observed accuracy. Note that permutations use fixed `C=1.0, γ='scale'` rather
than the tuned hyperparameters — a deliberate speed trade-off, but it means the
permutation null is not identical to the reported classifier.

**Stability.** For each finger, the mean feature vector is computed across trials
within each condition, and Pearson's *r* between the sequential and random means
indexes representational conservation. Amplitude modulation is quantified as
Cohen's *d* between trial-wise mean activations, with Welch's *t*-test and
Benjamini–Hochberg FDR correction across fingers.

**Pattern classification.** Both *r* and *d* are z-scored within each subject and
method, then split at z = 0 (the method median) into four quadrants:
Core-stable, Core-modulated, Context-dependent, Noisy. A fifth label, *Mixed*,
captures fingers where z-scores are undefined (zero variance in *r* or *d*, or a
single finger in the group).

**Counterbalancing.** Subjects in `GROUP_A = [1, 3, 5, 6, 9, 11, 12, 13, 14, 16]`
began with the sequential block; all others began with random. This is recorded
as `starting_paradigm` on every result.

### Reporting (`unified_analysis_pipeline.py`)

| Phase | Function | Produces |
|---|---|---|
| 1 | `load`, `build_subject_table` | Master 100-row subject × method table |
| 1 | `validate_against_pattern_csv` | Cross-check of rebuilt vs. cached pattern proportions |
| 2 (A1) | `analyse_classification_performance` | Accuracy vs. chance (20%), t-test or Wilcoxon by Shapiro–Wilk, FDR + Bonferroni |
| 2 (A3) | `analyse_omnibus_across_methods` | Friedman χ² across methods, per condition and for stability |
| 2 (A4) | `summarise_permutation_pvalues` | Stouffer Z and Fisher χ² aggregation of subject-level permutation p-values |
| 3 (B1) | `analyse_stability_performance_link` | Spearman correlations between stability metrics and accuracy, in three scopes |
| 4 | `generate_latex_tables` | Tables 2–5 |
| 5 | `validate_against_published` | Discrepancy report vs. supplied reference values |
| 6 | `_quick_figure_summary` | Placeholder bar chart |

The three correlation scopes are:

- `pooled_raw` — all subject-method rows pooled (carries a between-method confound)
- `pooled_centred` — both variables centred within method (**the headline analysis**)
- `per_method` — within-method correlations, minimum n = 6

FDR correction is applied within each scope family separately.

---

## 6. Output → manuscript mapping

| Output file | Manuscript element |
|---|---|
| `chance_level_tests.csv` | Table 2 (accuracy by cue order) |
| `omnibus_method_tests.csv` | Friedman χ² reported in Results |
| `permutation_pvalue_summary.csv` | Stouffer/Fisher statistics in Results |
| `stability_performance_link.csv` | Stability × performance paragraph |
| `table2_accuracy_by_cue_order.tex` | Table 2 |
| `table3_pattern_distribution.tex` | Table 3 |
| `table4_method_summary.tex` | Table 4 |
| `table5_modulation.tex` | Table 5 |
| `validation_report.json` | Internal QA — not published |
| `method_summary_quick.png` | Not used in the manuscript |

---

## 7. Reproducing the published analysis

1. Obtain the dataset (see [Data](#9-data-and-citation)).
2. Run the upstream feature extraction to populate the `Feature Analysis` folders.
3. Edit the path constants in both scripts.
4. `python extraction_pipeline.py` → note the timestamped cache directory.
5. Point `CACHE_DIR` in `unified_analysis_pipeline.py` at that directory.
6. `python unified_analysis_pipeline.py`.
7. Compare `validation_report.json` against the manuscript's reported values.

Determinism: all cross-validation uses `random_state=42`. Permutation testing
uses NumPy's global RNG and is **not** seeded, so permutation p-values will vary
slightly between runs. Set `np.random.seed(...)` at the top of
`PermutationTester.rsvmperm` if byte-identical reproducibility is required.

---

## 8. Known issues

These are documented rather than fixed so that the archived code matches the
analysis that produced the manuscript. Anyone extending the code should address
them.



**Code**

- **Absolute Windows paths** in both scripts, including a personal username.
  Replace with configuration before publishing.
- **`PUBLISHED_VALUES` embeds superseded numbers**, including the Friedman
  χ² = 34.13 that the pipeline itself contradicts. Rename to
  `LEGACY_MANUSCRIPT_VALUES` and comment that these are pre-correction targets,
  not ground truth.
- **`Analysis.pipeline()` discards its `output_dir` argument**, overwriting it
  with the module-level `OUTPUT` constant.
- **`MulticlassSVM` is defined but never used.** `Analysis.classbc` implements
  its own inline grid search, with a slightly different grid
  (`γ ∈ {scale, 0.1, 1.0}` vs. `{scale, 0.01, 0.1, 1.0}`).
- **Global `warnings.filterwarnings('ignore')`** suppresses genuine convergence
  and fitting warnings. Scope it to specific calls.
- **`_quick_figure_summary` is a placeholder.** The manuscript figures are
  produced elsewhere.
- **`pickle` cache is version-fragile.** Unpickling pandas objects across major
  versions frequently fails. Pin dependencies and state the versions used.
- **`loadepo` silently truncates epochs** to `len(labels)` on mismatch, printing
  an error but continuing.
- **`scanfeat` prints an npz path** whenever the parent folder exists, even if
  the npz itself does not.

**Statistical**

- **100% core-stable is arithmetically impossible** under the z-score median
  split. With 5 fingers, the deviations from the mean sum to zero, so at most 2
  values can lie strictly above it — a ceiling of 40%. The validator flags this
  correctly.
- **`Mixed` is a residual category**, arising when z-scores are undefined rather
  than from a substantive classification decision. Report it as such.
- **Cohen's *d* is computed on scalar trial means** (grand mean across all
  features per trial), not on the full feature vector. This is a coarse summary
  of modulation and should be described precisely.
- **Permutation p-values have a floor of 1/201 ≈ 0.005** with 200 permutations.
  Stouffer and Fisher aggregations clip at `1/(n_perm+1)` accordingly.

---

## 9. Data and citation

The dataset is available at **https://doi.org/10.5522/04/33332307**.

If you use this code, please cite:

> Damm, L. M., Jiang, D., & Demosthenous, A. (2026). *Sequence Structure Effects
> on Feature Extraction Stability in Finger Motor Imagery.* 

**Reporting**

- **Table 4's `Avg |d|` column is mislabelled.** It is written from
  `avg_abs_cohens_d`, which `build_subject_table` derives from
  `modulation_df['cohens_d']` — the sequential-vs-random condition effect, i.e.
  the same quantity as Table 5. The caption describes it as finger
  discriminability, which is a different metric and is not computed anywhere in
  this codebase.
- **Table 5 p-values are hardcoded to `NA`.** `p_raw` is computed in
  `generate_latex_tables` and then discarded, so the table is not reproducible
  from the code as shipped.
- **Table 4 writes SVM accuracy as a proportion** (e.g. `0.141`) under a column
  headed `SVM Acc (%)`. Multiply by 100 or relabel the column.
- **Table 2 recomputes confidence intervals** via `_ci95` instead of using the
  `ci_low` / `ci_high` values already stored on each row by `_chance_block`.
  Redundant, and a divergence risk if one path is edited and not the other.
- **Table 2 emits an extra `p_FDR` column** not present in the manuscript table.

### Licence

**MIT**

---

## 10. Contact

Laura M. Damm — laura.damm.23@ucl.ac.uk
Department of Electrical and Electronic Engineering, University College London
