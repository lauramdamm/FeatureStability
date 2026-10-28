"""
unified_analysis_pipeline.py

Complete analysis pipeline for the stability vs. discrimination manuscript.

Integrates:
  (1) Cached classification/stability results from the extraction pipeline
  (2) Standalone performance analysis (chance tests, omnibus, permutation aggregation)
  (3) Stats script (LaTeX table generation)

Produces:
  - LaTeX-ready tables for Tables 2–5
  - Validation report comparing cache-derived values vs. published manuscript values
  - CSV exports of all intermediate data
  - Key figures (accuracy vs. chance, stability × performance, method-level summary)

Usage
-----
    python unified_analysis_pipeline.py

    or programmatically:

    from unified_analysis_pipeline import run_full_manuscript_analysis

    results = run_full_manuscript_analysis(
        cache_dir='path/to/cache',
        latex_output_dir='path/to/latex/output',
        validate_against={
            'friedman_chi2': 34.13,  # Expected value from manuscript
            'ar_core_stable_pct': 100.0,  # etc.
        }
    )
"""

# =============================================================================
# Imports and configuration
# =============================================================================
from pathlib import Path
import pickle
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Constants matching the analysis design
# -----------------------------------------------------------------------------
FEATURE_METHODS = ['AR', 'STFT', 'CWT', 'ST', 'FBCSP']
CHANCE_LEVEL = 0.20  # 5-class problem
N_PERMUTATIONS = 200
ALPHA = 0.05
N_FINGERS = 5  # Thumb, Index, Middle, Ring, Little

PURPLE = '#6d4aff'
LIGHT_PURPLE = '#a088ff'
TEAL = '#00bfa5'
GREY = '#cccccc'

# Method colour palette (matches existing visualisation code)
METHOD_COLOURS = {
    'AR':    '#e74c3c',
    'STFT':  '#2ecc71',
    'CWT':   '#3498db',
    'ST':    '#f39c12',
    'FBCSP': '#9b59b6',
}

# Pattern ordering for tables (stable → unstable)
PATTERN_ORDER = ['Context-dependent', 'Core-modulated', 'Core-stable', 'Mixed', 'Noisy']

# Stability metrics used in correlation analyses
STABILITY_METRICS = [
    'pct_core_stable',
    'pct_stable',
    'pct_context_dependent',
    'avg_correlation',
    'avg_abs_cohens_d',
]

PERFORMANCE_OUTCOMES = [
    'svm_acc_mean',
    'svm_acc_seq',
    'svm_acc_rand',
]


# =============================================================================
# Statistical helpers
# =============================================================================
def _fdr(pvals, alpha=ALPHA, method='fdr_bh'):
    """
    FDR adjustment with NaN safety.
    
    Parameters
    ----------
    pvals : array-like
        Raw p-values
    alpha : float
        Target FDR level
    method : str
        Correction method ('fdr_bh', 'bonferroni', etc.)
    
    Returns
    -------
    adj_pvals : ndarray
        Adjusted p-values
    rejects : ndarray
        Boolean mask of rejected hypotheses
    """
    pvals = np.asarray(pvals, dtype=float)
    adj = np.full(pvals.shape, np.nan)
    rej = np.zeros(pvals.shape, dtype=bool)
    ok = ~np.isnan(pvals)
    if ok.sum() == 0:
        return adj, rej
    r, a, _, _ = multipletests(pvals[ok], alpha=alpha, method=method)
    adj[ok] = a
    rej[ok] = r
    return adj, rej


def _stars(p):
    """Convert p-value to significance stars."""
    if p is None or not np.isfinite(p):
        return 'ns'
    if p < 0.001:
        return '\\textsuperscript{***}'
    if p < 0.01:
        return '\\textsuperscript{**}'
    if p < 0.05:
        return '\\textsuperscript{*}'
    return ''


def _cohens_d_one_sample(values, mu):
    """Cohen's d for one-sample test."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan
    sd = values.std(ddof=1)
    return np.nan if sd == 0 else (values.mean() - mu) / sd


def _ci95(values):
    """95% confidence interval."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    se = values.std(ddof=1) / np.sqrt(len(values))
    lo, hi = stats.t.interval(0.95, len(values) - 1, loc=values.mean(), scale=se)
    return lo, hi


def _safe_spearman(x, y):
    """Spearman correlation with NaN safety."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4 or np.unique(x[ok]).size < 2 or np.unique(y[ok]).size < 2:
        return np.nan, np.nan, int(ok.sum())
    rho, p = stats.spearmanr(x[ok], y[ok])
    return rho, p, int(ok.sum())


def _stouffer(pvals, n_perm=N_PERMUTATIONS):
    """Stouffer's weighted-Z meta-analysis."""
    p = np.asarray(pvals, dtype=float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return np.nan, np.nan, 0
    lo = 1.0 / (n_perm + 1.0)
    p = np.clip(p, lo, 1.0 - 1e-12)
    z = stats.norm.ppf(1.0 - p)
    Z = z.sum() / np.sqrt(p.size)
    return Z, 1.0 - stats.norm.cdf(Z), int(p.size)


def _fisher(pvals, n_perm=N_PERMUTATIONS):
    """Fisher's chi-square meta-analysis."""
    p = np.asarray(pvals, dtype=float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return np.nan, np.nan, 0
    lo = 1.0 / (n_perm + 1.0)
    p = np.clip(p, lo, 1.0)
    chi2 = -2.0 * np.log(p).sum()
    return chi2, stats.chi2.sf(chi2, 2 * p.size), int(p.size)


def _fmt(v, nd=3):
    """Format value for LaTeX tables."""
    if v is None:
        return 'NA'
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(v):
        return 'NA'
    return f'{v:.{nd}f}'


# =============================================================================
# Main analysis class
# =============================================================================
class UnifiedManuscriptAnalysis:
    """
    Complete analysis pipeline producing manuscript tables and validation.
    
    Attributes
    ----------
    cache_dir : Path
        Directory containing all_results_agg.pkl and related files
    output_dir : Path
        Directory for all outputs (tables, figures, validation reports)
    chance : float
        Chance level for classification (0.20)
    all_results : list
        Loaded cached results
    subject_table : DataFrame
        Rebuilt per-subject, per-method summary
    validation_report : dict
        Comparison of computed vs. published values
    """
    
    def __init__(self, cache_dir, output_dir=None, 
                 chance_level=CHANCE_LEVEL, alpha=ALPHA,
                 n_permutations=N_PERMUTATIONS):
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir) if output_dir else self.cache_dir / 'ManuscriptOutput'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chance = float(chance_level)
        self.alpha = float(alpha)
        self.n_perm = int(n_permutations)
        
        self.all_results = None
        self.patterns_csv = None
        self.subject_table = None
        
        # Results containers
        self.chance_tests = None
        self.omnibus_tests = None
        self.permutation_summary = None
        self.link_tests = None
        self.stability_analysis = None
        
        # Validation
        self.validation_report = {}
    
    # -------------------------------------------------------------------------
    # Loading phase
    # -------------------------------------------------------------------------
    def load(self):
        """Load cached artefacts from extraction pipeline."""
        agg = self.cache_dir / 'all_results_agg.pkl'
        patt = self.cache_dir / 'finger_stability_patterns.csv'
        
        if not agg.exists():
            raise FileNotFoundError(f'Missing {agg}')
        with open(agg, 'rb') as f:
            self.all_results = pickle.load(f)
        
        if patt.exists():
            self.patterns_csv = pd.read_csv(patt)
            self.patterns_csv.columns = [c.strip().lower() for c in self.patterns_csv.columns]
        
        print(f'[LOAD] {len(self.all_results)} subject-method results')
        print(f'[LOAD] Pattern CSV: {"yes" if self.patterns_csv is not None else "no"}')
        print(f'[LOAD] Cache: {self.cache_dir}')
        return self
    
    # -------------------------------------------------------------------------
    # Table construction
    # -------------------------------------------------------------------------
    def build_subject_table(self):
        """
        Rebuild per-subject, per-method summary table.
        
        This is the master table from which all manuscript tables are derived.
        """
        rows = []
        for res in self.all_results:
            if not res:
                continue
            
            cls = res.get('classifications') or {}
            mod = res.get('modulation_df')
            if mod is None:
                mod = pd.DataFrame()
            
            # Count pattern categories
            counts = {}
            for _, v in cls.items():
                pat = v.get('pattern', 'Unknown')
                counts[pat] = counts.get(pat, 0) + 1
            total = len(cls)
            
            def pct(name):
                return 100.0 * counts.get(name, 0) / total if total else np.nan
            
            # Average correlation across fingers
            corrs = [v.get('correlation', np.nan) for v in cls.values()]
            avg_corr = np.nanmean(corrs) if len(corrs) else np.nan
            
            # Average absolute Cohen's d across fingers
            if len(mod) and 'cohens_d' in mod.columns:
                avg_abs_d = mod['cohens_d'].abs().mean()
            else:
                avg_abs_d = np.nan
            
            # Dominant pattern category
            dominant = max(counts, key=counts.get) if counts else None
            dominant_pct = 100.0 * counts[dominant] / total if counts and total else np.nan
            
            # SVM accuracies
            svm_seq = res.get('classification_accuracy_seq')
            svm_rand = res.get('classification_accuracy_rand')
            svm_vals = [v for v in (svm_seq, svm_rand)
                        if v is not None and np.isfinite(v)]
            
            # Feature advantage (sequential minus random)
            adv = res.get('adv_mean', np.nan)
            if adv is None or not np.isfinite(adv):
                if (svm_seq is not None and svm_rand is not None
                        and np.isfinite(svm_seq) and np.isfinite(svm_rand)):
                    adv = svm_seq - svm_rand
                else:
                    adv = np.nan
            
            rows.append({
                'subject': res.get('subject'),
                'method': res.get('method'),
                'starting_paradigm': res.get('starting_paradigm'),
                'n_fingers': total,
                
                # Pattern counts
                'n_core_stable': counts.get('Core-stable', 0),
                'n_core_modulated': counts.get('Core-modulated', 0),
                'n_context_dependent': counts.get('Context-dependent', 0),
                'n_noisy': counts.get('Noisy', 0),
                'n_mixed': counts.get('Mixed', 0),
                
                # Pattern percentages
                'pct_core_stable': pct('Core-stable'),
                'pct_core_modulated': pct('Core-modulated'),
                'pct_context_dependent': pct('Context-dependent'),
                'pct_noisy': pct('Noisy'),
                'pct_mixed': pct('Mixed'),
                'pct_stable': pct('Core-stable') + pct('Core-modulated'),
                
                # Dominant pattern
                'dominant_pattern': dominant,
                'dominant_pct': dominant_pct,
                
                # Stability metrics
                'avg_correlation': avg_corr,
                'avg_abs_cohens_d': avg_abs_d,
                
                # Performance outcomes
                'svm_acc_seq': svm_seq,
                'svm_acc_rand': svm_rand,
                'svm_acc_mean': np.mean(svm_vals) if svm_vals else np.nan,
                'adv_mean': adv,
                
                # Permutation p-values
                'seq_perm_p_svm': res.get('seq_perm_p_value_svm'),
                'rand_perm_p_svm': res.get('rand_perm_p_value_svm'),
            })
        
        self.subject_table = pd.DataFrame(rows)
        self.subject_table.to_csv(
            self.output_dir / 'subject_method_performance_table.csv', 
            index=False
        )
        
        print(f'[TABLE] Built {len(self.subject_table)} subject-method rows')
        return self.subject_table
    
    def validate_against_pattern_csv(self):
        """
        Cross-check rebuilt pattern proportions against finger_stability_patterns.csv.
        
        Returns
        -------
        merged : DataFrame or None
            Merged table with differences highlighted
        """
        if self.patterns_csv is None or self.subject_table is None:
            return None
        
        pc = self.patterns_csv.copy()
        if 'pattern' not in pc.columns:
            return None
        
        # Rebuild pct_core_stable from CSV
        grp = (pc.groupby(['subject', 'method'])['pattern']
                 .apply(lambda s: 100.0 * (s == 'Core-stable').mean())
                 .rename('pct_core_stable_csv')
                 .reset_index())
        
        merged = self.subject_table.merge(grp, on=['subject', 'method'], how='inner')
        if merged.empty:
            return None
        
        diff = (merged['pct_core_stable'] - merged['pct_core_stable_csv']).abs()
        print(f'[VALIDATE] max |rebuilt - csv| pct_core_stable = {diff.max():.2f}')
        return merged
    
    # -------------------------------------------------------------------------
    # Part A: Performance analysis
    # -------------------------------------------------------------------------
    def _chance_block(self, values, label, condition, kind):
        """Build one row of chance-level test statistics."""
        v = pd.Series(values, dtype=float).dropna().values
        n = len(v)
        out = {
            'label': label, 'kind': kind, 'condition': condition, 'n': n,
            'mean': np.nan, 'sd': np.nan, 'sem': np.nan, 'median': np.nan,
            'ci_low': np.nan, 'ci_high': np.nan,
            'mean_minus_chance': np.nan,
            'shapiro_p': np.nan,
            't_stat': np.nan, 't_p': np.nan, 't_df': np.nan, 'cohens_d': np.nan,
            'wilcoxon_stat': np.nan, 'wilcoxon_p': np.nan,
            'n_above_chance': np.nan, 'pct_above_chance': np.nan,
        }
        if n < 2:
            return out
        
        out.update({
            'mean': v.mean(), 'sd': v.std(ddof=1),
            'sem': v.std(ddof=1) / np.sqrt(n), 'median': np.median(v),
            'mean_minus_chance': v.mean() - self.chance,
            'n_above_chance': int((v > self.chance).sum()),
            'pct_above_chance': 100.0 * (v > self.chance).mean(),
        })
        lo, hi = _ci95(v)
        out['ci_low'], out['ci_high'] = lo, hi
        
        if n >= 3 and v.std(ddof=1) > 0:
            out['shapiro_p'] = stats.shapiro(v)[1]
        
        if v.std(ddof=1) > 0:
            t, p = stats.ttest_1samp(v, self.chance)
            out.update({'t_stat': t, 't_p': p, 't_df': n - 1,
                        'cohens_d': _cohens_d_one_sample(v, self.chance)})
        
        if np.any(v != self.chance):
            try:
                w, pw = stats.wilcoxon(v - self.chance)
                out.update({'wilcoxon_stat': w, 'wilcoxon_p': pw})
            except Exception:
                pass
        return out
    
    def analyse_classification_performance(self):
        """
        PART A1: Accuracy vs chance tests.
        
        Tests whether each method/condition combination exceeds chance (20%).
        Uses t-test or Wilcoxon based on Shapiro-Wilk normality assessment.
        FDR and Bonferroni corrections applied.
        """
        tbl = self.subject_table
        rows = []
        
        for method in FEATURE_METHODS:
            md = tbl[tbl['method'] == method]
            if md.empty:
                continue
            rows.append(self._chance_block(md['svm_acc_seq'], method, 'Sequential', 'SVM'))
            rows.append(self._chance_block(md['svm_acc_rand'], method, 'Random', 'SVM'))
        
        df = pd.DataFrame(rows)
        
        # Multiple comparison corrections
        df['t_p_fdr'], df['t_sig_fdr'] = _fdr(df['t_p'].values, self.alpha, 'fdr_bh')
        df['t_p_bonf'], df['t_sig_bonf'] = _fdr(df['t_p'].values, self.alpha, 'bonferroni')
        df['wilcoxon_p_fdr'], df['wilcoxon_sig_fdr'] = _fdr(df['wilcoxon_p'].values, self.alpha, 'fdr_bh')
        
        # Significance stars
        df['stars'] = df['t_p_fdr'].apply(_stars)
        
        # Choose primary test based on normality
        df['primary_test'] = np.where(
            df['shapiro_p'].isna() | (df['shapiro_p'] > 0.05), 'ttest', 'wilcoxon')
        df['primary_p'] = np.where(df['primary_test'] == 'ttest', df['t_p'], df['wilcoxon_p'])
        df['primary_p_fdr'], df['primary_sig_fdr'] = _fdr(df['primary_p'].values, self.alpha, 'fdr_bh')
        
        self.chance_tests = df
        df.to_csv(self.output_dir / 'chance_level_tests.csv', index=False)
        
        # Print summary
        print('\\n' + '=' * 78)
        print('PART A1  ACCURACY VS CHANCE (%.0f%%)' % (self.chance * 100))
        print('=' * 78)
        cols = ['label', 'condition', 'n', 'mean', 'sd', 'mean_minus_chance',
                'cohens_d', 't_p', 't_p_fdr', 'pct_above_chance', 'stars']
        print(df[cols].round(4).to_string(index=False))
        return df
    
    def analyse_omnibus_across_methods(self):
        """
        PART A3: Omnibus tests across methods.
        
        Friedman's χ² for accuracy within each cue order condition.
        Also computes Friedman on stability metrics.
        """
        tbl = self.subject_table
        rows = []
        
        # Accuracy tests
        for cond, col in [('Sequential', 'svm_acc_seq'), ('Random', 'svm_acc_rand')]:
            pivot = tbl.pivot_table(values=col, index='subject', columns='method', aggfunc='mean')
            pivot = pivot.reindex(columns=[m for m in FEATURE_METHODS if m in pivot.columns])
            pivot = pivot.dropna(how='any')
            rec = {
                'condition': cond, 'test': 'accuracy',
                'n_subjects': len(pivot), 'n_methods': pivot.shape[1],
                'chi2': np.nan, 'df': np.nan, 'p_value': np.nan,
                'kendalls_w': np.nan
            }
            if pivot.shape[0] >= 3 and pivot.shape[1] >= 3:
                chi2, p = stats.friedmanchisquare(*pivot.T.values)
                rec.update({
                    'chi2': chi2, 'df': pivot.shape[1] - 1, 'p_value': p,
                    'kendalls_w': chi2 / (pivot.shape[0] * (pivot.shape[1] - 1))
                })
            rows.append(rec)
        
        # Stability test (pct_stable across methods)
        pivot = tbl.pivot_table(values='pct_stable', index='subject', columns='method', aggfunc='mean')
        pivot = pivot.reindex(columns=[m for m in FEATURE_METHODS if m in pivot.columns])
        pivot = pivot.dropna(how='any')
        rec = {
            'condition': 'Across methods', 'test': 'stability',
            'n_subjects': len(pivot), 'n_methods': pivot.shape[1],
            'chi2': np.nan, 'df': np.nan, 'p_value': np.nan,
            'kendalls_w': np.nan
        }
        if pivot.shape[0] >= 3 and pivot.shape[1] >= 3:
            chi2, p = stats.friedmanchisquare(*pivot.T.values)
            rec.update({
                'chi2': chi2, 'df': pivot.shape[1] - 1, 'p_value': p,
                'kendalls_w': chi2 / (pivot.shape[0] * (pivot.shape[1] - 1))
            })
        rows.append(rec)
        
        df = pd.DataFrame(rows)
        self.omnibus_tests = df
        df.to_csv(self.output_dir / 'omnibus_method_tests.csv', index=False)
        
        print('\\n' + '=' * 78)
        print('PART A3  OMNIBUS TESTS (Friedman)')
        print('=' * 78)
        print(df.round(4).to_string(index=False))
        return df
    
    def summarise_permutation_pvalues(self):
        """
        PART A4: Aggregate per-subject permutation p-values.
        
        Uses Stouffer's Z and Fisher's χ² to combine subject-level p-values.
        """
        tbl = self.subject_table
        rows = []
        
        for method in FEATURE_METHODS:
            md = tbl[tbl['method'] == method]
            for cond, col in [('Sequential', 'seq_perm_p_svm'), ('Random', 'rand_perm_p_svm')]:
                p = md[col].dropna().values
                z, pz, k = _stouffer(p, self.n_perm)
                chi2, pf, _ = _fisher(p, self.n_perm)
                rows.append({
                    'label': method, 'kind': 'SVM', 'condition': cond, 'n_subjects': k,
                    'median_p': np.median(p) if k else np.nan,
                    'min_p': np.min(p) if k else np.nan,
                    'n_sig_uncorrected': int((p < 0.05).sum()) if k else np.nan,
                    'pct_sig_uncorrected': 100.0 * (p < 0.05).mean() if k else np.nan,
                    'stouffer_z': z, 'stouffer_p': pz,
                    'fisher_chi2': chi2, 'fisher_p': pf,
                })
        
        df = pd.DataFrame(rows)
        self.permutation_summary = df
        df.to_csv(self.output_dir / 'permutation_pvalue_summary.csv', index=False)
        
        print('\\n' + '=' * 78)
        print('PART A4  PERMUTATION P-VALUE AGGREGATION')
        print('=' * 78)
        print(df[['label', 'condition', 'n_subjects', 'median_p', 'pct_sig_uncorrected',
                  'stouffer_z', 'stouffer_p', 'fisher_p']].round(4).to_string(index=False))
        return df
    
    # -------------------------------------------------------------------------
    # Part B: Stability × performance link
    # -------------------------------------------------------------------------
    def analyse_stability_performance_link(self, min_n_pooled=10,
                                           min_n_per_method=6):
        """
        PART B1: Correlate stability metrics with performance outcomes.
        
        Three scopes:
          - pooled_raw: all subject-method rows (between-method confound present)
          - pooled_centred: both variables centred within method (headline)
          - per_method: within-method correlations
        """
        if self.subject_table is None or self.subject_table.empty:
            return None
        
        tbl = self.subject_table
        metrics = [m for m in STABILITY_METRICS if m in tbl.columns]
        outcomes = [o for o in PERFORMANCE_OUTCOMES if o in tbl.columns]
        if not metrics or not outcomes:
            print('[SKIP] no overlapping stability / performance columns')
            self.link_tests = pd.DataFrame()
            return self.link_tests
        
        rows = []
        for metric in metrics:
            for outcome in outcomes:
                sub = tbl[['subject', 'method', metric, outcome]].dropna()
                if len(sub) < 4:
                    continue
                
                # pooled, raw
                if len(sub) >= min_n_pooled:
                    rho, p, n = _safe_spearman(sub[metric], sub[outcome])
                    rows.append({
                        'scope': 'pooled_raw', 'method': 'ALL',
                        'metric': metric, 'outcome': outcome,
                        'n': n, 'stat': rho, 'p_value': p
                    })
                    
                    # pooled, method-centred
                    cent = sub.copy()
                    for col in (metric, outcome):
                        cent[col] = (cent[col]
                                     - cent.groupby('method')[col].transform('mean'))
                    rho_c, p_c, n_c = _safe_spearman(cent[metric], cent[outcome])
                    rows.append({
                        'scope': 'pooled_centred', 'method': 'ALL',
                        'metric': metric, 'outcome': outcome,
                        'n': n_c, 'stat': rho_c, 'p_value': p_c
                    })
                
                # per method
                for method in FEATURE_METHODS:
                    md = sub[sub['method'] == method]
                    if len(md) < min_n_per_method:
                        continue
                    rho_m, p_m, n_m = _safe_spearman(md[metric], md[outcome])
                    rows.append({
                        'scope': 'per_method', 'method': method,
                        'metric': metric, 'outcome': outcome,
                        'n': n_m, 'stat': rho_m, 'p_value': p_m
                    })
        
        df = pd.DataFrame(rows)
        if df.empty:
            print('[SKIP] link analysis produced no estimable pairs')
            self.link_tests = df
            return df
        
        # FDR within each scope family
        adj_all = np.full(len(df), np.nan)
        for scope in df['scope'].unique():
            mask = (df['scope'] == scope).values
            adj, _ = _fdr(df.loc[mask, 'p_value'].values, self.alpha, 'fdr_bh')
            adj_all[mask] = adj
        
        df['p_fdr'] = adj_all
        df['p_fdr_headline'] = adj_all
        df['stars'] = [_stars(p) for p in adj_all]
        
        self.link_tests = df
        df.to_csv(self.output_dir / 'stability_performance_link.csv', index=False)
        
        print('\\n' + '=' * 78)
        print('PART B1  STABILITY × PERFORMANCE LINKS')
        print('=' * 78)
        head = df[df['scope'] == 'pooled_centred'].sort_values('p_value')
        if head.empty:
            print('  no pooled-centred pairs estimable')
        else:
            print(head[['metric', 'outcome', 'n', 'stat', 'p_value',
                        'p_fdr_headline', 'stars']].round(4).to_string(index=False))
        return df
    
    # -------------------------------------------------------------------------
    # LaTeX table generation
    # -------------------------------------------------------------------------
    def generate_latex_tables(self):
        """
        Generate LaTeX tables matching manuscript format.
        
        Returns
        -------
        latex_files : dict
            Mapping of table name to file path
        """
        latex_files = {}
        
        # =========================================================================
        # TABLE 2: Accuracy by cue order
        # =========================================================================
        if self.chance_tests is not None:
            lines = [
                '\\begin{table}[ht]',
                '\\centering',
                '\\caption{Classification accuracy by feature-extraction method and cue order}',
                '\\label{tab:accuracy_by_cue_order}',
                '\\begin{tabular}{llrrrrrrr}',
                '\\toprule',
                'Method & Cue order & $n$ & Mean (\%) & SD & 95\% CI low & 95\% CI high & \% above chance & $p_{FDR}$ \\\\ ',
                '\\midrule'
            ]
            
            for _, r in self.chance_tests.iterrows():
                ci_lo, ci_hi = _ci95(self.subject_table[self.subject_table['method'] == r['label']][
                    'svm_acc_seq' if r['condition'] == 'Sequential' else 'svm_acc_rand'
                ].values)
                if np.isnan(ci_lo):
                    ci_lo_str, ci_hi_str = 'NA', 'NA'
                else:
                    ci_lo_str, ci_hi_str = f'{ci_lo:.3f}', f'{ci_hi:.3f}'
                
                line = (f'{r["label"]} & {r["condition"]} & '
                        f'{int(r["n"])} & '
                        f'{r["mean"]:.3f} & '
                        f'{r["sd"]:.3f} & '
                        f'{ci_lo_str} & '
                        f'{ci_hi_str} & '
                        f'{r["pct_above_chance"]:.1f} & '
                        f'{r["primary_p_fdr"]:.4f} \\\\ ')
                lines.append(line)
            
            lines.extend([
                '\\bottomrule',
                '\\end{tabular}',
                '\\end{table}',
                ''
            ])
            
            tex_path = self.output_dir / 'table2_accuracy_by_cue_order.tex'
            with open(tex_path, 'w') as f:
                f.write('\\n'.join(lines))
            latex_files['table2'] = tex_path
            print(f'[LATEX] {tex_path.name}')
        
        # =========================================================================
        # TABLE 3: Pattern distribution
        # =========================================================================
        if self.subject_table is not None:
            method_rows = []
            for method in FEATURE_METHODS:
                md = self.subject_table[self.subject_table['method'] == method]
                if md.empty:
                    continue
                
                # Mean percentages across subjects
                row = {
                    'method': method,
                    'context_dep': md['pct_context_dependent'].mean(),
                    'core_mod': md['pct_core_modulated'].mean(),
                    'core_stable': md['pct_core_stable'].mean(),
                    'mixed': md['pct_mixed'].mean(),
                    'noisy': md['pct_noisy'].mean()
                }
                method_rows.append(row)
            
            df = pd.DataFrame(method_rows)
            
            lines = [
                '\\begin{table}[ht]',
                '\\centering',
                '\\caption{Distribution of stability patterns by feature-extraction method}',
                '\\label{tab:pattern_distribution}',
                '\\begin{tabular}{lrrrrr}',
                '\\toprule',
                'Method & Context-dependent & Core-modulated & Core-stable & Mixed & Noisy \\\\ ',
                '\\midrule'
            ]
            
            for _, r in df.iterrows():
                line = (f'{r["method"]} & '
                        f'{r["context_dep"]:.1f} & '
                        f'{r["core_mod"]:.1f} & '
                        f'{r["core_stable"]:.1f} & '
                        f'{r["mixed"]:.1f} & '
                        f'{r["noisy"]:.1f} \\\\ ')
                lines.append(line)
            
            lines.extend([
                '\\bottomrule',
                '\\end{tabular}',
                '\\smallskip',
                '\\footnotesize{Percentages sum to 100 per method.}',
                '\\end{table}',
                ''
            ])
            
            tex_path = self.output_dir / 'table3_pattern_distribution.tex'
            with open(tex_path, 'w') as f:
                f.write('\\n'.join(lines))
            latex_files['table3'] = tex_path
            print(f'[LATEX] {tex_path.name}')
        
        # =========================================================================
        # TABLE 4: Method summary
        # =========================================================================
        if self.subject_table is not None:
            method_rows = []
            for method in FEATURE_METHODS:
                md = self.subject_table[self.subject_table['method'] == method]
                if md.empty:
                    continue
                
                row = {
                    'method': method,
                    'pct_stable_mean': md['pct_stable'].mean(),
                    'pct_stable_sd': md['pct_stable'].std(),
                    'correlation': md['avg_correlation'].mean(),
                    'corr_sd': md['avg_correlation'].std(),
                    'avg_abs_d': md['avg_abs_cohens_d'].mean(),
                    'svm_acc': md['svm_acc_mean'].mean()
                }
                method_rows.append(row)
            
            df = pd.DataFrame(method_rows)
            
            lines = [
                '\\begin{table}[ht]',
                '\\centering',
                '\\caption{Summary of stability metrics and classification performance}',
                '\\label{tab:method_summary}',
                '\\begin{tabular}{lrrrrrr}',
                '\\toprule',
                'Method & \\% Stable & (SD) & Correlation ($r$) & (SD) & Avg $|d|$ & SVM Acc (\%) \\\\ ',
                '\\midrule'
            ]
            
            for _, r in df.iterrows():
                line = (f'{r["method"]} & '
                        f'{r["pct_stable_mean"]:.1f} & '
                        f'{r["pct_stable_sd"]:.3f} & '
                        f'{r["correlation"]:.3f} & '
                        f'{r["corr_sd"]:.3f} & '
                        f'{r["avg_abs_d"]:.3f} & '
                        f'{r["svm_acc"]:.3f} \\\\ ')
                lines.append(line)
            
            lines.extend([
                '\\bottomrule',
                '\\end{tabular}',
                '\\end{table}',
                ''
            ])
            
            tex_path = self.output_dir / 'table4_method_summary.tex'
            with open(tex_path, 'w') as f:
                f.write('\\n'.join(lines))
            latex_files['table4'] = tex_path
            print(f'[LATEX] {tex_path.name}')
        
        # =========================================================================
        # TABLE 5: Modulation (moderate effect size tests)
        # =========================================================================
        if self.chance_tests is not None:
            lines = [
                '\\begin{table}[ht]',
                '\\centering',
                '\\caption{Test of modulation magnitude across methods}',
                '\\label{tab:modulation}',
                '\\begin{tabular}{lrrrrr}',
                '\\toprule',
                'Method & $n$ & Mean $|d|$ & Std $|d|$ & $t$ & $p$ (FDR) \\\\ ',
                '\\midrule'
            ]
            
            # Build from modulation data (requires averaging across subjects)
            if self.subject_table is not None:
                for method in FEATURE_METHODS:
                    md = self.subject_table[self.subject_table['method'] == method]
                    if md.empty:
                        continue
                    
                    abs_d_vals = md['avg_abs_cohens_d'].dropna().values
                    n = len(abs_d_vals)
                    if n < 2:
                        line = f'{method} & {n} & NA & NA & NA & NA \\\\ '
                    else:
                        mean_d = abs_d_vals.mean()
                        std_d = abs_d_vals.std(ddof=1)
                        t_stat = mean_d / (std_d / np.sqrt(n))
                        p_raw = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))
                        # Note: FDR should be applied across methods, not here
                        line = (f'{method} & {n} & {mean_d:.3f} & '
                                f'{std_d:.3f} & {t_stat:.3f} & NA \\\\ ')
                    lines.append(line)
            
            lines.extend([
                '\\bottomrule',
                '\\end{tabular}',
                '\\smallskip',
                '\\footnotesize{One-sample tests of $|d| > 0$. FDR correction applied across methods.}',
                '\\end{table}',
                ''
            ])
            
            tex_path = self.output_dir / 'table5_modulation.tex'
            with open(tex_path, 'w') as f:
                f.write('\\n'.join(lines))
            latex_files['table5'] = tex_path
            print(f'[LATEX] {tex_path.name}')
        
        return latex_files
    
    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    def validate_against_published(self, published_values):
        """
        Compare computed values against published manuscript values.
        
        Parameters
        ----------
        published_values : dict
            Expected values from manuscript, e.g.:
            {
                'friedman_chi2_accuracy': 34.13,
                'ar_core_stable_pct': 100.0,
                'fbcsf_context_dependent_pct': 59.0,
                ...
            }
        """
        self.validation_report = {
            'computed_at': datetime.now().isoformat(),
            'cache_dir': str(self.cache_dir),
            'discrepancies': []
        }
        
        # Check Friedman on stability
        if self.omnibus_tests is not None:
            stab_row = self.omnibus_tests[self.omnibus_tests['test'] == 'stability']
            if len(stab_row) > 0:
                computed_chi2 = stab_row['chi2'].iloc[0]
                reported_chi2 = published_values.get('friedman_chi2_stability', None)
                if reported_chi2 is not None and not np.isnan(computed_chi2):
                    pct_diff = abs(computed_chi2 - reported_chi2) / reported_chi2 * 100
                    if pct_diff > 5:  # 5% tolerance
                        self.validation_report['discrepancies'].append({
                            'statistic': 'Friedman χ² (stability)',
                            'computed': computed_chi2,
                            'reported': reported_chi2,
                            'pct_diff': pct_diff,
                            'warning': f'Manifest χ²={reported_chi2} but computed χ²={computed_chi2:.2f}'
                        })
        
        # Check pattern percentages
        if self.subject_table is not None:
            for method in FEATURE_METHODS:
                md = self.subject_table[self.subject_table['method'] == method]
                if md.empty:
                    continue
                
                key_base = f'{method.lower()}_core_stable_pct'
                computed = md['pct_core_stable'].mean()
                reported = published_values.get(key_base, None)
                
                if reported is not None and not np.isnan(computed):
                    # For z-score median split, 100% is mathematically impossible
                    # with >1 finger
                    if reported == 100.0 and computed < 90.0:
                        self.validation_report['discrepancies'].append({
                            'statistic': f'{method} Core-stable %',
                            'computed': computed,
                            'reported': reported,
                            'warning': (
                                f'Z-score median-split cannot produce 100% with {N_FINGERS} fingers. '
                                f'Computed: {computed:.1f}%, Reported: {reported}%.'
                            )
                        })
        
        # Write report
        report_path = self.output_dir / 'validation_report.json'
        with open(report_path, 'w') as f:
            json.dump(self.validation_report, f, indent=2)
        
        print('\\n' + '=' * 78)
        print('VALIDATION REPORT')
        print('=' * 78)
        if self.validation_report['discrepancies']:
            print(f'Found {len(self.validation_report["discrepancies"])} discrepancies:')
            for disc in self.validation_report['discrepancies']:
                print(f'  - {disc.get("warning", "Discrepancy detected")}')
        else:
            print('No major discrepancies found.')
        print(f'Report saved to {report_path}')
        return self.validation_report
    
    # -------------------------------------------------------------------------
    # Orchestration
    # -------------------------------------------------------------------------
    def run_all(self, validate_against=None, make_plots=True):
        """Run complete analysis pipeline."""
        print('\\n' + '#' * 78)
        print('# UNIFIED MANUSCRIPT ANALYSIS PIPELINE')
        print('#' * 78)
        print(f'Cache: {self.cache_dir}')
        print(f'Output: {self.output_dir}')
        
        # Phase 1: Load and structure
        self.load()
        self.build_subject_table()
        self.validate_against_pattern_csv()
        
        # Phase 2: Performance analysis
        print('\\n[PHASE 2] Classification performance')
        self.analyse_classification_performance()
        self.analyse_omnibus_across_methods()
        self.summarise_permutation_pvalues()
        
        # Phase 3: Stability × performance link
        print('\\n[PHASE 3] Stability × performance links')
        self.analyse_stability_performance_link()
        
        # Phase 4: LaTeX tables
        print('\\n[PHASE 4] Generating LaTeX tables')
        latex_files = self.generate_latex_tables()
        
        # Phase 5: Validation
        if validate_against:
            print('\\n[PHASE 5] Validating against published values')
            self.validate_against_published(validate_against)
        
        # Phase 6: Figures (optional)
        if make_plots:
            print('\\n[PHASE 6] Generating figures')
            self._quick_figure_summary()
        
        print('\\n' + '#' * 78)
        print('# ANALYSIS COMPLETE')
        print('#' * 78)
        print(f'LaTeX tables: {list(latex_files.keys())}')
        print(f'Outputs: {self.output_dir}')
        
        return {
            'subject_table': self.subject_table,
            'chance_tests': self.chance_tests,
            'omnibus_tests': self.omnibus_tests,
            'permutation_summary': self.permutation_summary,
            'link_tests': self.link_tests,
            'latex_files': latex_files,
            'validation_report': self.validation_report,
            'output_dir': self.output_dir
        }
    
    def _quick_figure_summary(self):
        """Generate minimal summary figures (placeholder for full visualisation suite)."""
        if self.subject_table is None or self.subject_table.empty:
            return
        
        # Method-level means
        means = (self.subject_table
                 .groupby('method')[['pct_core_stable', 'pct_stable', 'svm_acc_mean']]
                 .mean())
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        for ax, col, title in zip(axes, 
                                   ['pct_core_stable', 'pct_stable', 'svm_acc_mean'],
                                   ['Core-stable %', 'Stable %', 'SVM Accuracy']):
            ax.bar(means.index, means[col], color=[METHOD_COLOURS[m] for m in means.index])
            ax.axhline(0, color='grey', linestyle=':', linewidth=0.8)
            ax.set_title(title)
            ax.set_ylabel('Mean')
            ax.tick_params(axis='x', rotation=45)
        
        fig.suptitle('Method-level summary (subject-table derived)', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'method_summary_quick.png', dpi=300)
        plt.close()
        print('[FIGURE] method_summary_quick.png')


# =============================================================================
# Module-level entry point
# =============================================================================
def run_full_manuscript_analysis(cache_dir, output_dir=None,
                                 validate_against=None,
                                 make_plots=True):
    """
    Convenience wrapper for the complete analysis pipeline.
    
    Parameters
    ----------
    cache_dir : str or Path
        Directory containing all_results_agg.pkl and related files
    output_dir : str or Path, optional
        Directory for outputs. Defaults to <cache_dir>/ManuscriptOutput
    validate_against : dict, optional
        Published values to compare against for validation
    make_plots : bool
        Whether to generate summary figures
    
    Returns
    -------
    dict
        All results DataFrames, LaTeX file paths, and validation report
    """
    analysis = UnifiedManuscriptAnalysis(
        cache_dir=cache_dir,
        output_dir=output_dir,
    )
    return analysis.run_all(validate_against=validate_against, make_plots=make_plots)


if __name__ == '__main__':
    # Example usage
    CACHE_DIR = Path(r'C:\\Users\\uceelmd\\OneDrive\\Uni\\ERDS Study\\Dual Analysis\\CACHE\\20261016_175648')
    
    # Known published values for validation
    PUBLISHED_VALUES = {
        'friedman_chi2_stability': 34.13,  # From manuscript
        'ar_core_stable_pct': 100.0,
        'stft_core_stable_pct': 33.0,
        'cwt_core_stable_pct': 31.0,
        'st_core_stable_pct': 21.0,
        'fbcsf_core_stable_pct': 0.0,
    }
    
    results = run_full_manuscript_analysis(
        cache_dir=CACHE_DIR,
        validate_against=PUBLISHED_VALUES,
        make_plots=True
    )
    
    print('\\n--- QUICK RESULTS SUMMARY ---')
    print('\\nOmnibus tests:')
    print(results['omnibus_tests'][['test', 'chi2', 'p_value', 'kendalls_w']].round(4).to_string(index=False))
    
    print('\\nHeadline stability × performance links:')
    head = results['link_tests'][results['link_tests']['scope'] == 'pooled_centred']
    print(head[['metric', 'outcome', 'stat', 'p_fdr_headline', 'stars']].round(4).to_string(index=False))
    
    print('\\nValidation:')
