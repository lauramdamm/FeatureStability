from datetime import datetime
import mne
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

BASE = r'C:\Users\uceelmd\OneDrive\Uni\ERDS Study\Subjects\Data Collection'
OUTPUT = r'C:\Users\uceelmd\OneDrive\Uni\ERDS Study\Dual Analysis'
CACHE_DIR = Path(r'C:\Users\uceelmd\OneDrive\Uni\ERDS Study\Dual Analysis\CACHE')
CACHE_DIR.mkdir(exist_ok=True)
METHODS = ['CWT','STFT', 'ST', 'AR', 'FBCSP']



class LD:
    def __init__(self):
        self.base = Path(BASE)

    @staticmethod
    def loadepo(subject_id, condition, run_num):
        epochdir = Path(BASE) / f'Subject{subject_id:02d}' / 'Explore' / condition / f'Run{run_num:02d}'
        
        epo_fif = epochdir / 'epochs_epo.fif'
        labels_npz = epochdir / 'labels.npy'
        
        if not epo_fif.exists() or not labels_npz.exists():
            return None, None
        
        raw_epochs = mne.read_epochs(str(epo_fif))
        
        epochs = raw_epochs.get_data()  # Shape: (n_events, n_channels, n_times)
        
        labels = np.load(labels_npz)
        
        # Ensure label count matches epochs count
        if len(labels) != epochs.shape[0]:
            print(f"[ERROR] Label count ({len(labels)}) doesn't match epochs ({epochs.shape[0]})")
            # Take first len(labels) epochs
            epochs = epochs[:len(labels)]
        
        return epochs, labels
    
    @staticmethod
    def scanfeat():
        all_paths = {}
        
        for sub_id in range(1,21):
            subject_dir = Path(BASE) / f'Subject{sub_id:02d}' / 'Explore'
            
            if not subject_dir.exists():
                continue
            
            all_paths[sub_id] = {}
            
            for method in METHODS:
                seq_folder = subject_dir / 'Sequential' / method / 'Feature Analysis'
                rand_folder = subject_dir / 'Random' / method / 'Feature Analysis'
                
                
                if seq_folder.exists():
                    if method == 'FBCSP':
                            npz_path = seq_folder / f'csp_features.npz'
                    else:
                            npz_path = seq_folder / f'{method.lower()}_features.npz'
                    print(f'{npz_path}')
                
                if rand_folder.exists():
                    if method == 'FBCSP':
                            npz_path = rand_folder / f'csp_features.npz'
                    else:
                            npz_path = rand_folder / f'{method.lower()}_features.npz'
                
                all_paths[sub_id][method] = {
                'sequential': seq_folder / f'csp_features.npz' if method == 'FBCSP' else seq_folder / f'{method.lower()}_features.npz',
                'random': rand_folder / f'csp_features.npz' if method == 'FBCSP' else rand_folder / f'{method.lower()}_features.npz'}
        
        return all_paths

        
    @staticmethod
    def loadfeat(npz_path):
        if not npz_path.exists():
            return None, None
        
        data = np.load(npz_path)
        features = data['features']
        labels = data['labels']

        return features, labels



class MulticlassSVM:
    def __init__(self, kernel='rbf', probability=False):
        self.kernel = kernel
        self.probability = probability
        self.model = None
        self.scaler = StandardScaler()
        self.best_params = None
    
    def featprep(self, X):
        if len(X.shape) == 3:
            return X.reshape(X.shape[0], -1)
        return X
    
    def train(self, X_train, y_train, cv_samples=3, verbose=True):
        X_train_flat = self.featprep(X_train)
        X_scaled = self.scaler.fit_transform(X_train_flat)
        
        param_grid = {'svc__C': [0.1, 1.0, 10.0, 100.0],'svc__gamma': ['scale', 0.01, 0.1, 1.0]}
        
        pipeline = Pipeline([('svc', SVC(kernel=self.kernel, probability=self.probability))])
        
        skf = StratifiedKFold(n_splits=cv_samples, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=skf,
            scoring='accuracy',
            n_jobs=1,  
            verbose=0 if verbose else 0,
            refit=True)
        
        grid_search.fit(X_scaled, y_train)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        if verbose:
            print(f"[INFO] Best SVM parameters: {self.best_params}")
            print(f"[INFO] Best SVM CV score: {grid_search.best_score_:.4f}")
        
        return self.best_params
    
    def predict(self, X_test):
        X_test_flat = self.featprep(X_test)
        X_test_scaled = self.scaler.transform(X_test_flat)
        return self.model.predict(X_test_scaled)
    
    def evaluate(self, X_test, y_test, verbose=True):
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        if verbose:
            print(f"\n[INFO] SVM Test Accuracy: {accuracy:.4f}")
            print(classification_report(y_test, predictions))
        
        return accuracy, predictions
    
    def save(self, path):
        pickle.dump({'model': self.model, 'scaler': self.scaler, 'params': self.best_params}, 
                    open(path, 'wb'))
        print(f"[INFO] SVM Model saved to {path}")
    
    def load(self, path):
        data = pickle.load(open(path, 'rb'))
        self.model = data['model']
        self.scaler = data['scaler']
        self.best_params = data['params']
        print(f"[INFO] SVM Model loaded from {path}")



class PermutationTester:             
    @staticmethod
    def rsvmperm(X, y, n_permutations=200, cv_samples=3):
        X_flat = X.reshape(X.shape[0], -1) if len(X.shape) == 3 else X
        
        # Use stratified CV for proper evaluation
        skf = StratifiedKFold(n_splits=cv_samples, shuffle=True, random_state=42)
        
        # Compute actual CV accuracy
        actual_fold_accs = []
        for train_idx, test_idx in skf.split(X_flat, y):
            X_train, X_test = X_flat[train_idx], X_flat[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            scaler = StandardScaler().fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            svc = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
            svc.fit(X_train_scaled, y_train)
            acc = svc.score(X_test_scaled, y_test)
            actual_fold_accs.append(acc)
        
        actual_cv_acc = np.mean(actual_fold_accs)
        
        # Permutation loop with full CV per permutation
        permuted_cv_accs = []
        for i in range(n_permutations):
            if (i + 1) % 50 == 0:
                print(f"      Perm {i+1}/{n_permutations}")
            
            y_shuffled = np.random.permutation(y)
            
            # Run full CV on shuffled labels
            perm_fold_accs = []
            for train_idx, test_idx in skf.split(X_flat, y_shuffled):
                X_train, X_test = X_flat[train_idx], X_flat[test_idx]
                y_train, y_test = y_shuffled[train_idx], y_shuffled[test_idx]
                
                scaler = StandardScaler().fit(X_train)
                X_train_scaled = scaler.transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                svc = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
                svc.fit(X_train_scaled, y_train)
                acc = svc.score(X_test_scaled, y_test)
                perm_fold_accs.append(acc)
            
            permuted_cv_accs.append(np.mean(perm_fold_accs))
        
        p_value = np.mean([a >= actual_cv_acc for a in permuted_cv_accs])
        
        return actual_cv_acc, np.array(permuted_cv_accs), p_value

class Analysis:
    def __init__(self):
        self.svm_by_condition_cache = {}
        self.covariance_cache = {}    


    def sub_med(self, all_results, output_dir):
        metrics = []
        
        for result in all_results:
            if result is None or len(result.get('classifications', {})) == 0:
                continue
            
            subj = result['subject']
            meth = result['method']
            classifications = result['classifications']
            
            core_stable_count = sum(1 for s in classifications.values() if s['pattern'] == 'Core-stable')
            total_fingers = len(classifications)

            pct_core_stable = 100 * core_stable_count / total_fingers if total_fingers > 0 else 0
            
            avg_corr = np.mean([s['correlation'] for s in classifications.values()])
            avg_cohens_d = np.abs(result['modulation_df']['cohens_d']).mean() if len(result['modulation_df']) > 0 else 0
            
            metrics.append({
                'subject': subj,
                'method': meth,
                'starting_paradigm': result.get('starting_paradigm'),
                'n_fingers': total_fingers,
                'pct_core_stable': pct_core_stable,
                'avg_correlation': avg_corr,
                'avg_cohens_d': avg_cohens_d,
                'svm_accuracy_seq': result.get('classification_accuracy_seq'), 
                'svm_accuracy_rand': result.get('classification_accuracy_rand'),  
                'seq_perm_p_value_svm': result.get('seq_perm_p_value_svm'),
                'rand_perm_p_value_svm': result.get('rand_perm_p_value_svm')})

        metrics_df = pd.DataFrame(metrics)
        filepath = os.path.join(output_dir,'subject_level_stability_metrics.csv' )
        metrics_df.to_csv(filepath, index=False)
        
        print(f"[INFO] Saved {len(metrics_df)} subject-method rows")
        return metrics_df

    
    def run_ana(self, seq_features, seq_labels, rand_features, rand_labels):
        unique_fingers = np.sort(np.unique(np.concatenate([seq_labels, rand_labels])))
        
        analysis_results = {'correlations': {},'modulation_records': []}
        
        for finger in unique_fingers:
            seq_mask = seq_labels == finger
            rand_mask = rand_labels == finger
            
            seq_finger = seq_features[seq_mask]
            rand_finger = rand_features[rand_mask]
            
            if len(seq_finger) < 3 or len(rand_finger) < 3:
                continue
            
            seq_flat = seq_finger.reshape(seq_finger.shape[0], -1).mean(axis=0)
            rand_flat = rand_finger.reshape(rand_finger.shape[0], -1).mean(axis=0)
            
            r = np.corrcoef(seq_flat, rand_flat)[0, 1]
            analysis_results['correlations'][int(finger)] = r if not np.isnan(r) else 0
                        
            seq_trial_means = seq_finger.mean(axis=1)  # One value per trial
            rand_trial_means = rand_finger.mean(axis=1)  # One value per trial
            diff = seq_trial_means.mean() - rand_trial_means.mean()
            pooled_std = np.sqrt((seq_trial_means.var(ddof=1) + rand_trial_means.var(ddof=1)) / 2)
            cohens_d = diff / pooled_std

            t_stat, p_val = stats.ttest_ind(seq_trial_means, rand_trial_means, equal_var=False)
            
            analysis_results['modulation_records'].append({
                'finger': int(finger),
                'cohens_d': cohens_d,
                't_stat': t_stat,
                'p_value': p_val,
                'n_seq': len(seq_finger),
                'n_rand': len(rand_finger),
                'mean_seq': seq_trial_means.mean(),
                'mean_rand': rand_trial_means.mean()})
        
        return analysis_results

    
    def fdrcorr(self, modulation_df, alpha=0.05):

        if len(modulation_df) == 0:
            modulation_df['significant_fdr'] = False
            modulation_df['p_adjusted'] = 1.0
            return modulation_df
        
        p_col = 'combined_p' if 'combined_p' in modulation_df.columns else 'p_value'
        raw_p_values = modulation_df[p_col].values
        
        reject, p_adjusted, _, _ = multipletests(raw_p_values, alpha=alpha, method='fdr_bh', returnsorted=False)
        
        modulation_df['significant_fdr'] = reject
        modulation_df['p_adjusted'] = p_adjusted
        
        return modulation_df
    
    
    def patterns(self, correlations, modulation_df, method,calibration_results=None):
        """
        Classify each finger using z-scored continuous metrics within method.
        
        relative positioning (above/below method mean), enabling direct cross-method comparison.
        
        Args:
            correlations: Dict mapping finger → r value
            modulation_df: DataFrame with per-finger modulation records
            method: Feature extraction method name
            calibration_results: Optional dict with null-derived thresholds
            
        Returns:
            classifications: Dict mapping finger → pattern metadata
        """
        if len(modulation_df) == 0:
            return {}
        
        finger_data = []
        for finger, r in correlations.items():
            finger_df = modulation_df[modulation_df['finger'] == finger]
            
            if len(finger_df) == 0:
                continue
            
            sig_val = float(finger_df['p_adjusted'].min()) if 'p_adjusted' in finger_df.columns else 1.0
            mean_d = float(finger_df['cohens_d'].mean())
            
            finger_data.append({
                'finger': int(finger),
                'r': r,
                'mean_d': mean_d,
                'sig_p': sig_val})
        
        finger_df = pd.DataFrame(finger_data)
        
        # Standardise within method
        finger_df['z_r'] = (finger_df['r'] - finger_df['r'].mean()) / finger_df['r'].std()
        finger_df['z_d'] = (finger_df['mean_d'] - finger_df['mean_d'].mean()) / finger_df['mean_d'].std()
        
        # Threshold at z = 0 (method median)
        HIGH_R_THRESHOLD = 0.0
        HIGH_D_THRESHOLD = 0.0
        
        classifications = {}
        for _, row in finger_df.iterrows():
            finger = int(row['finger'])
            r = row['r']
            d = row['mean_d']
            z_r = row['z_r']
            z_d = row['z_d']
            
            # Assign pattern based on relative position (unified across all methods)
            if z_r > HIGH_R_THRESHOLD and z_d <= HIGH_D_THRESHOLD:
                pattern = "Core-stable"
            elif z_r > HIGH_R_THRESHOLD and z_d > HIGH_D_THRESHOLD:
                pattern = "Core-modulated"
            elif z_r <= HIGH_R_THRESHOLD and z_d > HIGH_D_THRESHOLD:
                pattern = "Context-dependent"
            elif z_r <= HIGH_R_THRESHOLD and z_d <= HIGH_D_THRESHOLD:
                pattern = "Noisy"
            else:
                pattern = "Mixed"
            
            classifications[finger] = {
                'correlation': float(r),
                'mean_effect_size': float(d),
                'significance_p': float(row['sig_p']),
                'z_correlation': float(z_r),
                'z_modulation': float(z_d),
                'pattern': pattern,
                'threshold_mode': 'uniform_z-score',
                'calibration_results': calibration_results}
        
        return classifications


    def classbc (self, seq_feat, seq_lab, rand_feat, rand_lab, cv_samples=3):
        # === SEQUENTIAL CONDITION ===
        accs_seq = []
        for train_idx, val_idx in StratifiedKFold(
                n_splits=cv_samples, shuffle=True, random_state=42).split(seq_feat, seq_lab):
            X_tr, X_val = seq_feat[train_idx], seq_feat[val_idx]
            y_tr, y_val = seq_lab[train_idx], seq_lab[val_idx]
            
            scaler = StandardScaler().fit(X_tr.reshape(X_tr.shape[0], -1))
            X_tr_scaled = scaler.transform(X_tr.reshape(X_tr.shape[0], -1))
            X_val_scaled = scaler.transform(X_val.reshape(X_val.shape[0], -1))
            
            pipeline = Pipeline([('svc', SVC(kernel='rbf'))])
            param_grid = {'svc__C': [0.1, 1.0, 10.0], 'svc__gamma': ['scale', 0.1, 1.0]}
            grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy')
            grid.fit(X_tr_scaled, y_tr)
            
            accs_seq.append(grid.best_estimator_.score(X_val_scaled, y_val))
        
        # === RANDOM CONDITION ===
        accs_rand = []
        for train_idx, val_idx in StratifiedKFold(
                n_splits=cv_samples, shuffle=True, random_state=42).split(rand_feat, rand_lab):
            X_tr, X_val = rand_feat[train_idx], rand_feat[val_idx]
            y_tr, y_val = rand_lab[train_idx], rand_lab[val_idx]
            
            scaler = StandardScaler().fit(X_tr.reshape(X_tr.shape[0], -1))
            X_tr_scaled = scaler.transform(X_tr.reshape(X_tr.shape[0], -1))
            X_val_scaled = scaler.transform(X_val.reshape(X_val.shape[0], -1))
            
            pipeline = Pipeline([('svc', SVC(kernel='rbf'))])
            grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy')
            grid.fit(X_tr_scaled, y_tr)
            
            accs_rand.append(grid.best_estimator_.score(X_val_scaled, y_val))
        
        return {'accuracy_seq': np.mean(accs_seq),'accuracy_rand': np.mean(accs_rand)}


    def ana_ss(self, subject_id, method, seq_path, rand_path,starting_paradigm=None):
        print(f"\n[INFO] Subject {subject_id:02d} - {method})")
        


        seq_feat, seq_lab = LD.loadfeat(seq_path)
        rand_feat, rand_lab = LD.loadfeat(rand_path)
            
        
        if seq_feat is None or rand_feat is None:
            print(f"[ERROR] Missing feature data for Subject {subject_id:02d} - {method}")
            return None
        
        try:
            clf_result = self.classbc(seq_feat, seq_lab, rand_feat, rand_lab, cv_samples=3)
            
            try:
                actual_acc_svm, perm_accs_svm, seq_p_svm = PermutationTester.rsvmperm(seq_feat, seq_lab, n_permutations=200, cv_samples=3)
                rand_actual, _, rand_p_svm = PermutationTester.rsvmperm(rand_feat, rand_lab, n_permutations=200, cv_samples=3)
            except Exception as e:
                print(f"    [ERROR] SVM permutation failed: {e}")
                seq_p_svm = None
                rand_p_svm = None
            
        except Exception as e:
            print(f"  [ERROR] Classification failed: {e}")
            seq_p_svm = None
            rand_p_svm = None
            clf_result = {'accuracy_seq': None, 'accuracy_rand': None}
        
        # Feature Stability Analysis (single dataset)
        analysis_result = self.run_ana(seq_feat, seq_lab, rand_feat, rand_lab)
        
        # Apply FDR correction
        modulation_df = pd.DataFrame(analysis_result['modulation_records'])
        modulation_df = self.fdrcorr(modulation_df)
        
        # Classify patterns
        classifications = self.patterns(analysis_result['correlations'], modulation_df, method=method)
        
        
        result = {
            'subject': subject_id,
            'method': method,
            'starting_paradigm': starting_paradigm,
            'n_samples': len(seq_feat),
            'correlations': analysis_result['correlations'],
            'modulation_df': modulation_df,
            'classifications': classifications,
            'classification_accuracy_seq': clf_result.get('accuracy_seq'),
            'classification_accuracy_rand': clf_result.get('accuracy_rand'),
            'seq_perm_p_value_svm': seq_p_svm,
            'rand_perm_p_value_svm': rand_p_svm}
        
        return result

    
    def pipeline(self,output_dir=None):

        output_dir = OUTPUT
        

        print("[INFO] Scanning and loading feature files...")
        all_paths = LD.scanfeat()
        total_methods = sum(len(m) for m in all_paths.values())
        print(f"[INFO] Found {total_methods} subject-method combinations")
        

        print(f"[INFO] Feature-based classification...")
        GROUP_A = [1, 3, 5, 6, 9, 11, 12, 13, 14, 16]
        
        all_results = []
        iteration = 0
        
        for sub_id, methods in all_paths.items():
            starting = 'Sequential' if sub_id in GROUP_A else 'Random'
            
            for method, paths in methods.items():
                iteration += 1
                
                result = self.ana_ss(
                    sub_id, method, paths['sequential'], paths['random'],
                    starting_paradigm=starting
                )
                if result:
                    all_results.append(result)
        
        print(f"\n[INFO] Processed {len(all_results)} subject-method combinations")

        print(f"\n[INFO] Computing subject-level stability metrics...")
        subject_metrics = self.sub_med(all_results, output_dir)
        
        print(f"\n[INFO] Saving per-subject results...")
        for result in all_results:
            sub = result['subject']
            meth = result['method']

            class_df = pd.DataFrame([
                {'finger': k, **v}  # k is just finger now
                for k, v in result['classifications'].items()
            ])
            class_df.to_csv(output_dir / f'Subject{sub:02d}_{meth}_classifications.csv', index=False)
            result['modulation_df'].to_csv(output_dir / f'Subject{sub:02d}_{meth}_modulation.csv', index=False)

        print(f"[INFO] Results saved to: {output_dir}")

        
        return all_results



class ExtractionPipeline:
    def __init__(self, base_dir, output_dir=CACHE_DIR):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.analysis = Analysis()
        
    def extract(self):
        
        print("\n[INFO] Scanning feature files...")
        all_paths = LD.scanfeat()
        
        print(f"\n[INFO] Extracting feature-based metrics...")
        GROUP_A = [1, 3, 5, 6, 9, 11, 12, 13, 14, 16]
        
        all_per_method_results = []
        extraction_log = []
        
        for sub_id, methods in all_paths.items():
            starting = 'Sequential' if sub_id in GROUP_A else 'Random'
            
            for method, paths in methods.items():
                result = self.analysis.ana_ss(sub_id, method, paths['sequential'], paths['random'], starting_paradigm=starting)
                
                if result:
                    all_per_method_results.append(result)
                    
                    # Save individual subject-method result
                    subj_method_file = self.output_dir / f'Subject{sub_id:02d}_{method}_result.pkl'
                    with open(subj_method_file, 'wb') as f:
                        pickle.dump(result, f)
                    
                    extraction_log.append({
                        'subject': sub_id,
                        'method': method,
                        'status': 'SUCCESS',
                        'n_samples': result['n_samples'],
                        'svm_acc_seq': result.get('classification_accuracy_seq'),
                        'svm_acc_rand': result.get('classification_accuracy_rand')})
                    
                    print(f"[INFO] Subject{sub_id:02d}-{method}")
        

        log_df = pd.DataFrame(extraction_log)
        log_df.to_csv(self.output_dir / 'extraction_log.csv', index=False)
        

        agg_file = self.output_dir / 'all_results_agg.pkl'
        with open(agg_file, 'wb') as f:
            pickle.dump(all_per_method_results, f)
        print(f"\n[INFO] Extracted {len(all_per_method_results)} subject-method combos")
        print(f"[INFO] Aggregated results saved to {agg_file}")
        print(f"[INFO] Extraction log saved {log_df.shape[0]} to {self.output_dir / 'extraction_log.csv'}")
        
        return all_per_method_results
    
    def csvsum(self, all_results):
        
        summaries = []
        for res in all_results:
            summaries.append({
                'subject': res['subject'],
                'method': res['method'],
                'starting_paradigm': res.get('starting_paradigm'),
                'n_samples': res['n_samples'],
                'svm_accuracy_seq': res.get('classification_accuracy_seq'),
                'svm_accuracy_rand': res.get('classification_accuracy_rand'),
                'seq_perm_p_svm': res.get('seq_perm_p_value_svm'),
                'rand_perm_p_svm': res.get('rand_perm_p_value_svm'),})
        
        pd.DataFrame(summaries).to_csv(self.output_dir / 'subject_method_summary.csv', index=False)
        
        finger_patterns = []
        for res in all_results:
            for finger, stats in res['classifications'].items():
                finger_patterns.append({
                    'subject': res['subject'],
                    'method': res['method'],
                    'finger': finger,
                    **stats})
        
        pd.DataFrame(finger_patterns).to_csv(self.output_dir / 'finger_stability_patterns.csv', index=False)
        
        print(f"[INFO] Created subject_method_summary.csv and finger_stability_patterns.csv")




if __name__ == '__main__':
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    CACHE_SUBDIR = CACHE_DIR / timestamp
    CACHE_SUBDIR.mkdir(exist_ok=True)
    
    pipeline = ExtractionPipeline(BASE, CACHE_SUBDIR)
    all_results = pipeline.extract()
    pipeline.csvsum(all_results)
