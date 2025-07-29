"""
Hyperparameter Optimization Tracking and Analysis Module

This module provides comprehensive tracking and visualization capabilities for HPO experiments,
including subject combination analysis, class distribution monitoring, fold composition review,
and hyperparameter sensitivity analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict, Counter
import json
import os
from datetime import datetime
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class HPOTracker:
    """Comprehensive HPO tracking and analysis system"""
    
    def __init__(self, output_dir="hpo_analysis", save_frequency=5):
        self.output_dir = output_dir
        self.save_frequency = save_frequency
        self.tracking_data = {
            'trial_numbers': [],
            'f1_scores': [],
            'fold_f1_scores': [],  # List of lists: each trial's fold scores
            'fold_details': [],    # Detailed fold information
            'subject_combinations': [],
            'hyperparameters': [],
            'training_data_sizes': [],
            'class_distributions': [],
            'cv_fold_count': [],
            'fold_compositions': [],  # Subject distribution per fold
            'undersampling_stats': [], # Before/after undersampling stats
            'timestamp': [],
            'device_used': []
        }
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize tracking file
        self.tracking_file = os.path.join(self.output_dir, "hpo_tracking_data.json")
        
        print(f"🔍 HPO Tracker initialized. Output directory: {self.output_dir}")
    
    def track_trial(self, trial_number, f1_scores, fold_details, selected_subjects, 
                   hyperparameters, training_data_size, class_distribution, 
                   cv_splits, fold_compositions=None, undersampling_stats=None, device=None):
        """Track a complete trial with all relevant metrics"""
        
        avg_f1 = np.mean(f1_scores)
        
        self.tracking_data['trial_numbers'].append(trial_number)
        self.tracking_data['f1_scores'].append(avg_f1)
        self.tracking_data['fold_f1_scores'].append(f1_scores)
        self.tracking_data['fold_details'].append(fold_details)
        self.tracking_data['subject_combinations'].append(selected_subjects)
        self.tracking_data['hyperparameters'].append(dict(hyperparameters))
        self.tracking_data['training_data_sizes'].append(training_data_size)
        self.tracking_data['class_distributions'].append(class_distribution)
        self.tracking_data['cv_fold_count'].append(cv_splits)
        self.tracking_data['fold_compositions'].append(fold_compositions or [])
        self.tracking_data['undersampling_stats'].append(undersampling_stats or {})
        self.tracking_data['timestamp'].append(datetime.now().isoformat())
        self.tracking_data['device_used'].append(device or "unknown")
        
        # Print trial summary
        self._print_trial_summary(trial_number, f1_scores, avg_f1, selected_subjects)
        
        # Save periodically
        if trial_number % self.save_frequency == 0:
            self.save_tracking_data()
    
    def _print_trial_summary(self, trial_number, f1_scores, avg_f1, subjects):
        """Print a summary of the current trial"""
        fold_scores_str = [f'{f:.3f}' for f in f1_scores]
        std_f1 = np.std(f1_scores)
        
        print(f"📊 Trial {trial_number}: F1 scores per fold: {fold_scores_str}")
        print(f"   Average F1 = {avg_f1:.4f}, Std = {std_f1:.4f}, Subjects = {subjects}")
    
    def save_tracking_data(self):
        """Save tracking data to JSON file"""
        try:
            with open(self.tracking_file, 'w') as f:
                json.dump(self.tracking_data, f, indent=2, default=str)
            print(f"💾 Tracking data saved to {self.tracking_file}")
        except Exception as e:
            print(f"❌ Error saving tracking data: {e}")
    
    def load_tracking_data(self, file_path=None):
        """Load tracking data from file"""
        file_path = file_path or self.tracking_file
        try:
            with open(file_path, 'r') as f:
                self.tracking_data = json.load(f)
            print(f"📂 Tracking data loaded from {file_path}")
        except Exception as e:
            print(f"❌ Error loading tracking data: {e}")
    
    def analyze_subject_combinations(self):
        """Detailed analysis of subject combination performance"""
        print(f"\n{'='*80}")
        print("👥 SUBJECT COMBINATION ANALYSIS")
        print('='*80)
        
        if not self.tracking_data['trial_numbers']:
            print("No data available for analysis")
            return {}
        
        combo_performance = defaultdict(list)
        
        # Group trials by subject combination
        for i, subjects in enumerate(self.tracking_data['subject_combinations']):
            combo_key = tuple(sorted(subjects))
            combo_performance[combo_key].append({
                'f1': self.tracking_data['f1_scores'][i],
                'trial': self.tracking_data['trial_numbers'][i],
                'size': self.tracking_data['training_data_sizes'][i],
                'fold_scores': self.tracking_data['fold_f1_scores'][i],
                'fold_std': np.std(self.tracking_data['fold_f1_scores'][i])
            })
        
        # Calculate statistics for each combination
        combo_stats = {}
        for combo, trials in combo_performance.items():
            f1_scores = [t['f1'] for t in trials]
            fold_stds = [t['fold_std'] for t in trials]
            
            combo_stats[combo] = {
                'mean_f1': np.mean(f1_scores),
                'std_f1': np.std(f1_scores),
                'min_f1': np.min(f1_scores),
                'max_f1': np.max(f1_scores),
                'mean_fold_variability': np.mean(fold_stds),
                'count': len(f1_scores),
                'trials': trials
            }
        
        # Sort by performance
        sorted_combos = sorted(combo_stats.items(), key=lambda x: x[1]['mean_f1'], reverse=True)
        
        print(f"Found {len(sorted_combos)} unique subject combinations across {len(self.tracking_data['trial_numbers'])} trials\n")
        
        print(f"{'Rank':<4} {'Mean F1':<8} {'Std F1':<8} {'Min F1':<8} {'Max F1':<8} {'Fold Var':<8} {'Trials':<7} {'Subjects'}")
        print(f"{'-'*4} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*50}")
        
        for rank, (combo, stats) in enumerate(sorted_combos, 1):
            print(f"{rank:<4} {stats['mean_f1']:<8.3f} {stats['std_f1']:<8.3f} {stats['min_f1']:<8.3f} {stats['max_f1']:<8.3f} {stats['mean_fold_variability']:<8.3f} {stats['count']:<7} {list(combo)}")
            
            # Show details for top/bottom combinations
            if rank <= 3 or rank > len(sorted_combos) - 3:
                print(f"   └─ Example trials:")
                for trial in stats['trials'][:2]:  # Show first 2 trials
                    print(f"      Trial {trial['trial']}: F1={trial['f1']:.3f}±{trial['fold_std']:.3f}, Size={trial['size']}")
                print()
        
        # Statistical analysis
        print(f"\n📈 SUBJECT COMBINATION INSIGHTS:")
        
        # Best vs worst combinations
        best_combo = sorted_combos[0]
        worst_combo = sorted_combos[-1]
        
        print(f"   Best combination: {list(best_combo[0])} (Mean F1: {best_combo[1]['mean_f1']:.3f})")
        print(f"   Worst combination: {list(worst_combo[0])} (Mean F1: {worst_combo[1]['mean_f1']:.3f})")
        print(f"   Performance gap: {best_combo[1]['mean_f1'] - worst_combo[1]['mean_f1']:.3f}")
        
        # Variability analysis
        combo_means = [stats['mean_f1'] for _, stats in sorted_combos]
        print(f"   Combination variability (std of means): {np.std(combo_means):.3f}")
        
        # Most/least consistent combinations
        most_consistent = min(sorted_combos, key=lambda x: x[1]['std_f1'])
        least_consistent = max(sorted_combos, key=lambda x: x[1]['std_f1'])
        
        print(f"   Most consistent: {list(most_consistent[0])} (Std: {most_consistent[1]['std_f1']:.3f})")
        print(f"   Least consistent: {list(least_consistent[0])} (Std: {least_consistent[1]['std_f1']:.3f})")
        
        return combo_stats
    
    def analyze_class_distributions(self):
        """Analyze class distribution patterns and their impact on performance"""
        print(f"\n{'='*80}")
        print("📊 CLASS DISTRIBUTION ANALYSIS")
        print('='*80)
        
        if not self.tracking_data['class_distributions']:
            print("No class distribution data available")
            return {}
        
        # Extract class distribution patterns
        class_patterns = defaultdict(list)
        
        for i, class_dist in enumerate(self.tracking_data['class_distributions']):
            # Create a pattern signature (sorted class counts)
            if isinstance(class_dist, dict):
                sorted_counts = tuple(sorted(class_dist.values()))
                class_patterns[sorted_counts].append({
                    'trial': self.tracking_data['trial_numbers'][i],
                    'f1': self.tracking_data['f1_scores'][i],
                    'distribution': class_dist,
                    'total_samples': sum(class_dist.values()),
                    'num_classes': len(class_dist),
                    'class_balance': min(class_dist.values()) / max(class_dist.values()) if class_dist.values() else 0
                })
        
        print(f"Found {len(class_patterns)} unique class distribution patterns\n")
        
        # Analyze each pattern
        pattern_stats = {}
        for pattern, trials in class_patterns.items():
            f1_scores = [t['f1'] for t in trials]
            balances = [t['class_balance'] for t in trials]
            
            pattern_stats[pattern] = {
                'mean_f1': np.mean(f1_scores),
                'std_f1': np.std(f1_scores),
                'count': len(trials),
                'mean_balance': np.mean(balances),
                'mean_total_samples': np.mean([t['total_samples'] for t in trials]),
                'trials': trials
            }
        
        # Sort by performance
        sorted_patterns = sorted(pattern_stats.items(), key=lambda x: x[1]['mean_f1'], reverse=True)
        
        print(f"{'Rank':<4} {'Mean F1':<8} {'Balance':<8} {'Samples':<8} {'Trials':<7} {'Pattern (class counts)'}")
        print(f"{'-'*4} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*30}")
        
        for rank, (pattern, stats) in enumerate(sorted_patterns, 1):
            print(f"{rank:<4} {stats['mean_f1']:<8.3f} {stats['mean_balance']:<8.3f} {stats['mean_total_samples']:<8.0f} {stats['count']:<7} {pattern}")
        
        # Correlation analysis
        all_balances = []
        all_f1s = []
        all_sizes = []
        
        for pattern, stats in pattern_stats.items():
            for trial in stats['trials']:
                all_balances.append(trial['class_balance'])
                all_f1s.append(trial['f1'])
                all_sizes.append(trial['total_samples'])
        
        if len(all_balances) > 2:
            balance_corr, balance_p = stats.pearsonr(all_balances, all_f1s)
            size_corr, size_p = stats.pearsonr(all_sizes, all_f1s)
            
            print(f"\n📈 CLASS DISTRIBUTION INSIGHTS:")
            print(f"   Class balance vs F1 correlation: r={balance_corr:.3f}, p={balance_p:.3f}")
            print(f"   Dataset size vs F1 correlation: r={size_corr:.3f}, p={size_p:.3f}")
        
        # Undersampling analysis if available
        if any(self.tracking_data['undersampling_stats']):
            self._analyze_undersampling_impact()
        
        return pattern_stats
    
    def _analyze_undersampling_impact(self):
        """Analyze the impact of undersampling on performance"""
        print(f"\n🔄 UNDERSAMPLING IMPACT ANALYSIS:")
        
        undersampling_effects = []
        
        for i, undersample_stats in enumerate(self.tracking_data['undersampling_stats']):
            if undersample_stats:
                f1 = self.tracking_data['f1_scores'][i]
                
                if 'before_size' in undersample_stats and 'after_size' in undersample_stats:
                    reduction_ratio = undersample_stats['after_size'] / undersample_stats['before_size']
                    undersampling_effects.append({
                        'trial': self.tracking_data['trial_numbers'][i],
                        'f1': f1,
                        'reduction_ratio': reduction_ratio,
                        'before_size': undersample_stats['before_size'],
                        'after_size': undersample_stats['after_size']
                    })
        
        if undersampling_effects:
            reduction_ratios = [e['reduction_ratio'] for e in undersampling_effects]
            f1_scores = [e['f1'] for e in undersampling_effects]
            
            if len(reduction_ratios) > 2:
                corr, p_val = stats.pearsonr(reduction_ratios, f1_scores)
                print(f"   Undersampling reduction vs F1 correlation: r={corr:.3f}, p={p_val:.3f}")
                
                avg_reduction = np.mean(reduction_ratios)
                print(f"   Average data reduction: {(1-avg_reduction)*100:.1f}%")
    
    def analyze_fold_compositions(self):
        """Analyze fold composition and its impact on performance"""
        print(f"\n{'='*80}")
        print("📁 FOLD COMPOSITION ANALYSIS")
        print('='*80)
        
        if not self.tracking_data['fold_details']:
            print("No fold composition data available")
            return {}
        
        # Analyze fold variability within trials
        fold_variability_stats = []
        
        for i, fold_details in enumerate(self.tracking_data['fold_details']):
            if fold_details:
                trial_f1 = self.tracking_data['f1_scores'][i]
                fold_f1s = [fold['f1_score'] for fold in fold_details if 'f1_score' in fold]
                
                if fold_f1s:
                    fold_variability_stats.append({
                        'trial': self.tracking_data['trial_numbers'][i],
                        'trial_f1': trial_f1,
                        'fold_f1s': fold_f1s,
                        'fold_std': np.std(fold_f1s),
                        'fold_min': np.min(fold_f1s),
                        'fold_max': np.max(fold_f1s),
                        'fold_range': np.max(fold_f1s) - np.min(fold_f1s),
                        'num_folds': len(fold_f1s)
                    })
        
        if not fold_variability_stats:
            print("No fold F1 score data available")
            return {}
        
        # Sort by fold variability
        sorted_by_variability = sorted(fold_variability_stats, key=lambda x: x['fold_std'], reverse=True)
        
        print(f"Analyzed {len(fold_variability_stats)} trials with fold-level F1 scores\n")
        
        print(f"{'Trial':<6} {'Trial F1':<8} {'Fold Std':<8} {'Fold Min':<8} {'Fold Max':<8} {'Range':<8} {'Folds':<6}")
        print(f"{'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
        
        # Show most variable trials
        print("Most variable trials:")
        for stat in sorted_by_variability[:5]:
            print(f"{stat['trial']:<6} {stat['trial_f1']:<8.3f} {stat['fold_std']:<8.3f} {stat['fold_min']:<8.3f} {stat['fold_max']:<8.3f} {stat['fold_range']:<8.3f} {stat['num_folds']:<6}")
        
        print("\nLeast variable trials:")
        for stat in sorted_by_variability[-5:]:
            print(f"{stat['trial']:<6} {stat['trial_f1']:<8.3f} {stat['fold_std']:<8.3f} {stat['fold_min']:<8.3f} {stat['fold_max']:<8.3f} {stat['fold_range']:<8.3f} {stat['num_folds']:<6}")
        
        # Statistical insights
        fold_stds = [s['fold_std'] for s in fold_variability_stats]
        trial_f1s = [s['trial_f1'] for s in fold_variability_stats]
        
        if len(fold_stds) > 2:
            variability_corr, variability_p = stats.pearsonr(fold_stds, trial_f1s)
            
            print(f"\n📈 FOLD COMPOSITION INSIGHTS:")
            print(f"   Fold variability vs Trial F1 correlation: r={variability_corr:.3f}, p={variability_p:.3f}")
            print(f"   Average fold standard deviation: {np.mean(fold_stds):.3f}")
            print(f"   Range of fold standard deviations: {np.min(fold_stds):.3f} - {np.max(fold_stds):.3f}")
            
            # Identify problematic folds
            high_var_threshold = np.percentile(fold_stds, 75)
            high_var_trials = [s for s in fold_variability_stats if s['fold_std'] > high_var_threshold]
            
            print(f"   High variability trials (top 25%): {len(high_var_trials)}")
            if high_var_trials:
                print(f"   Common characteristics of high-variability trials:")
                avg_f1_high_var = np.mean([t['trial_f1'] for t in high_var_trials])
                avg_f1_low_var = np.mean([t['trial_f1'] for t in fold_variability_stats if t['fold_std'] <= high_var_threshold])
                print(f"     - Average F1: {avg_f1_high_var:.3f} (vs {avg_f1_low_var:.3f} for low variability)")
        
        return fold_variability_stats
    
    def analyze_hyperparameter_sensitivity(self):
        """Analyze hyperparameter sensitivity and correlations with performance"""
        print(f"\n{'='*80}")
        print("🎛️ HYPERPARAMETER SENSITIVITY ANALYSIS")
        print('='*80)
        
        if len(self.tracking_data['hyperparameters']) < 10:
            print("Insufficient data for hyperparameter analysis (need at least 10 trials)")
            return {}
        
        # Extract all hyperparameters
        all_param_names = set()
        for hp in self.tracking_data['hyperparameters']:
            all_param_names.update(hp.keys())
        
        param_analysis = {}
        f1_scores = self.tracking_data['f1_scores']
        
        print(f"Analyzing {len(all_param_names)} hyperparameters across {len(f1_scores)} trials\n")
        
        # Analyze numeric parameters
        numeric_correlations = []
        
        for param_name in sorted(all_param_names):
            values = []
            corresponding_f1s = []
            
            for i, hp in enumerate(self.tracking_data['hyperparameters']):
                if param_name in hp:
                    val = hp[param_name]
                    if isinstance(val, (int, float)) and not np.isnan(val):
                        values.append(val)
                        corresponding_f1s.append(f1_scores[i])
            
            if len(values) >= 10:  # Need enough data points
                try:
                    correlation, p_value = stats.pearsonr(values, corresponding_f1s)
                    
                    param_analysis[param_name] = {
                        'type': 'numeric',
                        'correlation': correlation,
                        'p_value': p_value,
                        'values': values,
                        'f1_scores': corresponding_f1s,
                        'mean_value': np.mean(values),
                        'std_value': np.std(values),
                        'min_value': np.min(values),
                        'max_value': np.max(values)
                    }
                    
                    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                    numeric_correlations.append((param_name, correlation, p_value, significance))
                
                except Exception as e:
                    print(f"   Error analyzing {param_name}: {e}")
        
        # Sort by absolute correlation
        numeric_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print("NUMERIC PARAMETER CORRELATIONS:")
        print(f"{'Parameter':<20} {'Correlation':<12} {'P-value':<10} {'Significance':<12} {'Mean±Std'}")
        print(f"{'-'*20} {'-'*12} {'-'*10} {'-'*12} {'-'*20}")
        
        for param_name, corr, p_val, sig in numeric_correlations:
            stats_info = param_analysis[param_name]
            mean_std = f"{stats_info['mean_value']:.3f}±{stats_info['std_value']:.3f}"
            print(f"{param_name:<20} {corr:>11.3f} {p_val:>9.3f} {sig:>11s} {mean_std}")
        
        # Analyze categorical parameters
        print(f"\nCATEGORICAL PARAMETER ANALYSIS:")
        
        categorical_analysis = {}
        
        for param_name in sorted(all_param_names):
            values = []
            corresponding_f1s = []
            
            for i, hp in enumerate(self.tracking_data['hyperparameters']):
                if param_name in hp:
                    val = hp[param_name]
                    if not isinstance(val, (int, float)) or param_name in ['batch_size', 'num_tcn_layers']:
                        values.append(str(val))
                        corresponding_f1s.append(f1_scores[i])
            
            if len(values) >= 5:  # Need enough data points
                unique_values = list(set(values))
                if len(unique_values) > 1 and len(unique_values) <= 10:  # Reasonable number of categories
                    
                    value_stats = {}
                    for unique_val in unique_values:
                        indices = [i for i, v in enumerate(values) if v == unique_val]
                        val_f1s = [corresponding_f1s[i] for i in indices]
                        
                        value_stats[unique_val] = {
                            'count': len(val_f1s),
                            'mean_f1': np.mean(val_f1s),
                            'std_f1': np.std(val_f1s),
                            'f1_scores': val_f1s
                        }
                    
                    # ANOVA test if enough groups
                    if len(unique_values) >= 2:
                        groups = [value_stats[val]['f1_scores'] for val in unique_values if len(value_stats[val]['f1_scores']) > 0]
                        if len(groups) >= 2 and all(len(g) > 0 for g in groups):
                            try:
                                f_stat, anova_p = stats.f_oneway(*groups)
                                categorical_analysis[param_name] = {
                                    'type': 'categorical',
                                    'f_statistic': f_stat,
                                    'p_value': anova_p,
                                    'value_stats': value_stats,
                                    'unique_values': unique_values
                                }
                            except Exception as e:
                                print(f"   Error in ANOVA for {param_name}: {e}")
        
        # Display categorical results
        for param_name, analysis in categorical_analysis.items():
            significance = "***" if analysis['p_value'] < 0.001 else "**" if analysis['p_value'] < 0.01 else "*" if analysis['p_value'] < 0.05 else ""
            print(f"\n{param_name} (ANOVA F={analysis['f_statistic']:.2f}, p={analysis['p_value']:.3f} {significance}):")
            
            # Sort values by mean F1
            sorted_values = sorted(analysis['value_stats'].items(), key=lambda x: x[1]['mean_f1'], reverse=True)
            
            for value, stats in sorted_values:
                print(f"   {value}: F1={stats['mean_f1']:.3f}±{stats['std_f1']:.3f} (n={stats['count']})")
        
        # Summary insights
        print(f"\n📈 HYPERPARAMETER INSIGHTS:")
        
        if numeric_correlations:
            strongest_corr = numeric_correlations[0]
            print(f"   Strongest correlation: {strongest_corr[0]} (r={strongest_corr[1]:.3f})")
            
            significant_params = [p for p in numeric_correlations if p[2] < 0.05]
            print(f"   Significant numeric parameters: {len(significant_params)}/{len(numeric_correlations)}")
        
        if categorical_analysis:
            significant_categorical = [p for p, a in categorical_analysis.items() if a['p_value'] < 0.05]
            print(f"   Significant categorical parameters: {len(significant_categorical)}/{len(categorical_analysis)}")
        
        combined_analysis = {**param_analysis, **categorical_analysis}
        return combined_analysis
    
    def generate_comprehensive_visualizations(self):
        """Generate all visualizations in one comprehensive analysis"""
        print(f"\n{'='*80}")
        print("📊 GENERATING COMPREHENSIVE VISUALIZATIONS")
        print('='*80)
        
        if not self.tracking_data['trial_numbers']:
            print("No data available for visualization")
            return
        
        # Create main figure with subplots
        fig = plt.figure(figsize=(20, 24))
        gs = fig.add_gridspec(6, 3, hspace=0.3, wspace=0.3)
        
        # 1. F1 Score Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_f1_distribution(ax1)
        
        # 2. F1 Score Evolution
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_f1_evolution(ax2)
        
        # 3. Training Size vs F1
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_training_size_vs_f1(ax3)
        
        # 4. Subject Combination Performance
        ax4 = fig.add_subplot(gs[1, :])
        self._plot_subject_combinations(ax4)
        
        # 5. Hyperparameter Correlations
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_hyperparameter_correlations(ax5)
        
        # 6. Learning Rate Impact
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_learning_rate_impact(ax6)
        
        # 7. Fold Variability
        ax7 = fig.add_subplot(gs[2, 2])
        self._plot_fold_variability(ax7)
        
        # 8. Class Distribution Impact
        ax8 = fig.add_subplot(gs[3, 0])
        self._plot_class_distribution_impact(ax8)
        
        # 9. TCN Layer Analysis
        ax9 = fig.add_subplot(gs[3, 1])
        self._plot_tcn_layer_analysis(ax9)
        
        # 10. Device Performance
        ax10 = fig.add_subplot(gs[3, 2])
        self._plot_device_performance(ax10)
        
        # 11. Temporal Analysis
        ax11 = fig.add_subplot(gs[4, :])
        self._plot_temporal_analysis(ax11)
        
        # 12. Performance Distribution Heatmap
        ax12 = fig.add_subplot(gs[5, :])
        self._plot_performance_heatmap(ax12)
        
        plt.suptitle('Comprehensive HPO Analysis Dashboard', fontsize=20, y=0.98)
        
        # Save the comprehensive plot
        output_file = os.path.join(self.output_dir, 'comprehensive_hpo_analysis.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Comprehensive visualization saved to {output_file}")
        
        plt.show()
    
    def _plot_f1_distribution(self, ax):
        """Plot F1 score distribution"""
        all_fold_scores = [score for trial_scores in self.tracking_data['fold_f1_scores'] for score in trial_scores]
        trial_scores = self.tracking_data['f1_scores']
        
        box_data = [all_fold_scores, trial_scores]
        bp = ax.boxplot(box_data, labels=['Individual Folds', 'Trial Averages'], patch_artist=True)
        
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_title('F1 Score Distribution')
        ax.set_ylabel('F1 Score')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        fold_stats = f"Folds: μ={np.mean(all_fold_scores):.3f}, σ={np.std(all_fold_scores):.3f}"
        trial_stats = f"Trials: μ={np.mean(trial_scores):.3f}, σ={np.std(trial_scores):.3f}"
        ax.text(0.02, 0.98, f"{fold_stats}\n{trial_stats}", transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def _plot_f1_evolution(self, ax):
        """Plot F1 score evolution over trials"""
        trials = self.tracking_data['trial_numbers']
        scores = self.tracking_data['f1_scores']
        
        ax.plot(trials, scores, 'b-', alpha=0.7, label='Trial F1')
        ax.scatter(trials, scores, c='red', s=30, alpha=0.7, zorder=5)
        
        # Moving average
        if len(scores) > 3:
            window_size = min(5, len(scores))
            moving_avg = pd.Series(scores).rolling(window=window_size, center=True).mean()
            ax.plot(trials, moving_avg, 'g-', linewidth=3, label=f'Moving Avg ({window_size})')
        
        ax.set_title('F1 Score Evolution')
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('F1 Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_training_size_vs_f1(self, ax):
        """Plot training size vs F1 score"""
        sizes = self.tracking_data['training_data_sizes']
        scores = self.tracking_data['f1_scores']
        
        ax.scatter(sizes, scores, alpha=0.7, c='blue')
        
        # Add trend line
        if len(sizes) > 2:
            z = np.polyfit(sizes, scores, 1)
            p = np.poly1d(z)
            ax.plot(sizes, p(sizes), "r--", alpha=0.8)
            
            # Correlation
            corr = np.corrcoef(sizes, scores)[0, 1]
            ax.text(0.02, 0.98, f'Correlation: {corr:.3f}', transform=ax.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        ax.set_title('Training Size vs F1 Score')
        ax.set_xlabel('Training Data Size')
        ax.set_ylabel('F1 Score')
        ax.grid(True, alpha=0.3)
    
    def _plot_subject_combinations(self, ax):
        """Plot subject combination performance"""
        combo_performance = defaultdict(list)
        
        for i, subjects in enumerate(self.tracking_data['subject_combinations']):
            combo_key = tuple(sorted(subjects))
            combo_performance[combo_key].append(self.tracking_data['f1_scores'][i])
        
        if len(combo_performance) > 1:
            combo_names = [f"Combo {i+1}" for i in range(len(combo_performance))]
            combo_scores = list(combo_performance.values())
            
            bp = ax.boxplot(combo_scores, labels=combo_names, patch_artist=True)
            
            # Color by performance
            means = [np.mean(scores) for scores in combo_scores]
            colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(combo_scores)))
            
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_title('F1 Score by Subject Combination')
            ax.set_xlabel('Subject Combination')
            ax.set_ylabel('F1 Score')
            plt.setp(ax.get_xticklabels(), rotation=45)
        else:
            ax.text(0.5, 0.5, 'Only one subject combination found', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Subject Combinations (Insufficient Data)')
        
        ax.grid(True, alpha=0.3)
    
    def _plot_hyperparameter_correlations(self, ax):
        """Plot hyperparameter correlations"""
        # Extract numeric hyperparameters
        param_names = []
        correlations = []
        p_values = []
        
        all_param_names = set()
        for hp in self.tracking_data['hyperparameters']:
            all_param_names.update(hp.keys())
        
        f1_scores = self.tracking_data['f1_scores']
        
        for param_name in all_param_names:
            values = []
            corresponding_f1s = []
            
            for i, hp in enumerate(self.tracking_data['hyperparameters']):
                if param_name in hp:
                    val = hp[param_name]
                    if isinstance(val, (int, float)) and not np.isnan(val):
                        values.append(val)
                        corresponding_f1s.append(f1_scores[i])
            
            if len(values) >= 5:
                try:
                    corr, p_val = stats.pearsonr(values, corresponding_f1s)
                    param_names.append(param_name)
                    correlations.append(corr)
                    p_values.append(p_val)
                except:
                    continue
        
        if param_names:
            # Sort by absolute correlation
            sorted_indices = np.argsort(np.abs(correlations))[::-1]
            
            y_pos = np.arange(len(param_names))
            colors = ['red' if p < 0.05 else 'blue' for p in p_values]
            
            bars = ax.barh(y_pos, [correlations[i] for i in sorted_indices], 
                          color=[colors[i] for i in sorted_indices], alpha=0.7)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels([param_names[i] for i in sorted_indices])
            ax.set_xlabel('Correlation with F1 Score')
            ax.set_title('Hyperparameter Correlations')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3)
            
            # Add significance markers
            for i, (bar, idx) in enumerate(zip(bars, sorted_indices)):
                if p_values[idx] < 0.05:
                    ax.text(bar.get_width() + 0.01 if bar.get_width() > 0 else bar.get_width() - 0.01, 
                           bar.get_y() + bar.get_height()/2, '*', 
                           ha='left' if bar.get_width() > 0 else 'right', va='center')
        else:
            ax.text(0.5, 0.5, 'No numeric hyperparameters found', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Hyperparameter Correlations (No Data)')
    
    def _plot_learning_rate_impact(self, ax):
        """Plot learning rate impact on performance"""
        learning_rates = []
        f1_scores = []
        
        for i, hp in enumerate(self.tracking_data['hyperparameters']):
            if 'lr' in hp:
                learning_rates.append(hp['lr'])
                f1_scores.append(self.tracking_data['f1_scores'][i])
        
        if learning_rates:
            ax.scatter(learning_rates, f1_scores, alpha=0.7)
            ax.set_xscale('log')
            ax.set_xlabel('Learning Rate')
            ax.set_ylabel('F1 Score')
            ax.set_title('Learning Rate vs F1 Score')
            ax.grid(True, alpha=0.3)
            
            # Add correlation
            if len(learning_rates) > 2:
                log_lr = np.log10(learning_rates)
                corr = np.corrcoef(log_lr, f1_scores)[0, 1]
                ax.text(0.02, 0.98, f'Log-correlation: {corr:.3f}', transform=ax.transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'No learning rate data found', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Learning Rate Impact (No Data)')
    
    def _plot_fold_variability(self, ax):
        """Plot fold variability analysis"""
        fold_stds = []
        trial_f1s = []
        
        for i, fold_scores in enumerate(self.tracking_data['fold_f1_scores']):
            if fold_scores:
                fold_stds.append(np.std(fold_scores))
                trial_f1s.append(self.tracking_data['f1_scores'][i])
        
        if fold_stds:
            ax.scatter(trial_f1s, fold_stds, alpha=0.7)
            ax.set_xlabel('Trial Average F1 Score')
            ax.set_ylabel('F1 Standard Deviation Across Folds')
            ax.set_title('Trial Performance vs Fold Variability')
            ax.grid(True, alpha=0.3)
            
            # Add correlation
            if len(fold_stds) > 2:
                corr = np.corrcoef(trial_f1s, fold_stds)[0, 1]
                ax.text(0.02, 0.98, f'Correlation: {corr:.3f}', transform=ax.transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'No fold variability data found', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Fold Variability (No Data)')
    
    def _plot_class_distribution_impact(self, ax):
        """Plot class distribution impact"""
        # Calculate class balance for each trial
        balances = []
        f1_scores = []
        
        for i, class_dist in enumerate(self.tracking_data['class_distributions']):
            if isinstance(class_dist, dict) and class_dist:
                values = list(class_dist.values())
                if values:
                    balance = min(values) / max(values)
                    balances.append(balance)
                    f1_scores.append(self.tracking_data['f1_scores'][i])
        
        if balances:
            ax.scatter(balances, f1_scores, alpha=0.7)
            ax.set_xlabel('Class Balance (min/max)')
            ax.set_ylabel('F1 Score')
            ax.set_title('Class Balance vs F1 Score')
            ax.grid(True, alpha=0.3)
            
            # Add correlation
            if len(balances) > 2:
                corr = np.corrcoef(balances, f1_scores)[0, 1]
                ax.text(0.02, 0.98, f'Correlation: {corr:.3f}', transform=ax.transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightpink', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'No class distribution data found', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Class Distribution Impact (No Data)')
    
    def _plot_tcn_layer_analysis(self, ax):
        """Plot TCN layer count analysis"""
        layer_counts = []
        f1_scores = []
        
        for i, hp in enumerate(self.tracking_data['hyperparameters']):
            if 'num_tcn_layers' in hp:
                layer_counts.append(hp['num_tcn_layers'])
                f1_scores.append(self.tracking_data['f1_scores'][i])
        
        if layer_counts:
            # Group by layer count
            layer_performance = defaultdict(list)
            for layers, f1 in zip(layer_counts, f1_scores):
                layer_performance[layers].append(f1)
            
            if len(layer_performance) > 1:
                layers = sorted(layer_performance.keys())
                layer_f1s = [layer_performance[l] for l in layers]
                
                bp = ax.boxplot(layer_f1s, labels=layers, patch_artist=True)
                colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))
                
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                
                ax.set_xlabel('Number of TCN Layers')
                ax.set_ylabel('F1 Score')
                ax.set_title('TCN Layer Count vs Performance')
            else:
                ax.scatter(layer_counts, f1_scores, alpha=0.7)
                ax.set_xlabel('Number of TCN Layers')
                ax.set_ylabel('F1 Score')
                ax.set_title('TCN Layer Count vs Performance')
        else:
            ax.text(0.5, 0.5, 'No TCN layer data found', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('TCN Layer Analysis (No Data)')
        
        ax.grid(True, alpha=0.3)
    
    def _plot_device_performance(self, ax):
        """Plot device performance comparison"""
        device_performance = defaultdict(list)
        
        for i, device in enumerate(self.tracking_data['device_used']):
            if device != "unknown":
                device_performance[device].append(self.tracking_data['f1_scores'][i])
        
        if len(device_performance) > 1:
            devices = list(device_performance.keys())
            device_f1s = [device_performance[d] for d in devices]
            
            bp = ax.boxplot(device_f1s, labels=devices, patch_artist=True)
            colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
            
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(colors[i % len(colors)])
            
            ax.set_xlabel('Device')
            ax.set_ylabel('F1 Score')
            ax.set_title('Performance by Device')
        else:
            ax.text(0.5, 0.5, 'Insufficient device data for comparison', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Device Performance (Insufficient Data)')
        
        ax.grid(True, alpha=0.3)
    
    def _plot_temporal_analysis(self, ax):
        """Plot performance over time"""
        if self.tracking_data['timestamp']:
            try:
                timestamps = [datetime.fromisoformat(ts) for ts in self.tracking_data['timestamp']]
                f1_scores = self.tracking_data['f1_scores']
                
                ax.plot(timestamps, f1_scores, 'b-', alpha=0.7, marker='o')
                ax.set_xlabel('Time')
                ax.set_ylabel('F1 Score')
                ax.set_title('Performance Over Time')
                ax.grid(True, alpha=0.3)
                
                # Rotate x-axis labels
                plt.setp(ax.get_xticklabels(), rotation=45)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error plotting temporal data: {str(e)}', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title('Temporal Analysis (Error)')
        else:
            ax.text(0.5, 0.5, 'No timestamp data available', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Temporal Analysis (No Data)')
    
    def _plot_performance_heatmap(self, ax):
        """Plot performance heatmap combining multiple factors"""
        # Create a simplified heatmap of trial performance
        trials = self.tracking_data['trial_numbers']
        f1_scores = self.tracking_data['f1_scores']
        
        if len(trials) > 0:
            # Reshape data for heatmap (trials in grid format)
            n_trials = len(trials)
            grid_size = int(np.ceil(np.sqrt(n_trials)))
            
            # Pad with NaNs if necessary
            padded_scores = f1_scores + [np.nan] * (grid_size * grid_size - n_trials)
            heatmap_data = np.array(padded_scores).reshape(grid_size, grid_size)
            
            im = ax.imshow(heatmap_data, cmap='RdYlBu', aspect='auto')
            ax.set_title('Trial Performance Heatmap')
            ax.set_xlabel('Trial Grid Position')
            ax.set_ylabel('Trial Grid Position')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('F1 Score')
            
            # Add trial numbers as text
            for i in range(grid_size):
                for j in range(grid_size):
                    trial_idx = i * grid_size + j
                    if trial_idx < len(trials):
                        text = ax.text(j, i, str(trials[trial_idx]),
                                     ha="center", va="center", color="black", fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No performance data available', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Performance Heatmap (No Data)')
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print(f"\n{'='*100}")
        print("📋 COMPREHENSIVE HPO ANALYSIS REPORT")
        print('='*100)
        
        if not self.tracking_data['trial_numbers']:
            print("No data available for analysis")
            return
        
        n_trials = len(self.tracking_data['trial_numbers'])
        
        print(f"📊 OVERVIEW:")
        print(f"   Total Trials Analyzed: {n_trials}")
        print(f"   Analysis Period: {self.tracking_data['timestamp'][0] if self.tracking_data['timestamp'] else 'Unknown'} to {self.tracking_data['timestamp'][-1] if self.tracking_data['timestamp'] else 'Unknown'}")
        
        # Run all analyses
        combo_stats = self.analyze_subject_combinations()
        class_stats = self.analyze_class_distributions()
        fold_stats = self.analyze_fold_compositions()
        hyperparam_stats = self.analyze_hyperparameter_sensitivity()
        
        # Generate visualizations
        self.generate_comprehensive_visualizations()
        
        # Save final report
        self.save_tracking_data()
        
        print(f"\n✅ ANALYSIS COMPLETE!")
        print(f"   All results saved to: {self.output_dir}")
        print(f"   Tracking data: {self.tracking_file}")
        print(f"   Visualizations: {self.output_dir}/comprehensive_hpo_analysis.png")


def create_hpo_tracker(output_dir="hpo_analysis", save_frequency=5):
    """Factory function to create an HPO tracker"""
    return HPOTracker(output_dir=output_dir, save_frequency=save_frequency)
