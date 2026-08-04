"""
run_ablation_study.py
Complete ablation study runner for GNN architectures
"""

import inspect
import mlflow
import json
import os
import sys
from datetime import datetime
import pandas as pd
from pathlib import Path
import traceback
import time


def load_experiments_from_json(json_path: str, dataset_name: str = None) -> tuple[dict, str]:
    """Load experiment configs from a JSON file, merging defaults into each experiment.

    Returns (configs_dict, mlflow_exp_name).
    Keys starting with '_' are stripped; 'name' is promoted to dict key.
    """
    with open(json_path) as f:
        data = json.load(f)

    raw_defaults = {k: v for k, v in data.get("defaults", {}).items()
                    if not k.startswith("_")}
    mlflow_exp_name = raw_defaults.pop("mlflow_exp_name", "Ablation-Study-ISG-JSON")

    configs: dict = {}
    for exp in data["experiments"]:
        merged = {k: v for k, v in exp.items() if not k.startswith("_")}
        name = merged.pop("name")
        config = {**raw_defaults, **merged}
        config.pop("mlflow_exp_name", None)  # consumed at study level
        if dataset_name is not None:
            config["dataset_name"] = dataset_name
        configs[name] = config

    return configs, mlflow_exp_name

from ablation_configs import get_ablation_configs, get_quick_ablation_configs, get_custom_configs
from test_isg_graph import main as standard_main          # GCNConv / GATConv / TransformerConv / GINConv
from test_isg_graph_raw_GTN import main as gtn_main
from test_isg_graph_raw_GTN import main_cached as gtn_main_cached  # Hybrid / ISG-VanillaGTN / Enhanced-Hybrid
import utils
import test_utils

# GNN types handled by test_isg_graph.py (standard PyG convolutions)
STANDARD_GNN_TYPES = frozenset({'TransformerConv', 'GATConv', 'GCNConv', 'GINConv'})

# Pre-compute accepted parameter names for standard_main (no **kwargs → must filter)
_STANDARD_MAIN_PARAMS = frozenset(inspect.signature(standard_main).parameters.keys())


class AblationStudyRunner:
    """Manages and executes ablation studies"""
    
    def __init__(self, dataset_name='autext23', output_dir='./ablation_results'):
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        (self.output_dir / 'plots').mkdir(exist_ok=True)
        (self.output_dir / 'results').mkdir(exist_ok=True)
        
        self.results = []
        self.failed_experiments = []
        
    def run_study(self, mode='quick', custom_experiments=None):
        """
        Run ablation study
        
        Args:
            mode: 'quick' (7 exp), 'full' (28 exp), or 'custom'
            custom_experiments: List of experiment names for custom mode
        """
        # Get configurations
        if mode == 'quick':
            configs = get_quick_ablation_configs(self.dataset_name)
            #study_name = f"Quick-Ablation-{self.dataset_name}"
        elif mode == 'full':
            configs = get_ablation_configs(self.dataset_name)
            #study_name = f"Full-Ablation-{self.dataset_name}"
        elif mode == 'custom' and custom_experiments:
            configs = get_custom_configs(self.dataset_name, custom_experiments)
            #study_name = f"Custom-Ablation-{self.dataset_name}"
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        study_name = f"Ablation-Study-{mode}"

        print(f"\n{'='*80}")
        print(f"🚀 Starting {mode.upper()} Ablation Study")
        print(f"📊 Dataset: {self.dataset_name}")
        print(f"🔬 Total experiments: {len(configs)}")
        print(f"{'='*80}\n")
        
        # Set up MLflow
        mlflow.set_experiment(study_name)
        
        start_time = time.time()
        
        # Run each experiment
        for i, (exp_name, config) in enumerate(configs.items(), 1):
            if config['leave_out_sources']:
                if config['dataset_name'] == 'autext23':
                    config['leave_out_sources'] = ["tweets"] 
                elif config['dataset_name'] == 'autext24':
                    config['leave_out_sources'] = ["literary", "news"] 
                elif config['dataset_name'] == 'semeval24':
                    config['leave_out_sources'] = ["wikihow", "wikipedia"]
                elif config['dataset_name'] == 'coling24':
                    config['leave_out_sources'] = ["mage"]

            print(f"\n{'='*80}")
            print(f"[{i}/{len(configs)}] Running: {exp_name}")
            print(f"📝 {config.get('description', 'N/A')}")
            print(f"🤖 Model: {config.get('gnn_type', 'N/A')}")
            print(f"{'='*80}\n")
            
            exp_start = time.time()
            
            try:
                self._run_single_experiment(exp_name, config)
                exp_time = time.time() - exp_start
                print(f"\n✅ {exp_name} completed in {exp_time/60:.1f} minutes")
                
            except Exception as e:
                exp_time = time.time() - exp_start
                print(f"\n❌ {exp_name} FAILED after {exp_time/60:.1f} minutes")
                print(f"Error: {str(e)}")
                print(traceback.format_exc())
                
                self.failed_experiments.append({
                    'experiment': exp_name,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
                
                # Log failure
                self.results.append({
                    'experiment': exp_name,
                    'gnn_type': config.get('gnn_type', 'N/A'),
                    'description': config.get('description', ''),
                    'test_accuracy': 0.0,
                    'test_f1': 0.0,
                    'best_accuracy': 0.0,
                    'best_f1': 0.0,
                    'status': f'FAILED: {str(e)[:100]}'
                })
        
        total_time = time.time() - start_time
        
        # Save and analyze results
        self._save_results(mode)
        self._print_summary(total_time)
        
        return pd.DataFrame(self.results)
    
    def _run_single_experiment(self, exp_name, config):
        """Run a single experiment"""
        
        # Store and remove non-parameter keys
        description = config.pop('description', 'No description')
        config.pop('mlflow_exp_name', None)

        # Add required paths
        config['file_name_data'] = f"isg_data_{self.dataset_name}_{config['cut_off_dataset']}perc"
        config['output_dir'] = f'{test_utils.EXTERNAL_DISK_PATH}isg_graph'

        # If the PKL doesn't exist yet but build_graph=false, force a one-time build
        pkl_path = Path(config['output_dir']) / f"{config['file_name_data']}.pkl"
        if not pkl_path.exists() and not config.get('build_graph', False):
            print(f"⚠️  Graph PKL not found: {pkl_path.name}")
            print(f"   build_graph=false in config but PKL is missing — forcing build_graph=true for this run.")
            config['build_graph'] = True

        # Start MLflow run
        with mlflow.start_run(run_name=exp_name):
            # ==================== LOG PARAMETERS ====================
            mlflow.log_param("experiment_name", exp_name)
            mlflow.log_param("description", description)
            mlflow.log_param("gnn_type", config.get('gnn_type', 'N/A'))
            mlflow.log_param("dataset_name", self.dataset_name)

            # Log all config parameters
            for key, value in config.items():
                try:
                    mlflow.log_param(key, value)
                except Exception as e:
                    print(f"⚠️  Could not log param {key}: {e}")

            force_rebuild = config.get('build_graph', False)
            gnn_type = config.get('gnn_type', '')

            try:
                # ==================== RUN TRAINING ====================
                if gnn_type in STANDARD_GNN_TYPES:
                    # test_isg_graph.py — no **kwargs, filter to accepted params
                    filtered = {k: v for k, v in config.items() if k in _STANDARD_MAIN_PARAMS}
                    standard_main(**filtered)
                else:
                    # test_isg_graph_raw_GTN.py — Hybrid / GTN variants (accepts **kwargs)
                    gtn_main_cached(**config, use_cache=True, force_rebuild=force_rebuild)
                
                # ==================== GET RESULTS FROM MLFLOW ====================
                run = mlflow.active_run()
                client = mlflow.tracking.MlflowClient()
                run_data = client.get_run(run.info.run_id).data
                
                # Extract metrics
                test_acc = run_data.metrics.get('Final-Accuracy-test', 0.0)
                test_f1 = run_data.metrics.get('Final-F1Macro-test', 0.0)
                best_acc = run_data.metrics.get('Best-Accuracy-test', 0.0)
                best_f1 = run_data.metrics.get('Best-F1Macro-test', 0.0)
                val_acc = run_data.metrics.get('Final-Accuracy-val', 0.0)
                val_f1 = run_data.metrics.get('Final-F1Macro-val', 0.0)
                
                # ==================== LOG SUMMARY METRICS ====================
                # These will appear as top-level metrics in MLflow UI
                mlflow.log_metric("summary_test_accuracy", test_acc)
                mlflow.log_metric("summary_test_f1", test_f1)
                mlflow.log_metric("summary_best_accuracy", best_acc)
                mlflow.log_metric("summary_best_f1", best_f1)
                mlflow.log_metric("summary_val_accuracy", val_acc)
                mlflow.log_metric("summary_val_f1", val_f1)
                
                # Log status as a tag (since status is a string)
                mlflow.set_tag("status", "SUCCESS")
                mlflow.set_tag("experiment_group", exp_name.split('_')[0])  # A, B, C, D, E
                
                # ==================== STORE RESULTS LOCALLY ====================
                self.results.append({
                    'experiment': exp_name,
                    'gnn_type': config.get('gnn_type', 'N/A'),
                    'description': description,
                    'test_accuracy': test_acc,
                    'test_f1': test_f1,
                    'best_accuracy': best_acc,
                    'best_f1': best_f1,
                    'val_accuracy': val_acc,
                    'val_f1': val_f1,
                    'status': 'SUCCESS',
                    'mlflow_run_id': run.info.run_id
                })
                
                # ==================== PRINT RESULTS ====================
                print(f"\n{'='*60}")
                print(f"📊 RESULTS FOR: {exp_name}")
                print(f"{'='*60}")
                print(f"  Validation:")
                print(f"    Accuracy: {val_acc:.4f}")
                print(f"    F1 Score: {val_f1:.4f}")
                print(f"  Test:")
                print(f"    Accuracy: {test_acc:.4f}")
                print(f"    F1 Score: {test_f1:.4f}")
                print(f"  Best:")
                print(f"    Accuracy: {best_acc:.4f}")
                print(f"    F1 Score: {best_f1:.4f}")
                print(f"  Status: ✅ SUCCESS")
                print(f"  MLflow Run ID: {run.info.run_id}")
                print(f"{'='*60}\n")
                
            except Exception as e:
                # ==================== HANDLE FAILURE ====================
                error_msg = str(e)
                print(f"\n❌ EXPERIMENT FAILED: {exp_name}")
                print(f"Error: {error_msg}\n")
                
                # Log failure to MLflow
                mlflow.log_metric("summary_test_accuracy", 0.0)
                mlflow.log_metric("summary_test_f1", 0.0)
                mlflow.log_metric("summary_best_accuracy", 0.0)
                mlflow.log_metric("summary_best_f1", 0.0)
                mlflow.set_tag("status", "FAILED")
                mlflow.set_tag("error_message", error_msg[:250])  # MLflow tag limit
                
                # Store failure locally
                self.results.append({
                    'experiment': exp_name,
                    'gnn_type': config.get('gnn_type', 'N/A'),
                    'description': description,
                    'test_accuracy': 0.0,
                    'test_f1': 0.0,
                    'best_accuracy': 0.0,
                    'best_f1': 0.0,
                    'val_accuracy': 0.0,
                    'val_f1': 0.0,
                    'status': f'FAILED: {error_msg[:100]}',
                    'mlflow_run_id': mlflow.active_run().info.run_id
                })
                
                # Re-raise to be caught by outer try-except
                raise
    
    def run_json(self, json_path: str, exp_filter: str = None):
        """Run ablation study from a JSON experiments file."""
        configs, study_name = load_experiments_from_json(json_path, self.dataset_name)

        if exp_filter:
            if exp_filter not in configs:
                raise ValueError(f"Experiment '{exp_filter}' not found in {json_path}. "
                                 f"Available: {list(configs.keys())}")
            configs = {exp_filter: configs[exp_filter]}

        print(f"\n{'='*80}")
        print(f"Starting Ablation Study: {study_name}")
        print(f"Dataset: {self.dataset_name}")
        print(f"Experiments: {len(configs)}")
        print(f"JSON: {json_path}")
        print(f"{'='*80}\n")

        mlflow.set_experiment(study_name)
        start_time = time.time()

        lodo_map = {
            'autext23':  ["tweets"],
            'autext24':  ["literary", "news"],
            'semeval24': ["wikihow", "wikipedia"],
            'coling24':  ["mage"],
        }

        for i, (exp_name, config) in enumerate(configs.items(), 1):
            if config.get('leave_out_sources'):
                ds = config.get('dataset_name', self.dataset_name)
                config['leave_out_sources'] = lodo_map.get(ds, [])

            print(f"\n{'='*80}")
            print(f"[{i}/{len(configs)}] Running: {exp_name}")
            print(f"  gnn_type={config.get('gnn_type','?')}  "
                  f"layers={config.get('num_gnn_layers','?')}  "
                  f"heads={config.get('heads_gnn','?')}  "
                  f"pool={config.get('pooling_type','?')}  "
                  f"build_graph={config.get('build_graph','?')}")
            print(f"{'='*80}\n")

            exp_start = time.time()
            try:
                self._run_single_experiment(exp_name, config)
                print(f"\n[OK] {exp_name} completed in {(time.time()-exp_start)/60:.1f} min")
            except Exception as e:
                elapsed = (time.time() - exp_start) / 60
                print(f"\n[FAILED] {exp_name} after {elapsed:.1f} min: {e}")
                print(traceback.format_exc())
                self.failed_experiments.append({
                    'experiment': exp_name,
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                })
                self.results.append({
                    'experiment': exp_name,
                    'gnn_type': config.get('gnn_type', 'N/A'),
                    'description': '',
                    'test_accuracy': 0.0, 'test_f1': 0.0,
                    'best_accuracy': 0.0, 'best_f1': 0.0,
                    'status': f'FAILED: {str(e)[:100]}',
                })

        self._save_results('json')
        self._print_summary(time.time() - start_time)
        return pd.DataFrame(self.results)

    def _save_results(self, mode):
        """Save results to CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main results
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values('test_f1', ascending=False)
        
        results_file = self.output_dir / 'results' / f"ablation_{self.dataset_name}_{mode}_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        
        print(f"\n📁 Results saved to: {results_file}")
        
        # Save failures if any
        if self.failed_experiments:
            failures_file = self.output_dir / 'results' / f"failures_{self.dataset_name}_{mode}_{timestamp}.txt"
            with open(failures_file, 'w') as f:
                for failure in self.failed_experiments:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"Experiment: {failure['experiment']}\n")
                    f.write(f"Error: {failure['error']}\n")
                    f.write(f"\nTraceback:\n{failure['traceback']}\n")
            print(f"⚠️  Failure log saved to: {failures_file}")
        
        return results_df
    
    def _print_summary(self, total_time):
        """Print summary of results"""
        results_df = pd.DataFrame(self.results)
        successful = results_df[results_df['status'] == 'SUCCESS']
        
        print(f"\n{'='*80}")
        print(f"📊 ABLATION STUDY COMPLETE!")
        print(f"{'='*80}")
        print(f"⏱️  Total time: {total_time/3600:.2f} hours")
        print(f"✅ Successful: {len(successful)}/{len(results_df)}")
        print(f"❌ Failed: {len(self.failed_experiments)}")
        print(f"{'='*80}\n")
        
        if len(successful) > 0:
            print("🏆 TOP 10 CONFIGURATIONS:\n")
            cols = ['experiment', 'gnn_type', 'description', 'test_f1', 'test_accuracy']
            top_results = successful.sort_values('test_f1', ascending=False)[cols].head(10)
            print(top_results.to_string(index=False))
            print()
            
            # Group by model type
            print("\n📈 RESULTS BY MODEL TYPE:\n")
            model_stats = successful.groupby('gnn_type').agg({
                'test_f1': ['mean', 'max', 'count'],
                'test_accuracy': ['mean', 'max']
            }).round(4)
            print(model_stats)


def analyze_ablation_results(results_csv):
    """
    Detailed analysis of ablation results
    """
    df = pd.read_csv(results_csv)
    df = df[df['status'] == 'SUCCESS']
    
    if len(df) == 0:
        print("❌ No successful experiments to analyze")
        return
    
    print("\n" + "="*80)
    print("📈 DETAILED ABLATION ANALYSIS")
    print("="*80 + "\n")
    
    # 1. Best overall
    best = df.loc[df['test_f1'].idxmax()]
    print(f"🥇 BEST OVERALL:")
    print(f"   Experiment: {best['experiment']}")
    print(f"   Model: {best['gnn_type']}")
    print(f"   Test F1: {best['test_f1']:.4f}")
    print(f"   Test Acc: {best['test_accuracy']:.4f}")
    print(f"   Description: {best['description']}\n")
    
    # 2. Baseline comparison
    baselines = df[df['experiment'].str.contains('baseline', case=False)]
    if len(baselines) > 0:
        baseline = baselines.iloc[0]
        improvement = (best['test_f1'] - baseline['test_f1']) * 100
        print(f"📍 BASELINE PERFORMANCE:")
        print(f"   Test F1: {baseline['test_f1']:.4f}")
        print(f"   Improvement (best vs baseline): +{improvement:.2f}%\n")
    
    # 3. Model type comparison
    print("🤖 PERFORMANCE BY MODEL TYPE:\n")
    model_comparison = df.groupby('gnn_type').agg({
        'test_f1': ['mean', 'std', 'max'],
        'test_accuracy': ['mean', 'std', 'max'],
        'experiment': 'count'
    }).round(4)
    model_comparison.columns = ['_'.join(col).strip() for col in model_comparison.columns.values]
    print(model_comparison)
    print()
    
    # 4. Feature contribution analysis
    print("🔍 FEATURE CONTRIBUTION ANALYSIS:\n")
    
    # Find experiments with specific features
    feature_analysis = {
        'POS Tags': df[df['experiment'].str.contains('pos', case=False)],
        'Token Distance': df[df['experiment'].str.contains('token', case=False)],
        'ISG Structural PE': df[df['experiment'].str.contains('structural', case=False)],
        'Global Pooling': df[df['experiment'].str.contains('global', case=False)],
        'Edge Concat': df[df['gnn_type'].str.contains('Hybrid|EdgeConcat', case=False)]
    }
    
    baseline_f1 = baselines.iloc[0]['test_f1'] if len(baselines) > 0 else df['test_f1'].min()
    
    for feature, feature_df in feature_analysis.items():
        if len(feature_df) > 0:
            avg_f1 = feature_df['test_f1'].mean()
            max_f1 = feature_df['test_f1'].max()
            improvement = ((avg_f1 - baseline_f1) / baseline_f1) * 100
            print(f"  {feature}:")
            print(f"    Avg F1: {avg_f1:.4f} (+{improvement:+.2f}%)")
            print(f"    Max F1: {max_f1:.4f}")
            print(f"    Count:  {len(feature_df)}")
            print()
    
    # 5. Ablation insights (remove one component)
    ablation_exps = df[df['experiment'].str.startswith('E')]
    if len(ablation_exps) > 0:
        print("🔬 ABLATION INSIGHTS (Component Removal Impact):\n")
        full_model = df[df['experiment'].str.contains('full', case=False) & 
                       ~df['experiment'].str.startswith('E')]
        if len(full_model) > 0:
            full_f1 = full_model.iloc[0]['test_f1']
            
            for _, row in ablation_exps.iterrows():
                drop = ((full_f1 - row['test_f1']) / full_f1) * 100
                component = row['description'].split('-')[1].strip() if '-' in row['description'] else 'unknown'
                print(f"  Remove {component}:")
                print(f"    F1: {row['test_f1']:.4f} ({drop:+.2f}% drop)")


def create_comparison_table(results_csv, output_path=None):
    """
    Create a publication-ready comparison table
    """
    df = pd.read_csv(results_csv)
    df = df[df['status'] == 'SUCCESS']
    
    # Select key experiments
    key_experiments = [
        'baseline',
        'vanilla.*edge.*concat',
        'isg.*full',
        'hybrid.*full'
    ]
    
    comparison = []
    for pattern in key_experiments:
        matches = df[df['experiment'].str.contains(pattern, case=False, regex=True)]
        if len(matches) > 0:
            best = matches.loc[matches['test_f1'].idxmax()]
            comparison.append({
                'Model': best['gnn_type'],
                'Configuration': best['description'],
                'Test F1': f"{best['test_f1']:.4f}",
                'Test Acc': f"{best['test_accuracy']:.4f}",
                'Best F1': f"{best['best_f1']:.4f}"
            })
    
    comp_df = pd.DataFrame(comparison)
    
    print("\n📋 COMPARISON TABLE:\n")
    print(comp_df.to_string(index=False))
    
    if output_path:
        comp_df.to_csv(output_path, index=False)
        print(f"\n💾 Comparison table saved to: {output_path}")
    
    return comp_df


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Run ablation study for GNN architectures')
    parser.add_argument('--dataset', type=str, default='autext23',
                        choices=['autext23', 'autext24', 'semeval24', 'coling24'],
                        help='Dataset to use')
    parser.add_argument('--mode', type=str, default='quick',
                        choices=['quick', 'full', 'custom'],
                        help='Ablation study mode (ignored when --experiments_json is set)')
    parser.add_argument('--experiments', type=str, nargs='+',
                        help='Custom experiment names (for custom mode)')
    parser.add_argument('--experiments_json', type=str, default=None,
                        help='Path to JSON experiments file. When set, runs JSON-defined experiments '
                             'instead of ablation_configs.py.')
    parser.add_argument('--exp', type=str, default=None,
                        help='Run a single experiment by name (requires --experiments_json).')
    parser.add_argument('--list', action='store_true',
                        help='List experiments in the JSON file and exit (requires --experiments_json).')
    parser.add_argument('--output_dir', type=str, default='./ablation_results',
                        help='Output directory')
    parser.add_argument('--analyze', type=str,
                        help='Path to results CSV for analysis only (skip training)')

    args = parser.parse_args()

    if args.analyze:
        print(f"Analyzing results from: {args.analyze}")
        analyze_ablation_results(args.analyze)
        create_comparison_table(args.analyze)
        return

    # ── JSON mode ────────────────────────────────────────────────────────────
    if args.experiments_json:
        if not os.path.exists(args.experiments_json):
            print(f"[ERROR] JSON file not found: {args.experiments_json}")
            sys.exit(1)

        if args.list:
            configs, study_name = load_experiments_from_json(args.experiments_json, args.dataset)
            print(f"\nExperiments in {args.experiments_json}  [{study_name}]:\n")
            for name, cfg in configs.items():
                print(f"  {name:<55}  gnn={cfg.get('gnn_type','?'):20s}  "
                      f"L={cfg.get('num_gnn_layers','?')}  "
                      f"H={cfg.get('heads_gnn','?')}  "
                      f"pool={cfg.get('pooling_type','?')}")
            print()
            sys.exit(0)

        runner = AblationStudyRunner(dataset_name=args.dataset, output_dir=args.output_dir)
        runner.run_json(args.experiments_json, exp_filter=args.exp)
        return

    # ── Legacy mode (ablation_configs.py) ────────────────────────────────────
    runner = AblationStudyRunner(
        dataset_name=args.dataset,
        output_dir=args.output_dir
    )

    runner.run_study(
        mode=args.mode,
        custom_experiments=args.experiments
    )


if __name__ == "__main__":
    main()