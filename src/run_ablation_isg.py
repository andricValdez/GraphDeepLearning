"""
run_ablation_isg.py  ·  Ablation study orchestrator for test_isg_graph.py
==========================================================================
Experiments are defined in experiments_ablation_isg.json (or the file you
pass with --experiments_json).  The JSON has a 'defaults' block merged into
every experiment, so only the varying fields need to be listed per entry.

Smart PKL reuse
---------------
Experiments that share the same language model, vocab settings, and feature
flags also share the same graph PKL on disk.  The orchestrator computes a
compact 'graph_key' from those params and embeds it in file_name_data, so
experiments that differ only in architecture (gnn_type, layers, heads …)
can reuse the same expensive ISG construction.

For each experiment the script:
  1. Merges defaults + experiment overrides into a full config.
  2. Computes graph_key and derives file_name_data from it.
  3. Checks if the PKL exists; if so, forces build_graph=False.
  4. Writes a temporary config JSON and calls:
       python test_isg_graph.py --config_path <tmp_config.json>
  5. Collects returncode and prints a final summary table.

Usage:
    python run_ablation_isg.py                                   # run all
    python run_ablation_isg.py --exp ablation_gcn_2L_mean_bn_2mp
    python run_ablation_isg.py --group arch_ablation             # by group
    python run_ablation_isg.py --dry_run                         # print cmds
    python run_ablation_isg.py --list                            # list exps
    python run_ablation_isg.py --force_build                     # rebuild
    python run_ablation_isg.py --experiments_json my_exps.json   # custom
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------
SRC_DIR       = Path(__file__).parent.resolve()
EXTERNAL_PATH = Path("/media/discoexterno/andric/data/experiments/isg_graph")

# ---------------------------------------------------------------------------
# Graph-key helpers
# Params that affect the saved PKL (graph topology + node features).
# Experiments that share the same values can reuse the same PKL.
# ---------------------------------------------------------------------------
LM_ALIASES = {
    "microsoft/deberta-v3-base":              "deberta",
    "FacebookAI/roberta-base":                "roberta",
    "google-bert/bert-base-uncased":          "bert",
    "google-bert/bert-base-multilingual-uncased": "mbert",
    "intfloat/multilingual-e5-large":         "me5",
}

# Default values assumed when a key is absent from the experiment config.
# Must mirror the 'defaults' block in the JSON.
GRAPH_DEFAULTS = {
    "lang_model_name":   "microsoft/deberta-v3-base",
    "min_df":            3,
    "max_features":      20000,
    "stop_words":        False,
    "special_chars":     False,
    "graph_type":        "undirected",
    "leave_out_sources": False,
    "add_pos_feat":      True,
    "add_domain_feat":   False,
    "reduce_dim_emb":    False,
    "reduced_dim":       256,
    "project_after_concat": False,
}


def compute_graph_key(cfg: dict) -> str:
    """
    Build a short human-readable key from graph-affecting params.
    Experiments with the same key share the same PKL.
    """
    parts = []

    lm = cfg.get("lang_model_name", GRAPH_DEFAULTS["lang_model_name"])
    parts.append(LM_ALIASES.get(lm, lm.split("/")[-1][:10]))

    min_df = cfg.get("min_df", GRAPH_DEFAULTS["min_df"])
    if min_df != GRAPH_DEFAULTS["min_df"]:
        parts.append(f"df{min_df}")

    mf = cfg.get("max_features", GRAPH_DEFAULTS["max_features"])
    if mf != GRAPH_DEFAULTS["max_features"]:
        parts.append(f"mf{mf}" if mf else "mfall")

    if cfg.get("stop_words", GRAPH_DEFAULTS["stop_words"]):
        parts.append("sw")
    if cfg.get("special_chars", GRAPH_DEFAULTS["special_chars"]):
        parts.append("sc")

    gt = cfg.get("graph_type", GRAPH_DEFAULTS["graph_type"])
    if gt != GRAPH_DEFAULTS["graph_type"]:
        parts.append(gt[:3])

    los = cfg.get("leave_out_sources", GRAPH_DEFAULTS["leave_out_sources"])
    if los:
        if isinstance(los, list):
            parts.append("lodo_" + "_".join(str(s) for s in los))
        else:
            parts.append("lodo")

    if not cfg.get("add_pos_feat", GRAPH_DEFAULTS["add_pos_feat"]):
        parts.append("nopos")
    if cfg.get("add_domain_feat", GRAPH_DEFAULTS["add_domain_feat"]):
        parts.append("dom")
    if cfg.get("reduce_dim_emb", GRAPH_DEFAULTS["reduce_dim_emb"]):
        rd = cfg.get("reduced_dim", GRAPH_DEFAULTS["reduced_dim"])
        parts.append(f"proj{rd}")

    return "_".join(parts)


def file_name_for(cfg: dict) -> str:
    graph_key = compute_graph_key(cfg)
    return f"isg_data_{cfg['dataset_name']}_{cfg['cut_off_dataset']}perc_{graph_key}"


# ---------------------------------------------------------------------------
# Load and merge experiments
# ---------------------------------------------------------------------------

def load_experiments(json_path: str) -> list[dict]:
    with open(json_path) as f:
        data = json.load(f)

    defaults = {k: v for k, v in data.get("defaults", {}).items()
                if not k.startswith("_")}

    experiments = []
    for exp in data["experiments"]:
        if "name" not in exp:
            continue
        merged = {**defaults, **{k: v for k, v in exp.items()
                                 if not k.startswith("_")}}
        merged["_group"] = exp.get("_group", "—")
        experiments.append(merged)
    return experiments


# ---------------------------------------------------------------------------
# PKL existence check
# ---------------------------------------------------------------------------

def graph_pkl_exists(cfg: dict) -> bool:
    pkl = EXTERNAL_PATH / f"{file_name_for(cfg)}.pkl"
    return pkl.exists()


# ---------------------------------------------------------------------------
# Run a single experiment
# ---------------------------------------------------------------------------

def run_experiment(exp: dict, dry_run: bool = False, force_build: bool = False) -> bool:
    cfg = dict(exp)
    name = cfg["name"]

    # Resolve build_graph
    if force_build:
        cfg["build_graph"] = True
    elif graph_pkl_exists(cfg):
        cfg["build_graph"] = False
    else:
        cfg["build_graph"] = True

    cfg["file_name_data"] = file_name_for(cfg)
    cfg["output_dir"]     = str(EXTERNAL_PATH)
    cfg["_done"]          = False

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix=f"ablation_{name}_"
    ) as f:
        json.dump(cfg, f, indent=2)
        cfg_path = f.name

    cmd = [sys.executable, str(SRC_DIR / "test_isg_graph.py"), "--config_path", cfg_path]

    print(f"\n{'='*70}")
    print(f"[RUN] {name}")
    print(f"  gnn_type       = {cfg.get('gnn_type')}")
    print(f"  num_gnn_layers = {cfg.get('num_gnn_layers')}")
    print(f"  heads_gnn      = {cfg.get('heads_gnn')}")
    print(f"  pooling_type   = {cfg.get('pooling_type')}")
    print(f"  norm_type      = {cfg.get('norm_type')}")
    print(f"  post_mp_layers = {cfg.get('post_mp_layers')}")
    print(f"  lang_model     = {cfg.get('lang_model_name','').split('/')[-1]}")
    print(f"  build_graph    = {cfg.get('build_graph')}")
    print(f"  file_name_data = {cfg['file_name_data']}")
    print(f"  cmd: {' '.join(cmd)}")

    if dry_run:
        print("[DRY RUN] Skipping execution.")
        return True

    t0     = time.time()
    result = subprocess.run(cmd, cwd=str(SRC_DIR))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"[ERROR] {name} failed (returncode={result.returncode})")
        return False

    print(f"[OK] {name} finished in {elapsed / 60:.1f} min")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ablation study orchestrator for test_isg_graph.py"
    )
    parser.add_argument(
        "--experiments_json", type=str,
        default=str(SRC_DIR / "experiments_ablation_isg.json"),
        help="Path to the experiments JSON (default: experiments_ablation_isg.json).",
    )
    parser.add_argument(
        "--exp", type=str, default=None,
        help="Run a single experiment by name.",
    )
    parser.add_argument(
        "--group", type=str, default=None,
        help="Run all experiments belonging to a specific _group tag.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--force_build", action="store_true",
        help="Force graph rebuild even if the PKL already exists.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available experiments and exit.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.experiments_json):
        print(f"[ERROR] Experiments file not found: {args.experiments_json}")
        sys.exit(1)

    experiments = load_experiments(args.experiments_json)

    # --list
    if args.list:
        print(f"\nExperiments in {args.experiments_json}:\n")
        header = (
            f"  {'Name':<55} {'group':<20} {'gnn_type':<22} "
            f"{'L':>2} {'H':>2} {'pool':<10} {'norm':<10}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for exp in experiments:
            grp = exp.get("_group", "—")
            print(
                f"  {exp['name']:<55} "
                f"{grp:<20} "
                f"{exp.get('gnn_type','?'):<22} "
                f"{exp.get('num_gnn_layers','?'):>2} "
                f"{exp.get('heads_gnn','?'):>2} "
                f"{exp.get('pooling_type','?'):<10} "
                f"{exp.get('norm_type','?'):<10}"
            )
        # Show unique groups
        groups = sorted({e.get("_group", "—") for e in experiments})
        print(f"\nGroups: {groups}\n")
        sys.exit(0)

    # Filter
    if args.exp:
        exps = [e for e in experiments if e["name"] == args.exp]
        if not exps:
            names = [e["name"] for e in experiments]
            print(f"[ERROR] Experiment '{args.exp}' not found.\nAvailable:\n  " +
                  "\n  ".join(names))
            sys.exit(1)
    elif args.group:
        exps = [e for e in experiments if e.get("_group") == args.group]
        if not exps:
            groups = sorted({e.get("_group", "—") for e in experiments})
            print(f"[ERROR] Group '{args.group}' not found.\nAvailable: {groups}")
            sys.exit(1)
    else:
        exps = experiments

    print(f"\nRunning {len(exps)} experiment(s).")
    print(f"Dry run: {args.dry_run}  |  Force build: {args.force_build}")
    print(f"JSON: {args.experiments_json}\n")

    summary = []
    for exp in exps:
        ok = run_experiment(exp, dry_run=args.dry_run, force_build=args.force_build)
        summary.append({"name": exp["name"], "group": exp.get("_group", "—"),
                        "status": "ok" if ok else "FAILED"})

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Experiment':<55} {'Group':<20} {'Status':^8}")
    print(f"  {'-'*55} {'-'*20} {'-'*8}")
    for s in summary:
        print(f"  {s['name']:<55} {s['group']:<20} {s['status']:^8}")
    print()

    failed = [s for s in summary if s["status"] != "ok"]
    if failed:
        print(f"[WARN] {len(failed)} experiment(s) failed: {[s['name'] for s in failed]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
