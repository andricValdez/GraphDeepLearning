"""
run_ablation_cooc.py  ·  Ablation study orchestrator for test_cooc_graph.py
============================================================================
Experiments are defined in experiments_ablation_cooc_{gcn,gat,transformer}.json
(or any file passed with --experiments_json).  The JSON has a 'defaults' block
merged into every experiment, so only the varying fields need to be listed per
entry.

Smart PKL reuse
---------------
Experiments that share the same language model, vocabulary, window, and graph
improvement flags also share the same graph PKL on disk.  The orchestrator
computes a compact 'graph_key' from those params and embeds it in
file_name_data, so experiments that differ only in architecture (gnn_type,
layers, heads, use_edge_attr …) can reuse the same expensive graph build.

For each experiment the script:
  1. Merges defaults + experiment overrides into a full config.
  2. Computes graph_key and derives file_name_data from it.
  3. Checks if the PKL exists; if so, forces build_graph=False.
  4. Writes a temporary config JSON and calls:
       python test_cooc_graph.py --config_path <tmp_config.json>
  5. Collects returncode and prints a final summary table.

Usage:
    python run_ablation_cooc.py --experiments_json experiments_ablation_cooc_gcn.json
    python run_ablation_cooc.py --exp gcn_base
    python run_ablation_cooc.py --group graph_ablation
    python run_ablation_cooc.py --dry_run
    python run_ablation_cooc.py --list
    python run_ablation_cooc.py --force_build
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
EXTERNAL_PATH = Path("/media/discoexterno/andric/data/experiments/cooc_graph")

# ---------------------------------------------------------------------------
# Graph-key helpers
# Only params that affect the saved PKL (graph topology + node features).
# GNN-only params (use_edge_attr, hidden_dim, num_layers, …) are excluded.
# ---------------------------------------------------------------------------
LM_ALIASES = {
    "microsoft/deberta-v3-base":                  "deberta",
    "FacebookAI/roberta-base":                    "roberta",
    "google-bert/bert-base-uncased":              "bert",
    "google-bert/bert-base-multilingual-uncased": "mbert",
    "intfloat/multilingual-e5-large":             "me5",
}

GRAPH_DEFAULTS = {
    "llm_name":         "microsoft/deberta-v3-base",
    "not_found_tokens": "avg",
    "window_size":      10,
    "min_df":           3,
    "max_df":           0.9,
    "max_features":     None,
    "stop_words":       False,
    "special_chars":    False,
    "use_edge_weights": True,
    "use_pmi":          False,
    "pmi_min_count":    5,
    "use_tfidf_feat":   True,
    "use_doc_node":     True,
    "use_self_loops":   True,
}


def compute_graph_key(cfg: dict) -> str:
    """
    Build a short human-readable key from graph-affecting params.
    Experiments with the same key share the same PKL.
    """
    parts = []

    lm = cfg.get("llm_name", GRAPH_DEFAULTS["llm_name"])
    parts.append(LM_ALIASES.get(lm, lm.split("/")[-1][:10]))

    nft = cfg.get("not_found_tokens", GRAPH_DEFAULTS["not_found_tokens"])
    if nft != GRAPH_DEFAULTS["not_found_tokens"]:
        parts.append(f"nft{nft}")

    w = cfg.get("window_size", GRAPH_DEFAULTS["window_size"])
    parts.append(f"w{w}")

    min_df = cfg.get("min_df", GRAPH_DEFAULTS["min_df"])
    if min_df != GRAPH_DEFAULTS["min_df"]:
        parts.append(f"df{min_df}")

    max_df = cfg.get("max_df", GRAPH_DEFAULTS["max_df"])
    if max_df != GRAPH_DEFAULTS["max_df"]:
        parts.append(f"mxdf{max_df}")

    mf = cfg.get("max_features", GRAPH_DEFAULTS["max_features"])
    if mf is not None:
        parts.append(f"mf{mf}")

    if cfg.get("stop_words", GRAPH_DEFAULTS["stop_words"]):
        parts.append("sw")
    if cfg.get("special_chars", GRAPH_DEFAULTS["special_chars"]):
        parts.append("sc")

    if not cfg.get("use_edge_weights", GRAPH_DEFAULTS["use_edge_weights"]):
        parts.append("noew")

    if cfg.get("use_pmi", GRAPH_DEFAULTS["use_pmi"]):
        pmi_cnt = cfg.get("pmi_min_count", GRAPH_DEFAULTS["pmi_min_count"])
        parts.append(f"pmi{pmi_cnt}")

    if not cfg.get("use_tfidf_feat", GRAPH_DEFAULTS["use_tfidf_feat"]):
        parts.append("notf")

    if not cfg.get("use_doc_node", GRAPH_DEFAULTS["use_doc_node"]):
        parts.append("nodn")

    if not cfg.get("use_self_loops", GRAPH_DEFAULTS["use_self_loops"]):
        parts.append("nosl")

    return "_".join(parts)


def file_name_for(cfg: dict) -> str:
    graph_key = compute_graph_key(cfg)
    return f"cooc_data_{cfg['dataset_name']}_{cfg['cut_off_dataset']}perc_{graph_key}"


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

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix=f"ablation_cooc_{name}_"
    ) as f:
        json.dump(cfg, f, indent=2)
        cfg_path = f.name

    cmd = [sys.executable, str(SRC_DIR / "test_cooc_graph.py"), "--config_path", cfg_path]

    print(f"\n{'='*70}")
    print(f"[RUN] {name}")
    print(f"  gnn_type        = {cfg.get('gnn_type')}")
    print(f"  num_layers      = {cfg.get('num_layers')}")
    print(f"  heads           = {cfg.get('heads')}")
    print(f"  hidden_dim      = {cfg.get('hidden_dim')}")
    print(f"  window_size     = {cfg.get('window_size')}")
    print(f"  use_edge_weights= {cfg.get('use_edge_weights')}")
    print(f"  use_pmi         = {cfg.get('use_pmi')}")
    print(f"  use_tfidf_feat  = {cfg.get('use_tfidf_feat')}")
    print(f"  use_doc_node    = {cfg.get('use_doc_node')}")
    print(f"  use_self_loops  = {cfg.get('use_self_loops')}")
    print(f"  use_edge_attr   = {cfg.get('use_edge_attr')}")
    print(f"  llm_name        = {cfg.get('llm_name','').split('/')[-1]}")
    print(f"  build_graph     = {cfg.get('build_graph')}")
    print(f"  file_name_data  = {cfg['file_name_data']}")
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
        description="Ablation study orchestrator for test_cooc_graph.py"
    )
    parser.add_argument(
        "--experiments_json", type=str,
        default=str(SRC_DIR / "experiments_ablation_cooc_gcn.json"),
        help="Path to the experiments JSON (default: experiments_ablation_cooc_gcn.json).",
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
            f"  {'Name':<35} {'Group':<18} {'GNN':<18} "
            f"{'L':>2} {'H':>2} {'hid':>4} {'w':>3} {'ew':>4} "
            f"{'pmi':>4} {'tf':>3} {'dn':>3} {'sl':>3}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for exp in experiments:
            grp = exp.get("_group", "—")
            print(
                f"  {exp['name']:<35} "
                f"{grp:<18} "
                f"{exp.get('gnn_type','?'):<18} "
                f"{exp.get('num_layers','?'):>2} "
                f"{exp.get('heads','?'):>2} "
                f"{exp.get('hidden_dim','?'):>4} "
                f"{exp.get('window_size','?'):>3} "
                f"{str(exp.get('use_edge_weights','?')):>4} "
                f"{str(exp.get('use_pmi','?')):>4} "
                f"{str(exp.get('use_tfidf_feat','?')):>3} "
                f"{str(exp.get('use_doc_node','?')):>3} "
                f"{str(exp.get('use_self_loops','?')):>3}"
            )
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
    print(f"  {'Experiment':<35} {'Group':<18} {'Status':^8}")
    print(f"  {'-'*35} {'-'*18} {'-'*8}")
    for s in summary:
        print(f"  {s['name']:<35} {s['group']:<18} {s['status']:^8}")
    print()

    failed = [s for s in summary if s["status"] != "ok"]
    if failed:
        print(f"[WARN] {len(failed)} experiment(s) failed: {[s['name'] for s in failed]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
