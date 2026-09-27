#!/usr/bin/env bash
# scripts/reproduce.sh — Phase 14 (IMPLEMENTATION_PLAN.md): orchestrates
# the S1-S17 pipeline end to end, in reduced-scope form by default
# (SEEDS/MODEL_CONFIG/EPOCHS env vars below) so this script is genuinely
# runnable as a smoke test, not just documentation.
#
# Honest about what each stage actually is today:
#   - S1, S6, S15, S17 have real, working commands below — this script
#     executes them for real.
#   - S3/S4/S5 reuse S6's same dissert.cli.train/dissert.orchestration.sweep
#     machinery (a "sanity"/"baseline"/"LR-sweep" run *is* a dissert.cli.train
#     run with different config knobs — there is no separate binary for them).
#   - S2, S7-S14, S16 have no standalone CLI script (IMPLEMENTATION_PLAN.md
#     Phases 4/9/11/12/13 built them as importable Python modules, meant
#     to be driven from a notebook/analysis script once real multi-seed
#     checkpoints exist to analyse) — this script prints the exact
#     module/function to call for each rather than faking a command that
#     doesn't exist.
#
# Usage:
#   ./scripts/reproduce.sh
#   SEEDS="42 43 44" MODEL_CONFIG=configs/experiment/gmkunet/gmkunet_t_clinicdb.yaml ./scripts/reproduce.sh
#
# Reduced-scope defaults below trade off against IMPLEMENTATION_PLAN.md's
# own end-to-end verification instruction ("1 seed, 1 fold, 2 datasets"):
# SEEDS defaults to 3, not 1 — S17's own minimum-seeds blocking rule
# (spec §18: 3 minimum) would otherwise correctly refuse to emit S6's
# table, so a literal 1-seed run cannot finish this script successfully.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# The dissert package is installed (editable) via `pip install -e .[dev]`,
# so every dissert.* import below resolves through the installed package —
# no PYTHONPATH export needed (Phase 0 repo reorg: src/ layout).

SEEDS="${SEEDS:-42 43 44}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/experiment/gmkunet/gmkunet_t_clinicdb.yaml}"
EPOCHS="${EPOCHS:-1}"
LEDGER_DIR="${LEDGER_DIR:-outputs/ledger}"
REPORTS_DIR="${REPORTS_DIR:-outputs/reports}"
SNAPSHOT_ID="${SNAPSHOT_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

echo "=============================================================="
echo "S1 Data build — manifest/dedup/split-load/leakage assertions"
echo "=============================================================="
pytest tests/test_data_contract.py -q

echo ""
echo "=============================================================="
echo "S2 Channel build — covered by S1's channel/geometry assertions"
echo "   (tests/test_channels.py; no separate CLI stage — channel"
echo "   construction happens inline in the dataset pipeline)"
echo "=============================================================="
pytest tests/test_channels.py -q

# Every run below uses a *scoped* experiment name
# (<original>_reproduce_<snapshot>), never the config's own bare
# experiment_name — confirmed the hard way while building this script:
# reusing the bare name overwrote an existing real run's
# outputs/experiments/<name>/<hash7>-s<seed>/eval/report.json with the
# smoke run's own. Seed no longer needs manually folding into the name —
# it's already part of the on-disk experiment_id ("{config_hash[:7]}-s
# {seed}", nested under experiment_name — see dissert/orchestration/runid.py), so
# each seed below lands in its own directory automatically.
BASE_EXP_NAME="$(python3 -c "from dissert.config.loader import load_config; print(load_config('$MODEL_CONFIG')['logging']['experiment_name'])")"
REPRODUCE_TAG="${BASE_EXP_NAME}_reproduce_${SNAPSHOT_ID}"

echo ""
echo "=============================================================="
echo "S3/S4/S6 Sanity training / baseline repro / main comparison"
echo "   — reduced to ${EPOCHS} epoch(s), seed(s): ${SEEDS}, on"
echo "   ${MODEL_CONFIG}, via dissert.orchestration.runner.run_sweep"
echo "   scoped experiment name: ${REPRODUCE_TAG} (outputs/experiments/${REPRODUCE_TAG}/<hash7>-s<seed>/)"
echo "=============================================================="
python3 - "$MODEL_CONFIG" "$EPOCHS" "$LEDGER_DIR" "$SEEDS" "$REPRODUCE_TAG" <<'PYEOF'
import sys
from dissert.config.loader import load_config
from dissert.orchestration.runner import run_sweep

model_config, epochs, ledger_dir, seeds_str, tag = sys.argv[1:6]
seeds = [int(s) for s in seeds_str.split()]

for seed in seeds:
    config = load_config(model_config)
    config["training"]["epochs"] = int(epochs)
    config["k_fold"]["enabled"] = False
    config["checkpoint"]["resume"] = False
    config["logging"]["experiment_name"] = tag

    results = run_sweep(config, seeds=[seed], ledger_dir=ledger_dir)
    for r in results:
        print(f"  run_id={r['run_id']} status={r['status']} best_metric={r['best_metric']}")
    failed = [r for r in results if r["status"] not in ("done", "skipped-done")]
    if failed:
        sys.exit(f"S6 sweep had {len(failed)} non-done run(s) for seed {seed}: {failed}")
PYEOF

echo ""
echo "=============================================================="
echo "S5 LR sweep — dissert/orchestration/sweep.py (needs search.budget_gpu_hours"
echo "   in configs/search_config.yaml; not run automatically here since"
echo "   it trains a full grid of trials — see dissert.orchestration.sweep's"
echo "   own CLI: python -m dissert.orchestration.sweep --base-config ... "
echo "   --search-config configs/search_config.yaml)"
echo "=============================================================="

echo ""
echo "=============================================================="
echo "S7 Statistics — dissert.analysis.stats.run_family_comparison(...) on per-image"
echo "   Parquet from dissert.metrics.aggregate.write_per_image_parquet(...);"
echo "   needs >=2 trained models' predictions to compare, so it is not"
echo "   invoked by this single-model smoke run — see dissert/analysis/stats/__init__.py"
echo "=============================================================="

echo ""
echo "=============================================================="
echo "S8 Ablation / S9 Channel study — additional dissert.cli.train runs over"
echo "   ablation/channel-mode config variants, aggregated the same way"
echo "   as S6 above (no separate stage; a matter of which configs are"
echo "   passed to dissert.orchestration.runner.run_sweep)"
echo "=============================================================="

echo ""
echo "=============================================================="
echo "S10 Attribution / S11 Shortcut audit / S12 Mechanism — Python"
echo "   APIs, run against a trained checkpoint + the guarded test"
echo "   loader's one-time token: dissert.xai.{occlusion,shapley,"
echo "   integrated_grads,branch,fusion_probe,segcam,sanity},"
echo "   dissert.analysis.robustness.geometric.shortcut_audit,"
echo "   dissert.analysis.mechanism.{erf,cka,"
echo "   failure_taxonomy} — see each module's own docstring"
echo "=============================================================="

echo ""
echo "=============================================================="
echo "S13 Uncertainty / S14 Robustness — dissert.analysis.uncertainty.{ensemble,retention},"
echo "   dissert.analysis.robustness.{corruptions,common,geometric} — Python APIs, run"
echo "   against the seed ensemble S6 above just produced"
echo "=============================================================="

echo ""
echo "=============================================================="
echo "S15 External — guarded one-time test evaluation via dissert.cli.eval,"
echo "   once per seed's checkpoint (no --fold: S6 above ran with"
echo "   k_fold disabled, so each seed's checkpoint is"
echo "   outputs/experiments/${REPRODUCE_TAG}/<hash7>-s<seed>/checkpoints/best.pth)"
echo "=============================================================="
for seed in $SEEDS; do
    python3 -m dissert.cli.eval --config "$MODEL_CONFIG" --experiment-name "$REPRODUCE_TAG" --seed "$seed" --allow-test-eval
done

echo ""
echo "=============================================================="
echo "S16 Profiling — dissert.analysis.profiling.{flops,latency,memory,export}; already"
echo "   folded into dissert.cli.eval's own report (model.flops/efficiency.* in"
echo "   the JSON dump S17 reads below) via check_flops_agreement/"
echo "   measure_latency (see dissert.cli.eval's own imports)"
echo "=============================================================="

echo ""
echo "=============================================================="
echo "S17 Reporting — render manuscript tables from dissert.cli.eval's JSON dumps"
echo "=============================================================="
python3 -m dissert.cli.report \
    --reports-glob "outputs/experiments/${REPRODUCE_TAG}/*-s*/eval/report.json" \
    --ledger-dir "$LEDGER_DIR" \
    --out-dir "$REPORTS_DIR" \
    --snapshot-id "$SNAPSHOT_ID"

echo ""
echo "Done. Tables written to ${REPORTS_DIR}/."
