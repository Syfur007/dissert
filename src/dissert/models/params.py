"""
utils/metrics.py — model-profiling helpers.

Segmentation-quality metrics (Dice/IoU/HD95/ASD/...) moved to the
top-level metrics/ package in Phase 1 of IMPLEMENTATION_PLAN.md — see
metrics/aggregate.py's compute_dataset_metrics(), the direct replacement for
what used to live here as get_binary_metrics()/compute_dataset_metrics().

measure_throughput and log_model_summary (Phase 10's named targets —
warmup=5 loop, no stated batch size, thop/ptflops-only complexity figure)
moved to the profiling/ package: profiling.latency.measure_latency (spec
§14's >=50-warmup/>=200-run/stated-batch-size protocol) and
profiling.flops.check_flops_agreement (analytic + fvcore agreement-checked
FLOPs, not a single unverified tool's number) respectively.
"""
def count_parameters(model):
    """Count the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model):
    """Count *every* parameter (trainable + frozen).

    Phase 2 (docs/design/transfer-xai-plan.md §6.4): a frozen pretrained
    encoder can be hundreds of millions of parameters while
    count_parameters() reports near-zero trainable ones — a budget_ceiling
    checked only against the trainable count would wave that model through.
    ModelRegistry.get() reports both counts; budget_ceiling applies to this
    one unless budget_on="trainable" is set explicitly.
    """
    return sum(p.numel() for p in model.parameters())
