# TEMPORARY ROOT SHIM — Phase 0 repo reorg (src/ layout, dissert package).
# eval.py now lives at src/dissert/cli/eval.py. This shim exists only so
# that external tooling invoking `python eval.py ...` at the repo root
# (XDash's repos/dissert.yaml eval_script, notebooks/kaggle_run.ipynb)
# keeps working without a coordinated update on both sides.
#
# DELETE THIS FILE once XDash's repos/dissert.yaml eval_script points at
# `src/dissert/cli/eval.py` (or is switched to `-m dissert.cli.eval`).
from dissert.cli.eval import main

if __name__ == "__main__":
    main()
