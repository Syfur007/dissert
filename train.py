# TEMPORARY ROOT SHIM — Phase 0 repo reorg (src/ layout, dissert package).
# train.py now lives at src/dissert/cli/train.py. This shim exists only so
# that external tooling invoking `python train.py ...` at the repo root
# (XDash's repos/dissert.yaml train_script, notebooks/kaggle_run.ipynb)
# keeps working without a coordinated update on both sides.
#
# DELETE THIS FILE once XDash's repos/dissert.yaml train_script points at
# `src/dissert/cli/train.py` (or is switched to `-m dissert.cli.train`).
from dissert.cli.train import main

if __name__ == "__main__":
    main()
