from loguru import logger


class EarlyStopping:
    """
    Early stopping utility to halt training when a monitored validation metric
    stops improving. This prevents overfitting and saves compute for trials
    that have already peaked.

    Supports both maximization ('max') and minimization ('min') metric modes,
    matching the same convention as CheckpointManager, so both components can
    track the same metric with the same direction.

    Args:
        patience    (int):   Number of epochs to wait after the last improvement
                             before triggering a stop. Default: 20.
        min_delta   (float): Minimum change in the monitored metric to be
                             considered an improvement. Acts as a threshold to
                             ignore negligibly small improvements. Default: 0.0.
        mode        (str):   'max' (higher is better, e.g. Dice/IoU) or 'min'
                             (lower is better, e.g. loss/HD95). Default: 'max'.
        verbose     (bool):  Log patience countdown messages. Default: True.

    Usage::
        es = EarlyStopping(patience=20, mode='max')
        # In your training loop:
        if es(val_dice):
            break
    """

    def __init__(self, patience: int = 20, min_delta: float = 0.0,
                 mode: str = "max", verbose: bool = True):
        if mode not in ("max", "min"):
            raise ValueError(f"EarlyStopping mode must be 'max' or 'min', got '{mode}'.")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.best_metric: float = float('-inf') if mode == "max" else float('inf')
        self.counter: int = 0
        self.triggered: bool = False

    # ------------------------------------------------------------------
    # Restore state when resuming from a checkpoint so that patience
    # counts are not reset to zero on resume (which would effectively
    # ignore all pre-resume epochs of no improvement).
    # ------------------------------------------------------------------

    def restore(self, best_metric: float, counter: int = 0) -> None:
        """Restore state from a resumed checkpoint.

        Args:
            best_metric: The best monitored value seen before the resume point.
            counter:     Epochs of no improvement already accumulated before
                         the resume point. Defaults to 0 (unknown, conservative).
        """
        self.best_metric = best_metric
        self.counter = counter
        if self.verbose:
            logger.info(
                f"EarlyStopping restored | best={best_metric:.4f} | "
                f"patience_counter={counter}/{self.patience}"
            )

    def _is_improvement(self, current: float) -> bool:
        """Return True if *current* improves on *best_metric* by at least *min_delta*."""
        if self.mode == "max":
            return current > self.best_metric + self.min_delta
        else:
            return current < self.best_metric - self.min_delta

    def __call__(self, current_metric: float) -> bool:
        """Check metric and return True if training should stop.

        Args:
            current_metric: The monitored metric value for the current epoch.

        Returns:
            True  — training should stop (patience exceeded).
            False — training should continue.
        """
        if self._is_improvement(current_metric):
            self.best_metric = current_metric
            self.counter = 0
            if self.verbose:
                logger.debug(
                    f"EarlyStopping | improvement detected → best={self.best_metric:.4f}"
                )
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"EarlyStopping | no improvement for {self.counter}/{self.patience} epoch(s) "
                    f"(best={self.best_metric:.4f}, current={current_metric:.4f})"
                )

        if self.counter >= self.patience:
            if self.verbose:
                logger.info(
                    f"EarlyStopping triggered after {self.patience} epochs without improvement. "
                    f"Best {self.mode} metric: {self.best_metric:.4f}"
                )
            self.triggered = True
            return True

        return False

    def state_dict(self) -> dict:
        """Return serialisable state for checkpoint persistence."""
        return {
            "best_metric": self.best_metric,
            "counter": self.counter,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
            "triggered": self.triggered,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore from a previously saved state_dict."""
        self.best_metric = state["best_metric"]
        self.counter = state["counter"]
        self.patience = state.get("patience", self.patience)
        self.min_delta = state.get("min_delta", self.min_delta)
        self.mode = state.get("mode", self.mode)
        self.triggered = state.get("triggered", False)
