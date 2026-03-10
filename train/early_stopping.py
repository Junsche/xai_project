# -*- coding: utf-8 -*-
"""
Early stopping helper for training loops.
"""

from __future__ import annotations


class EarlyStopping:
    """Minimal early stopping helper."""

    def __init__(self, mode: str = "max", patience: int = 5, min_delta: float = 0.0):
        if mode not in ["max", "min"]:
            raise ValueError(f"Unsupported early stopping mode: {mode}")

        self.mode = mode
        self.patience = int(patience)
        self.min_delta = float(min_delta)

        self.best = None
        self.bad = 0
        self.should_stop = False

    def step(self, cur: float) -> bool:
        """
        Update early stopping state with the current monitored metric.

        Returns:
            bool: True if training should stop, otherwise False.
        """
        if self.best is None:
            self.best = cur
            return False

        improved = (
            cur > self.best + self.min_delta
            if self.mode == "max"
            else cur < self.best - self.min_delta
        )

        if improved:
            self.best = cur
            self.bad = 0
        else:
            self.bad += 1
            if self.bad >= self.patience:
                self.should_stop = True

        return self.should_stop