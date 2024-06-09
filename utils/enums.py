"""PyNQS enum types."""

from __future__ import annotations

from enum import Enum


class ElocMethod(Enum):
    """eloc method.

    SIMPLE: exact calculate local energy
    REDUCE_PSI: ignore x' when |<x|H|x'>| <= eps or sampling from p(m) \propto |Hnm|
    SAMPLE_SPACE: use unique sample as x' not SD
    """

    SIMPLE = 1
    REDUCE = 2
    SAMPLE_SPACE = 3
