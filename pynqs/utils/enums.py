"""PyNQS enum types."""

from __future__ import annotations

from enum import Enum


class ElocMethod(Enum):
    r"""eloc method.

    SIMPLE: exact calculate local energy
    REDUCE_PSI: ignore x' when |<x|H|x'>| <= eps or sampling from p(m) \propto |Hnm|
    SAMPLE_SPACE: use unique sample as x' not SD
    """

    SIMPLE = 1
    REDUCE = 2
    SAMPLE_SPACE = 3


class SampleMethod(Enum):
    r"""Sampling methods for MCMC simulation.

    EXACT: exact optimizations
    AR: Autoregressive sampling
    MCMC: Standard Markov Chain Monte Carlo
    RESTRICTED: Restricted sampling in specific space
    PTMC: Parallel Tempering Monte Carlo
    HybridMC: Hybrid Monte Carlo using MPS propose
    """

    AR = 1
    MCMC = 2
    RESTRICTED = 3
    PTMC = 4
    HYBRIDMC = 5
    EXACT = 6
