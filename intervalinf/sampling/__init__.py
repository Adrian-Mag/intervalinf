"""Sampling module - Stochastic sampling methods for Gaussian measures.

This module provides tools for sampling from Gaussian measures using
spectral (Karhunen-Loève) expansions.

Classes
-------
KLSampler
    Sampler for Gaussian measures using truncated KL expansion.
TruncationInfo
    Information about truncation of the KL expansion.
"""

from intervalinf.sampling.kl_sampler import KLSampler, TruncationInfo

__all__ = ["KLSampler", "TruncationInfo"]
