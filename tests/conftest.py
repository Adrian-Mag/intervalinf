"""Shared test fixtures for intervalinf."""

import pytest
import numpy as np


@pytest.fixture
def unit_domain():
    """Standard [0, 1] domain."""
    from intervalinf.core import IntervalDomain
    return IntervalDomain(0, 1)


@pytest.fixture
def pi_domain():
    """[0, π] domain for trigonometric functions."""
    from intervalinf.core import IntervalDomain
    return IntervalDomain(0, np.pi)


@pytest.fixture
def simple_space(unit_domain):
    """A minimal space-like object for testing Function without full Lebesgue."""
    class SimpleSpace:
        def __init__(self, domain):
            self._function_domain = domain
            self.basis_functions = None
    return SimpleSpace(unit_domain)
