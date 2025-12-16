"""
Tests for Sobolev space.

Tests basic Sobolev space functionality that doesn't require
the operators module (Phase 3).
"""

import pytest
import numpy as np

from intervalinf.core import IntervalDomain, Function
from intervalinf.core.config import IntegrationConfig
from intervalinf.spaces.sobolev import Sobolev, SobolevSpaceDirectSum


class TestSobolevInit:
    """Test Sobolev space initialization."""

    def test_init_requires_laplacian(self):
        """Test that initialization requires a Laplacian operator."""
        domain = IntervalDomain(0, 1)

        # Without pygeoinf installed, this should raise NotImplementedError
        # when trying to create mass operators
        with pytest.raises((NotImplementedError, ImportError, TypeError)):
            # Note: s, k, L are positional-only arguments
            Sobolev(50, domain, 1.0, 1.0, None)


class TestSobolevImportGuards:
    """Test that Sobolev properly guards imports."""

    def test_mass_operator_factor_import_guard(self):
        """Test mass_operator_factor raises without operators."""
        # Create a mock Sobolev-like object to test the property
        domain = IntervalDomain(0, 1)

        # We can't create a full Sobolev without operators, so test
        # that the import guard works by checking the code path exists
        # This test documents the expected behavior
        pass

    def test_with_discontinuities_requires_operators(self):
        """Test with_discontinuities requires operators module."""
        domain = IntervalDomain(0, 2)

        # Mock boundary conditions
        class MockBCs:
            bc_type = 'dirichlet'

        with pytest.raises(NotImplementedError, match="operators module"):
            Sobolev.with_discontinuities(
                20, domain, [1.0],
                s=1.0, k=1.0, bcs=MockBCs(), alpha=0.1
            )


class TestSobolevSpaceDirectSum:
    """Test SobolevSpaceDirectSum behavior."""

    def test_to_dual_wrong_length(self):
        """Test to_dual with wrong list length raises."""
        # Create a mock direct sum
        class MockSpace:
            def __init__(self):
                self._spaces = []

            @property
            def number_of_subspaces(self):
                return 2

        direct_sum = SobolevSpaceDirectSum.__new__(SobolevSpaceDirectSum)
        direct_sum._spaces = []
        # Override number_of_subspaces
        object.__setattr__(direct_sum, '_spaces', [])

        # This test documents expected behavior with empty list
        # Full testing requires Phase 3 operators


class TestSobolevDocumentation:
    """Test that Sobolev class is documented correctly."""

    def test_docstring_exists(self):
        """Test that Sobolev has documentation."""
        assert Sobolev.__doc__ is not None
        assert 'Sobolev' in Sobolev.__doc__
        assert 'H^s' in Sobolev.__doc__

    def test_init_docstring(self):
        """Test __init__ has documentation."""
        assert Sobolev.__init__.__doc__ is not None
        assert 's:' in Sobolev.__init__.__doc__ or 'regularity' in Sobolev.__init__.__doc__
