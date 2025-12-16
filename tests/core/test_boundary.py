"""Tests for BoundaryConditions."""

import pytest
from intervalinf.core import BoundaryConditions


class TestBoundaryConditionsInit:
    """Test BoundaryConditions initialization."""

    def test_dirichlet_defaults(self):
        """Test Dirichlet with default values."""
        bc = BoundaryConditions("dirichlet")
        assert bc.type == "dirichlet"
        assert bc.get_parameter("left") == 0.0
        assert bc.get_parameter("right") == 0.0

    def test_dirichlet_custom(self):
        """Test Dirichlet with custom values."""
        bc = BoundaryConditions("dirichlet", left=1.0, right=2.0)
        assert bc.get_parameter("left") == 1.0
        assert bc.get_parameter("right") == 2.0

    def test_neumann_defaults(self):
        """Test Neumann with default values."""
        bc = BoundaryConditions("neumann")
        assert bc.type == "neumann"
        assert bc.get_parameter("left") == 0.0
        assert bc.get_parameter("right") == 0.0

    def test_periodic(self):
        """Test periodic boundary conditions."""
        bc = BoundaryConditions("periodic")
        assert bc.type == "periodic"

    def test_robin_requires_all_params(self):
        """Test Robin requires all parameters."""
        with pytest.raises(ValueError, match="require"):
            BoundaryConditions("robin", left_alpha=1.0)

    def test_robin_valid(self):
        """Test valid Robin boundary conditions."""
        bc = BoundaryConditions(
            "robin",
            left_alpha=1.0, left_beta=0.5, left_value=0.0,
            right_alpha=1.0, right_beta=0.5, right_value=0.0
        )
        assert bc.type == "robin"
        assert bc.get_parameter("left_alpha") == 1.0

    def test_invalid_type(self):
        """Test invalid boundary condition type."""
        with pytest.raises(ValueError, match="Invalid boundary condition"):
            BoundaryConditions("invalid")


class TestBoundaryConditionsFactories:
    """Test factory methods."""

    def test_dirichlet_factory(self):
        """Test dirichlet factory method."""
        bc = BoundaryConditions.dirichlet(1.0, 2.0)
        assert bc.type == "dirichlet"
        assert bc.get_parameter("left") == 1.0
        assert bc.get_parameter("right") == 2.0

    def test_neumann_factory(self):
        """Test neumann factory method."""
        bc = BoundaryConditions.neumann(0.5, -0.5)
        assert bc.type == "neumann"
        assert bc.get_parameter("left") == 0.5
        assert bc.get_parameter("right") == -0.5

    def test_periodic_factory(self):
        """Test periodic factory method."""
        bc = BoundaryConditions.periodic()
        assert bc.type == "periodic"

    def test_robin_factory(self):
        """Test robin factory method."""
        bc = BoundaryConditions.robin(1, 2, 3, 4, 5, 6)
        assert bc.get_parameter("left_alpha") == 1
        assert bc.get_parameter("right_value") == 6

    def test_mixed_dn_factory(self):
        """Test mixed Dirichlet-Neumann factory."""
        bc = BoundaryConditions.mixed_dirichlet_neumann(1.0, 2.0)
        assert bc.type == "mixed_dirichlet_neumann"
        assert bc.get_parameter("left") == 1.0
        assert bc.get_parameter("right") == 2.0

    def test_mixed_nd_factory(self):
        """Test mixed Neumann-Dirichlet factory."""
        bc = BoundaryConditions.mixed_neumann_dirichlet(1.0, 2.0)
        assert bc.type == "mixed_neumann_dirichlet"


class TestBoundaryConditionsHomogeneous:
    """Test is_homogeneous property."""

    def test_homogeneous_dirichlet(self):
        """Test homogeneous Dirichlet."""
        bc = BoundaryConditions.dirichlet(0, 0)
        assert bc.is_homogeneous

    def test_inhomogeneous_dirichlet(self):
        """Test inhomogeneous Dirichlet."""
        bc = BoundaryConditions.dirichlet(1, 0)
        assert not bc.is_homogeneous

    def test_homogeneous_neumann(self):
        """Test homogeneous Neumann."""
        bc = BoundaryConditions.neumann(0, 0)
        assert bc.is_homogeneous

    def test_periodic_is_homogeneous(self):
        """Test periodic is considered homogeneous."""
        bc = BoundaryConditions.periodic()
        assert bc.is_homogeneous


class TestBoundaryConditionsEquality:
    """Test equality comparison."""

    def test_equal_dirichlet(self):
        """Test equal Dirichlet conditions."""
        bc1 = BoundaryConditions.dirichlet(1, 2)
        bc2 = BoundaryConditions.dirichlet(1, 2)
        assert bc1 == bc2

    def test_unequal_values(self):
        """Test unequal values."""
        bc1 = BoundaryConditions.dirichlet(1, 2)
        bc2 = BoundaryConditions.dirichlet(1, 3)
        assert bc1 != bc2

    def test_unequal_types(self):
        """Test unequal types."""
        bc1 = BoundaryConditions.dirichlet()
        bc2 = BoundaryConditions.neumann()
        assert bc1 != bc2

    def test_not_equal_to_other_types(self):
        """Test not equal to non-BoundaryConditions."""
        bc = BoundaryConditions.dirichlet()
        assert bc != "dirichlet"
        assert bc != 42


class TestBoundaryConditionsStr:
    """Test string representations."""

    def test_str_periodic(self):
        """Test string for periodic."""
        bc = BoundaryConditions.periodic()
        assert str(bc) == "periodic"

    def test_str_dirichlet(self):
        """Test string for Dirichlet."""
        bc = BoundaryConditions.dirichlet(1, 2)
        s = str(bc)
        assert "dirichlet" in s
        assert "left=1" in s

    def test_repr(self):
        """Test repr."""
        bc = BoundaryConditions.dirichlet(1, 2)
        r = repr(bc)
        assert "BoundaryConditions" in r
        assert "dirichlet" in r
