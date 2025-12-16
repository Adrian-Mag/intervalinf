"""Tests for configuration classes."""

import pytest
from intervalinf.core import IntegrationConfig, ParallelConfig


class TestIntegrationConfigInit:
    """Test IntegrationConfig initialization."""

    def test_default_values(self):
        """Test default configuration values."""
        config = IntegrationConfig()
        assert config.method == "simpson"
        assert config.n_points == 1000

    def test_custom_values(self):
        """Test custom configuration values."""
        config = IntegrationConfig(
            method="trapz",
            n_points=5000,
        )
        assert config.method == "trapz"
        assert config.n_points == 5000


class TestIntegrationConfigMethods:
    """Test IntegrationConfig methods."""

    def test_valid_methods(self):
        """Test all valid methods."""
        for method in ["simpson", "trapz", "quad"]:
            config = IntegrationConfig(method=method)  # type: ignore
            assert config.method == method

    def test_copy(self):
        """Test copy method."""
        config = IntegrationConfig(method="simpson", n_points=2000)
        config2 = config.copy()
        assert config.method == config2.method
        assert config.n_points == config2.n_points

    def test_copy_with_overrides(self):
        """Test copy with overrides."""
        config = IntegrationConfig(method="simpson", n_points=2000)
        config2 = config.copy(n_points=3000)
        assert config.n_points == 2000
        assert config2.n_points == 3000

    def test_copy_invalid_parameter(self):
        """Test copy with invalid parameter."""
        config = IntegrationConfig()
        with pytest.raises(ValueError, match="Unknown parameter"):
            config.copy(invalid_param=5)


class TestIntegrationConfigFactories:
    """Test IntegrationConfig factory methods."""

    def test_high_accuracy(self):
        """Test high_accuracy factory."""
        config = IntegrationConfig.high_accuracy()
        assert config.n_points == 10000
        assert config.method == "simpson"

    def test_fast(self):
        """Test fast factory."""
        config = IntegrationConfig.fast()
        assert config.n_points == 500
        assert config.method == "trapz"

    def test_adaptive(self):
        """Test adaptive factory with dimension."""
        config = IntegrationConfig.adaptive(dim=20)
        assert config.n_points == 2000  # max(1000, 100*20)

        config2 = IntegrationConfig.adaptive(dim=5)
        assert config2.n_points == 1000  # max(1000, 100*5)


class TestIntegrationConfigEquality:
    """Test IntegrationConfig equality."""

    def test_equal_configs(self):
        """Test equal configs."""
        c1 = IntegrationConfig(method="simpson", n_points=1000)
        c2 = IntegrationConfig(method="simpson", n_points=1000)
        assert c1 == c2

    def test_unequal_method(self):
        """Test unequal method."""
        c1 = IntegrationConfig(method="simpson")
        c2 = IntegrationConfig(method="trapz")
        assert c1 != c2

    def test_unequal_n_points(self):
        """Test unequal n_points."""
        c1 = IntegrationConfig(n_points=1000)
        c2 = IntegrationConfig(n_points=2000)
        assert c1 != c2


class TestParallelConfigInit:
    """Test ParallelConfig initialization."""

    def test_default_values(self):
        """Test default values."""
        config = ParallelConfig()
        assert config.enabled is False
        assert config.n_jobs == -1

    def test_enabled(self):
        """Test enabled config."""
        config = ParallelConfig(enabled=True, n_jobs=4)
        assert config.enabled is True
        assert config.n_jobs == 4

    def test_disabled(self):
        """Test disabled config."""
        config = ParallelConfig(enabled=False)
        assert config.enabled is False


class TestParallelConfigMethods:
    """Test ParallelConfig methods."""

    def test_copy(self):
        """Test copy method."""
        config = ParallelConfig(enabled=True, n_jobs=4)
        config2 = config.copy()
        assert config.enabled == config2.enabled
        assert config.n_jobs == config2.n_jobs

    def test_copy_with_overrides(self):
        """Test copy with overrides."""
        config = ParallelConfig(enabled=True, n_jobs=4)
        config2 = config.copy(n_jobs=8)
        assert config.n_jobs == 4
        assert config2.n_jobs == 8


class TestParallelConfigFactories:
    """Test ParallelConfig factory methods."""

    def test_all_cores(self):
        """Test all_cores factory."""
        config = ParallelConfig.all_cores()
        assert config.enabled is True
        assert config.n_jobs == -1

    def test_serial(self):
        """Test serial factory."""
        config = ParallelConfig.serial()
        assert config.enabled is False
        assert config.n_jobs == 1

    def test_cores(self):
        """Test cores factory with specific count."""
        config = ParallelConfig.cores(8)
        assert config.enabled is True
        assert config.n_jobs == 8
