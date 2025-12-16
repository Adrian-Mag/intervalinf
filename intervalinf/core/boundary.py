"""
Boundary condition specifications for function spaces on intervals.

This module provides boundary condition classes and utilities for L² spaces,
Sobolev spaces, and FEM solvers on interval domains.
"""


class BoundaryConditions:
    """
    Boundary condition specifications for function spaces on intervals.

    This class provides a unified interface for all boundary condition types
    used across L² spaces, Sobolev spaces, and FEM solvers.

    Parameters
    ----------
    bc_type : str
        Type of boundary condition. One of:
        - 'dirichlet': u(a) = left, u(b) = right
        - 'neumann': u'(a) = left, u'(b) = right
        - 'robin': αu + βu' = value at boundaries
        - 'periodic': u(a) = u(b), u'(a) = u'(b)
        - 'mixed_dirichlet_neumann': u(a) = left, u'(b) = right
        - 'mixed_neumann_dirichlet': u'(a) = left, u(b) = right
    **kwargs
        Parameters for the specific boundary condition type.

    Examples
    --------
    >>> bc = BoundaryConditions.dirichlet()  # Homogeneous Dirichlet
    >>> bc = BoundaryConditions.neumann(left_derivative=1.0)
    >>> bc = BoundaryConditions.periodic()
    """

    def __init__(self, bc_type: str, **kwargs):
        self.type = bc_type
        self._params = kwargs
        self._validate()

    def _validate(self):
        """Validate boundary condition parameters."""
        valid_types = {
            "dirichlet",
            "neumann",
            "robin",
            "periodic",
            "mixed_dirichlet_neumann",
            "mixed_neumann_dirichlet",
        }

        if self.type not in valid_types:
            raise ValueError(
                f"Invalid boundary condition type '{self.type}'. "
                f"Valid types: {sorted(valid_types)}"
            )

        # Type-specific validation
        if self.type == "dirichlet":
            self._params.setdefault("left", 0.0)
            self._params.setdefault("right", 0.0)

        elif self.type == "neumann":
            self._params.setdefault("left", 0.0)
            self._params.setdefault("right", 0.0)

        elif self.type == "robin":
            required = [
                "left_alpha",
                "left_beta",
                "left_value",
                "right_alpha",
                "right_beta",
                "right_value",
            ]
            for param in required:
                if param not in self._params:
                    raise ValueError(
                        f"Robin boundary conditions require '{param}'"
                    )

        elif self.type == "periodic":
            pass  # No additional parameters needed

        elif self.type == "mixed_dirichlet_neumann":
            self._params.setdefault("left", 0.0)  # Dirichlet at left
            self._params.setdefault("right", 0.0)  # Neumann at right

        elif self.type == "mixed_neumann_dirichlet":
            self._params.setdefault("left", 0.0)  # Neumann at left
            self._params.setdefault("right", 0.0)  # Dirichlet at right

    @property
    def is_homogeneous(self) -> bool:
        """Check if boundary conditions are homogeneous."""
        if self.type == "dirichlet":
            return (
                self._params.get("left", 0) == 0
                and self._params.get("right", 0) == 0
            )
        elif self.type == "neumann":
            return (
                self._params.get("left", 0) == 0
                and self._params.get("right", 0) == 0
            )
        elif self.type == "periodic":
            return True  # Periodic BCs are considered homogeneous
        elif self.type == "mixed_dirichlet_neumann":
            return (
                self._params.get("left", 0) == 0
                and self._params.get("right", 0) == 0
            )
        elif self.type == "mixed_neumann_dirichlet":
            return (
                self._params.get("left", 0) == 0
                and self._params.get("right", 0) == 0
            )
        elif self.type == "robin":
            return (
                self._params.get("left_value", 0) == 0
                and self._params.get("right_value", 0) == 0
            )
        else:
            return False

    def get_parameter(self, name: str, default=None):
        """Get a boundary condition parameter."""
        return self._params.get(name, default)

    # Factory methods
    @classmethod
    def dirichlet(
        cls, left_value: float = 0, right_value: float = 0
    ) -> "BoundaryConditions":
        """
        Create Dirichlet boundary conditions:
        u(a) = left_value, u(b) = right_value.

        Parameters
        ----------
        left_value : float, optional
            Value at left boundary (default 0).
        right_value : float, optional
            Value at right boundary (default 0).
        """
        return cls("dirichlet", left=left_value, right=right_value)

    @classmethod
    def neumann(
        cls, left_derivative: float = 0, right_derivative: float = 0
    ) -> "BoundaryConditions":
        """
        Create Neumann boundary conditions: u'(a) = left, u'(b) = right.

        Parameters
        ----------
        left_derivative : float, optional
            Derivative at left boundary (default 0).
        right_derivative : float, optional
            Derivative at right boundary (default 0).
        """
        return cls("neumann", left=left_derivative, right=right_derivative)

    @classmethod
    def robin(
        cls,
        left_alpha: float,
        left_beta: float,
        left_value: float,
        right_alpha: float,
        right_beta: float,
        right_value: float,
    ) -> "BoundaryConditions":
        """
        Create Robin boundary conditions: αu + βu' = value at boundaries.

        Parameters
        ----------
        left_alpha, left_beta, left_value : float
            Left boundary coefficients and value.
        right_alpha, right_beta, right_value : float
            Right boundary coefficients and value.
        """
        return cls(
            "robin",
            left_alpha=left_alpha,
            left_beta=left_beta,
            left_value=left_value,
            right_alpha=right_alpha,
            right_beta=right_beta,
            right_value=right_value,
        )

    @classmethod
    def periodic(cls) -> "BoundaryConditions":
        """Create periodic boundary conditions: u(a) = u(b), u'(a) = u'(b)."""
        return cls("periodic")

    @classmethod
    def mixed_dirichlet_neumann(
        cls, left_value: float = 0, right_derivative: float = 0
    ) -> "BoundaryConditions":
        """
        Create mixed Dirichlet-Neumann BCs: u(a) = left, u'(b) = right.

        Parameters
        ----------
        left_value : float, optional
            Value at left boundary (default 0).
        right_derivative : float, optional
            Derivative at right boundary (default 0).
        """
        return cls(
            "mixed_dirichlet_neumann",
            left=left_value,
            right=right_derivative,
        )

    @classmethod
    def mixed_neumann_dirichlet(
        cls, left_derivative: float = 0, right_value: float = 0
    ) -> "BoundaryConditions":
        """
        Create mixed Neumann-Dirichlet BCs: u'(a) = left, u(b) = right.

        Parameters
        ----------
        left_derivative : float, optional
            Derivative at left boundary (default 0).
        right_value : float, optional
            Value at right boundary (default 0).
        """
        return cls(
            "mixed_neumann_dirichlet",
            left=left_derivative,
            right=right_value,
        )

    def __str__(self) -> str:
        if self.type == "periodic":
            return "periodic"
        else:
            params_str = ", ".join(f"{k}={v}" for k, v in self._params.items())
            return f"{self.type}({params_str})"

    def __repr__(self) -> str:
        return f"BoundaryConditions('{self.type}', {self._params})"

    def __eq__(self, other) -> bool:
        if isinstance(other, BoundaryConditions):
            return self.type == other.type and self._params == other._params
        return False
