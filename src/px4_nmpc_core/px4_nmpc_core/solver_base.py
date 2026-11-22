from abc import ABC, abstractmethod


class NmpcSolverBase(ABC):
    """Minimal solver contract used by the ROS node.

    Alternative backends (e.g. ACADOS) should implement the
    same method without changing the PX4/ROS integration.
    """

    @abstractmethod
    def solve(self, x0, xref, uprev):
        """Return (U_sequence, X_prediction, success, solve_ms, status)."""
        raise NotImplementedError
