from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol
import datetime as dt

import cvxpy as cp
import numpy as np


class ObjectiveFunction(ABC):
    """Abstract base class for objective functions in MVO optimization."""

    @abstractmethod
    def build(
        self,
        weights: cp.Variable,
        alphas: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> cp.Expression:
        """Build the CVXPY objective expression.

        Args:
            weights: CVXPY variable representing portfolio weights (n_assets,)
            alphas: Alpha signals for each asset (n_assets,)
            covariance_matrix: Covariance matrix (n_assets, n_assets)

        Returns:
            CVXPY expression to be maximized
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the objective function for logging/display."""
        pass


@dataclass
class MaximizeAlpha(ObjectiveFunction):
    """Maximize expected return (alpha).

    Objective: max w' * alpha
    """

    def build(
        self,
        weights: cp.Variable,
        alphas: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> cp.Expression:
        return weights @ alphas

    @property
    def name(self) -> str:
        return "MaximizeAlpha"


@dataclass
class MinimizeVariance(ObjectiveFunction):
    """Minimize portfolio variance.

    Objective: max -w' * Sigma * w
    """

    def build(
        self,
        weights: cp.Variable,
        alphas: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> cp.Expression:
        return -cp.quad_form(weights, covariance_matrix)

    @property
    def name(self) -> str:
        return "MinimizeVariance"


@dataclass
class MeanVariance(ObjectiveFunction):
    """Classic mean-variance optimization.

    Objective: max w' * alpha - (risk_aversion / 2) * w' * Sigma * w

    Args:
        risk_aversion: Risk aversion parameter (lambda). Higher values
            penalize variance more heavily. Default is 1.0.
    """

    risk_aversion: float = 1.0

    def build(
        self,
        weights: cp.Variable,
        alphas: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> cp.Expression:
        expected_return = weights @ alphas
        variance = cp.quad_form(weights, covariance_matrix)
        return expected_return - (self.risk_aversion / 2) * variance

    @property
    def name(self) -> str:
        return f"MeanVariance(lambda={self.risk_aversion})"


@dataclass
class MaximizeSharpe(ObjectiveFunction):
    """Maximize Sharpe ratio (approximation via risk budgeting).

    Uses the standard MVO formulation but normalizes by predicted volatility.
    Note: This is an approximation since Sharpe ratio is not convex.

    Objective: max w' * alpha / sqrt(w' * Sigma * w)

    We reformulate as: max w' * alpha subject to w' * Sigma * w <= target_vol^2
    """

    target_volatility: float = 0.10  # 10% annualized vol

    def build(
        self,
        weights: cp.Variable,
        alphas: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> cp.Expression:
        # Maximize alpha (variance constraint added separately)
        return weights @ alphas

    @property
    def name(self) -> str:
        return f"MaximizeSharpe(target_vol={self.target_volatility})"


@dataclass
class RiskParity(ObjectiveFunction):
    """Risk parity objective (equal risk contribution).

    Minimizes deviation from equal risk contribution.
    Note: This is a simplified version - true risk parity is non-convex.

    Uses log-barrier approximation for risk parity.
    """

    def build(
        self,
        weights: cp.Variable,
        alphas: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> cp.Expression:
        # Simplified: minimize variance while encouraging diversification
        # via log-barrier on weights
        n_assets = weights.shape[0]
        variance = cp.quad_form(weights, covariance_matrix)
        # Log barrier encourages non-zero weights
        diversification = cp.sum(cp.log(weights))
        return -variance + diversification / n_assets

    @property
    def name(self) -> str:
        return "RiskParity"


@dataclass
class MaximizeInfoRatio(ObjectiveFunction):
    """Maximize information ratio relative to a benchmark.

    Objective: max (w - w_bench)' * alpha / sqrt((w - w_bench)' * Sigma * (w - w_bench))

    Reformulated as MVO with active weights.

    Args:
        benchmark_weights: Benchmark portfolio weights (n_assets,)
        risk_aversion: Risk aversion on tracking error
    """

    benchmark_weights: np.ndarray | None = None
    risk_aversion: float = 1.0

    def build(
        self,
        weights: cp.Variable,
        alphas: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> cp.Expression:
        if self.benchmark_weights is None:
            # Equal weight benchmark
            n_assets = weights.shape[0]
            bench = np.ones(n_assets) / n_assets
        else:
            bench = self.benchmark_weights

        active_weights = weights - bench
        active_return = active_weights @ alphas
        tracking_variance = cp.quad_form(active_weights, covariance_matrix)

        return active_return - (self.risk_aversion / 2) * tracking_variance

    @property
    def name(self) -> str:
        return f"MaximizeInfoRatio(lambda={self.risk_aversion})"
