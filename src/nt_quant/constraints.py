from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import datetime as dt

import cvxpy as cp
import numpy as np
import polars as pl
import dataframely as dy

from nt_quant.models import Betas, AssetCharacteristics


class Constraint(ABC):
    """Abstract base class for portfolio constraints.

    Constraints are built fresh for each optimization date, allowing
    them to use time-varying data (e.g., betas, sector memberships).
    """

    @abstractmethod
    def build(
        self,
        weights: cp.Variable,
        asset_ids: list[str],
        date: dt.date,
    ) -> list[cp.Constraint]:
        """Build CVXPY constraint(s) for the given date.

        Args:
            weights: CVXPY variable representing portfolio weights (n_assets,)
            asset_ids: List of asset IDs in the same order as weights
            date: The optimization date (for time-varying constraints)

        Returns:
            List of CVXPY constraints
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the constraint for logging/display."""
        pass


# =============================================================================
# Basic Portfolio Constraints
# =============================================================================


@dataclass
class FullyInvested(Constraint):
    """Weights must sum to 1 (fully invested, long-only implied)."""

    def build(
        self,
        weights: cp.Variable,
        asset_ids: list[str],
        date: dt.date,
    ) -> list[cp.Constraint]:
        return [cp.sum(weights) == 1]

    @property
    def name(self) -> str:
        return "FullyInvested"


@dataclass
class DollarNeutral(Constraint):
    """Weights must sum to 0 (dollar-neutral long/short)."""

    def build(
        self,
        weights: cp.Variable,
        asset_ids: list[str],
        date: dt.date,
    ) -> list[cp.Constraint]:
        return [cp.sum(weights) == 0]

    @property
    def name(self) -> str:
        return "DollarNeutral"


@dataclass
class GrossExposure(Constraint):
    """Constraint on total gross exposure (sum of absolute weights).

    Args:
        max_exposure: Maximum gross exposure (e.g., 2.0 for 200%)
    """

    max_exposure: float = 2.0

    def build(
        self,
        weights: cp.Variable,
        asset_ids: list[str],
        date: dt.date,
    ) -> list[cp.Constraint]:
        return [cp.norm(weights, 1) <= self.max_exposure]

    @property
    def name(self) -> str:
        return f"GrossExposure(max={self.max_exposure})"


@dataclass
class LongOnly(Constraint):
    """All weights must be non-negative."""

    def build(
        self,
        weights: cp.Variable,
        asset_ids: list[str],
        date: dt.date,
    ) -> list[cp.Constraint]:
        return [weights >= 0]

    @property
    def name(self) -> str:
        return "LongOnly"


@dataclass
class BoxConstraint(Constraint):
    """Bound individual weights between min and max.

    Args:
        min_weight: Minimum weight per asset (can be negative for shorts)
        max_weight: Maximum weight per asset
    """

    min_weight: float = -0.10
    max_weight: float = 0.10

    def build(
        self,
        weights: cp.Variable,
        asset_ids: list[str],
        date: dt.date,
    ) -> list[cp.Constraint]:
        return [
            weights >= self.min_weight,
            weights <= self.max_weight,
        ]

    @property
    def name(self) -> str:
        return f"BoxConstraint(min={self.min_weight}, max={self.max_weight})"


# =============================================================================
# Risk Constraints
# =============================================================================


@dataclass
class MaxVariance(Constraint):
    """Constrain portfolio variance.

    Args:
        max_variance: Maximum portfolio variance
        covariance_matrix_getter: Callable that returns covariance matrix for date
    """

    max_variance: float
    covariance_matrix: np.ndarray | None = None

    def set_covariance_matrix(self, cov: np.ndarray) -> None:
        """Set the covariance matrix (called by optimizer before building)."""
        self.covariance_matrix = cov

    def build(
        self,
        weights: cp.Variable,
        asset_ids: list[str],
        date: dt.date,
    ) -> list[cp.Constraint]:
        if self.covariance_matrix is None:
            raise ValueError("Covariance matrix not set. Call set_covariance_matrix first.")
        return [cp.quad_form(weights, self.covariance_matrix) <= self.max_variance]

    @property
    def name(self) -> str:
        return f"MaxVariance(max={self.max_variance})"


@dataclass
class MaxVolatility(Constraint):
    """Constrain portfolio volatility (standard deviation).

    Args:
        max_volatility: Maximum portfolio volatility (annualized)
    """

    max_volatility: float
    covariance_matrix: np.ndarray | None = None

    def set_covariance_matrix(self, cov: np.ndarray) -> None:
        """Set the covariance matrix (called by optimizer before building)."""
        self.covariance_matrix = cov

    def build(
        self,
        weights: cp.Variable,
        asset_ids: list[str],
        date: dt.date,
    ) -> list[cp.Constraint]:
        if self.covariance_matrix is None:
            raise ValueError("Covariance matrix not set. Call set_covariance_matrix first.")
        # sqrt(w' * Sigma * w) <= max_vol
        # Equivalent to: w' * Sigma * w <= max_vol^2
        return [cp.quad_form(weights, self.covariance_matrix) <= self.max_volatility**2]

    @property
    def name(self) -> str:
        return f"MaxVolatility(max={self.max_volatility})"


# =============================================================================
# Factor Exposure Constraints
# =============================================================================


@dataclass
class BetaNeutral(Constraint):
    """Portfolio beta must equal zero (market neutral).

    Args:
        betas: DataFrame with asset betas over time
    """

    betas: dy.DataFrame[Betas] | None = None

    def build(
        self,
        weights: cp.Variable,
        asset_ids: list[str],
        date: dt.date,
    ) -> list[cp.Constraint]:
        if self.betas is None:
            raise ValueError("Betas DataFrame not provided")

        betas_today = (
            self.betas.filter(pl.col("date") == date)
            .filter(pl.col("asset_id").is_in(asset_ids))
            .sort("asset_id")
        )
        beta_vec = betas_today["beta"].to_numpy()

        return [weights @ beta_vec == 0]

    @property
    def name(self) -> str:
        return "BetaNeutral"


@dataclass
class TargetBeta(Constraint):
    """Portfolio beta must equal target value.

    Args:
        target: Target portfolio beta
        betas: DataFrame with asset betas over time
    """

    target: float = 1.0
    betas: dy.DataFrame[Betas] | None = None

    def build(
        self,
        weights: cp.Variable,
        asset_ids: list[str],
        date: dt.date,
    ) -> list[cp.Constraint]:
        if self.betas is None:
            raise ValueError("Betas DataFrame not provided")

        betas_today = (
            self.betas.filter(pl.col("date") == date)
            .filter(pl.col("asset_id").is_in(asset_ids))
            .sort("asset_id")
        )
        beta_vec = betas_today["beta"].to_numpy()

        return [weights @ beta_vec == self.target]

    @property
    def name(self) -> str:
        return f"TargetBeta(target={self.target})"


@dataclass
class BetaBounds(Constraint):
    """Portfolio beta must be within bounds.

    Args:
        min_beta: Minimum portfolio beta
        max_beta: Maximum portfolio beta
        betas: DataFrame with asset betas over time
    """

    min_beta: float = 0.8
    max_beta: float = 1.2
    betas: dy.DataFrame[Betas] | None = None

    def build(
        self,
        weights: cp.Variable,
        asset_ids: list[str],
        date: dt.date,
    ) -> list[cp.Constraint]:
        if self.betas is None:
            raise ValueError("Betas DataFrame not provided")

        betas_today = (
            self.betas.filter(pl.col("date") == date)
            .filter(pl.col("asset_id").is_in(asset_ids))
            .sort("asset_id")
        )
        beta_vec = betas_today["beta"].to_numpy()
        portfolio_beta = weights @ beta_vec

        return [
            portfolio_beta >= self.min_beta,
            portfolio_beta <= self.max_beta,
        ]

    @property
    def name(self) -> str:
        return f"BetaBounds(min={self.min_beta}, max={self.max_beta})"


# =============================================================================
# Sector / Group Constraints
# =============================================================================


@dataclass
class SectorNeutral(Constraint):
    """Each sector must have zero net exposure.

    Args:
        sector_map: Dict mapping asset_id -> sector, or DataFrame with mappings
    """

    sector_map: dict[str, str] | pl.DataFrame | None = None

    def build(
        self,
        weights: cp.Variable,
        asset_ids: list[str],
        date: dt.date,
    ) -> list[cp.Constraint]:
        if self.sector_map is None:
            raise ValueError("Sector map not provided")

        # Convert to dict if DataFrame
        if isinstance(self.sector_map, pl.DataFrame):
            sector_dict = dict(
                zip(
                    self.sector_map["asset_id"].to_list(),
                    self.sector_map["sector"].to_list(),
                )
            )
        else:
            sector_dict = self.sector_map

        # Get unique sectors
        sectors = set(sector_dict.get(aid, "Unknown") for aid in asset_ids)

        constraints = []
        for sector in sectors:
            # Create indicator vector for this sector
            indicator = np.array(
                [1.0 if sector_dict.get(aid) == sector else 0.0 for aid in asset_ids]
            )
            constraints.append(weights @ indicator == 0)

        return constraints

    @property
    def name(self) -> str:
        return "SectorNeutral"


@dataclass
class SectorBounds(Constraint):
    """Bound exposure to each sector.

    Args:
        min_exposure: Minimum exposure per sector
        max_exposure: Maximum exposure per sector
        sector_map: Dict mapping asset_id -> sector
    """

    min_exposure: float = -0.20
    max_exposure: float = 0.20
    sector_map: dict[str, str] | None = None

    def build(
        self,
        weights: cp.Variable,
        asset_ids: list[str],
        date: dt.date,
    ) -> list[cp.Constraint]:
        if self.sector_map is None:
            raise ValueError("Sector map not provided")

        sectors = set(self.sector_map.get(aid, "Unknown") for aid in asset_ids)

        constraints = []
        for sector in sectors:
            indicator = np.array(
                [1.0 if self.sector_map.get(aid) == sector else 0.0 for aid in asset_ids]
            )
            sector_exposure = weights @ indicator
            constraints.extend(
                [
                    sector_exposure >= self.min_exposure,
                    sector_exposure <= self.max_exposure,
                ]
            )

        return constraints

    @property
    def name(self) -> str:
        return f"SectorBounds(min={self.min_exposure}, max={self.max_exposure})"


# =============================================================================
# Turnover Constraints
# =============================================================================


@dataclass
class MaxTurnover(Constraint):
    """Limit turnover from previous weights.

    Args:
        max_turnover: Maximum one-way turnover (e.g., 0.20 for 20%)
        previous_weights: Previous portfolio weights (set before optimization)
    """

    max_turnover: float = 0.20
    previous_weights: np.ndarray | None = None

    def set_previous_weights(self, weights: np.ndarray) -> None:
        """Set the previous weights (called by optimizer before building)."""
        self.previous_weights = weights

    def build(
        self,
        weights: cp.Variable,
        asset_ids: list[str],
        date: dt.date,
    ) -> list[cp.Constraint]:
        if self.previous_weights is None:
            # No previous weights (first period) - no constraint
            return []

        # Turnover = 0.5 * sum(|w_new - w_old|)
        return [cp.norm(weights - self.previous_weights, 1) <= 2 * self.max_turnover]

    @property
    def name(self) -> str:
        return f"MaxTurnover(max={self.max_turnover})"


# =============================================================================
# Cardinality Constraints (Non-convex - requires special handling)
# =============================================================================


@dataclass
class MaxPositions(Constraint):
    """Limit the number of non-zero positions.

    Note: This is non-convex. Implementation uses big-M relaxation.
    For true cardinality constraints, use a MIP solver.

    Args:
        max_positions: Maximum number of positions
        min_weight_threshold: Minimum weight to count as a position
    """

    max_positions: int = 50
    min_weight_threshold: float = 0.001

    def build(
        self,
        weights: cp.Variable,
        asset_ids: list[str],
        date: dt.date,
    ) -> list[cp.Constraint]:
        # Note: True cardinality requires MIP. This is a relaxation.
        # For exact solution, user should use a MIP-capable solver
        # and binary indicator variables.

        # Placeholder: just return empty (user should implement MIP version)
        # or use L1 regularization in objective as soft constraint
        return []

    @property
    def name(self) -> str:
        return f"MaxPositions(max={self.max_positions})"


# =============================================================================
# Custom / User-Defined Constraints
# =============================================================================


@dataclass
class LinearConstraint(Constraint):
    """Generic linear constraint: A @ w == b or A @ w <= b.

    Args:
        A: Constraint matrix (n_constraints, n_assets)
        b: RHS vector (n_constraints,)
        equality: If True, use equality constraint; else inequality (<=)
    """

    A: np.ndarray | None = None
    b: np.ndarray | None = None
    equality: bool = True

    def build(
        self,
        weights: cp.Variable,
        asset_ids: list[str],
        date: dt.date,
    ) -> list[cp.Constraint]:
        if self.A is None or self.b is None:
            raise ValueError("Constraint matrix A and vector b must be provided")

        if self.equality:
            return [self.A @ weights == self.b]
        else:
            return [self.A @ weights <= self.b]

    @property
    def name(self) -> str:
        return f"LinearConstraint(equality={self.equality})"
