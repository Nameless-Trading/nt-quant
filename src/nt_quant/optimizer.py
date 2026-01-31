from dataclasses import dataclass, field
from typing import Any
import datetime as dt

import cvxpy as cp
import numpy as np

from nt_quant.objectives import ObjectiveFunction
from nt_quant.constraints import Constraint, MaxVariance, MaxVolatility, MaxTurnover


@dataclass
class OptimizationResult:
    """Result of a single-period portfolio optimization.

    Attributes:
        weights: Optimal portfolio weights (n_assets,)
        asset_ids: Asset IDs corresponding to weights
        date: Optimization date
        status: Solver status (optimal, infeasible, etc.)
        objective_value: Optimal objective value
        solve_time: Time to solve in seconds
        diagnostics: Additional solver diagnostics
    """

    weights: np.ndarray
    asset_ids: list[str]
    date: dt.date
    status: str
    objective_value: float | None
    solve_time: float
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def is_optimal(self) -> bool:
        """Whether the optimization found an optimal solution."""
        return self.status in ("optimal", "optimal_inaccurate")

    def to_dict(self) -> dict[str, float]:
        """Return weights as asset_id -> weight dictionary."""
        return dict(zip(self.asset_ids, self.weights.tolist()))


@dataclass
class PortfolioOptimizer:
    """Single-period mean-variance portfolio optimizer.

    Uses CVXPY for convex optimization with configurable objectives
    and constraints.

    Args:
        objective: Objective function to maximize
        constraints: List of portfolio constraints
        solver: CVXPY solver to use (default: OSQP for QP, ECOS for SOCP)
        solver_options: Additional solver options
        verbose: Whether to print solver output
    """

    objective: ObjectiveFunction
    constraints: list[Constraint] = field(default_factory=list)
    solver: str | None = None
    solver_options: dict[str, Any] = field(default_factory=dict)
    verbose: bool = False

    def optimize(
        self,
        asset_ids: list[str],
        alphas: np.ndarray,
        covariance_matrix: np.ndarray,
        date: dt.date,
        previous_weights: np.ndarray | None = None,
    ) -> OptimizationResult:
        """Run portfolio optimization for a single period.

        Args:
            asset_ids: List of asset IDs
            alphas: Alpha signals (n_assets,)
            covariance_matrix: Covariance matrix (n_assets, n_assets)
            date: Optimization date
            previous_weights: Previous period weights (for turnover constraints)

        Returns:
            OptimizationResult with optimal weights and diagnostics
        """
        n_assets = len(asset_ids)

        # Validate inputs
        if len(alphas) != n_assets:
            raise ValueError(f"alphas length ({len(alphas)}) != n_assets ({n_assets})")
        if covariance_matrix.shape != (n_assets, n_assets):
            raise ValueError(
                f"covariance_matrix shape {covariance_matrix.shape} != ({n_assets}, {n_assets})"
            )

        # Create optimization variable
        weights = cp.Variable(n_assets, name="weights")

        # Build objective
        objective_expr = self.objective.build(weights, alphas, covariance_matrix)
        objective = cp.Maximize(objective_expr)

        # Build constraints
        all_constraints = []
        for constraint in self.constraints:
            # Inject covariance matrix for risk constraints
            if isinstance(constraint, (MaxVariance, MaxVolatility)):
                constraint.set_covariance_matrix(covariance_matrix)

            # Inject previous weights for turnover constraints
            if isinstance(constraint, MaxTurnover) and previous_weights is not None:
                constraint.set_previous_weights(previous_weights)

            constraint_list = constraint.build(weights, asset_ids, date)
            all_constraints.extend(constraint_list)

        # Formulate and solve problem
        problem = cp.Problem(objective, all_constraints)

        # Select solver
        solver = self._select_solver(problem)

        try:
            import time

            start_time = time.time()
            problem.solve(solver=solver, verbose=self.verbose, **self.solver_options)
            solve_time = time.time() - start_time
        except cp.SolverError as e:
            return OptimizationResult(
                weights=np.zeros(n_assets),
                asset_ids=asset_ids,
                date=date,
                status="solver_error",
                objective_value=None,
                solve_time=0.0,
                diagnostics={"error": str(e)},
            )

        # Extract results
        if weights.value is not None:
            optimal_weights = weights.value
        else:
            optimal_weights = np.zeros(n_assets)

        return OptimizationResult(
            weights=optimal_weights,
            asset_ids=asset_ids,
            date=date,
            status=problem.status,
            objective_value=problem.value,
            solve_time=solve_time,
            diagnostics={
                "solver": str(solver),
                "num_constraints": len(all_constraints),
                "problem_size": n_assets,
            },
        )

    def _select_solver(self, problem: cp.Problem) -> Any:
        """Select appropriate solver based on problem structure."""
        if self.solver is not None:
            return getattr(cp, self.solver, self.solver)

        # Auto-select based on problem type
        if problem.is_qp():
            # Quadratic program - use OSQP
            return cp.OSQP
        elif problem.is_dcp():
            # Generic convex - use ECOS
            return cp.ECOS
        else:
            # Default
            return cp.SCS

    def add_constraint(self, constraint: Constraint) -> None:
        """Add a constraint to the optimizer."""
        self.constraints.append(constraint)

    def remove_constraint(self, constraint_name: str) -> bool:
        """Remove a constraint by name. Returns True if found and removed."""
        for i, c in enumerate(self.constraints):
            if c.name == constraint_name:
                self.constraints.pop(i)
                return True
        return False


@dataclass
class RobustOptimizer(PortfolioOptimizer):
    """Portfolio optimizer with robustness to estimation error.

    Implements robust optimization by adding uncertainty sets
    around alpha estimates.

    Args:
        alpha_uncertainty: Uncertainty radius for alpha estimates
        cov_uncertainty: Uncertainty scaling for covariance matrix
    """

    alpha_uncertainty: float = 0.0
    cov_uncertainty: float = 0.0

    def optimize(
        self,
        asset_ids: list[str],
        alphas: np.ndarray,
        covariance_matrix: np.ndarray,
        date: dt.date,
        previous_weights: np.ndarray | None = None,
    ) -> OptimizationResult:
        """Run robust portfolio optimization.

        If alpha_uncertainty > 0, uses worst-case alpha within uncertainty set.
        If cov_uncertainty > 0, inflates covariance matrix.
        """
        n_assets = len(asset_ids)
        weights = cp.Variable(n_assets, name="weights")

        # Robust alpha: worst case is alpha - uncertainty * |w|
        if self.alpha_uncertainty > 0:
            # Robust counterpart: max w'alpha - kappa * ||w||
            # where kappa is the uncertainty radius
            robust_alpha_term = (
                weights @ alphas - self.alpha_uncertainty * cp.norm(weights, 2)
            )
        else:
            robust_alpha_term = weights @ alphas

        # Robust covariance: inflate by uncertainty factor
        if self.cov_uncertainty > 0:
            robust_cov = covariance_matrix * (1 + self.cov_uncertainty)
        else:
            robust_cov = covariance_matrix

        # Build objective with robust terms
        objective_expr = self.objective.build(weights, alphas, robust_cov)

        # If objective uses alpha directly, we need to modify it
        # For simplicity, we add a penalty term
        if self.alpha_uncertainty > 0:
            # Subtract uncertainty penalty from objective
            objective_expr = objective_expr - self.alpha_uncertainty * cp.norm(weights, 2)

        objective = cp.Maximize(objective_expr)

        # Build constraints (using robust covariance for risk constraints)
        all_constraints = []
        for constraint in self.constraints:
            if isinstance(constraint, (MaxVariance, MaxVolatility)):
                constraint.set_covariance_matrix(robust_cov)
            if isinstance(constraint, MaxTurnover) and previous_weights is not None:
                constraint.set_previous_weights(previous_weights)

            constraint_list = constraint.build(weights, asset_ids, date)
            all_constraints.extend(constraint_list)

        # Solve
        problem = cp.Problem(objective, all_constraints)
        solver = self._select_solver(problem)

        try:
            import time

            start_time = time.time()
            problem.solve(solver=solver, verbose=self.verbose, **self.solver_options)
            solve_time = time.time() - start_time
        except cp.SolverError as e:
            return OptimizationResult(
                weights=np.zeros(n_assets),
                asset_ids=asset_ids,
                date=date,
                status="solver_error",
                objective_value=None,
                solve_time=0.0,
                diagnostics={"error": str(e)},
            )

        if weights.value is not None:
            optimal_weights = weights.value
        else:
            optimal_weights = np.zeros(n_assets)

        return OptimizationResult(
            weights=optimal_weights,
            asset_ids=asset_ids,
            date=date,
            status=problem.status,
            objective_value=problem.value,
            solve_time=solve_time,
            diagnostics={
                "solver": str(solver),
                "alpha_uncertainty": self.alpha_uncertainty,
                "cov_uncertainty": self.cov_uncertainty,
            },
        )
