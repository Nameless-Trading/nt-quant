from dataclasses import dataclass, field
from typing import Any
import datetime as dt

import cvxpy as cp
import numpy as np

from nt_quant.objectives import ObjectiveFunction, MeanVariance
from nt_quant.constraints import Constraint, MaxVariance, MaxVolatility, MaxTurnover


# =============================================================================
# Helper Functions for Active Risk Management
# =============================================================================


def compute_active_weights(
    weights: np.ndarray,
    benchmark_weights: np.ndarray,
) -> np.ndarray:
    """Compute active weights relative to a benchmark.

    Args:
        weights: Portfolio weights (n_assets,)
        benchmark_weights: Benchmark weights (n_assets,)

    Returns:
        Active weights (portfolio - benchmark)
    """
    return weights - benchmark_weights


def compute_active_risk(
    active_weights: np.ndarray,
    covariance_matrix: np.ndarray,
    annualize: bool = True,
) -> float:
    """Compute active risk (tracking error) of a portfolio.

    Args:
        active_weights: Active weights relative to benchmark (n_assets,)
        covariance_matrix: Covariance matrix (n_assets, n_assets)
        annualize: If True, annualize using sqrt(252)

    Returns:
        Active risk (tracking error)
    """
    variance = active_weights @ covariance_matrix @ active_weights.T
    volatility = np.sqrt(variance)
    if annualize:
        volatility *= np.sqrt(252)
    return float(volatility)


def predict_lambda(
    history: list[tuple[float, float]],
    target_active_risk: float,
) -> float:
    """Predict optimal lambda to achieve target active risk.

    Uses the relationship between lambda and active risk from historical
    optimization results to predict the lambda needed for a target risk.

    The model assumes: active_risk ≈ M / (2 * lambda)
    where M is a fitted constant.

    Args:
        history: List of (lambda, active_risk) tuples from previous optimizations
        target_active_risk: Target active risk to achieve

    Returns:
        Predicted lambda value
    """
    data = np.array(history)
    lambda_vals = data[:, 0]
    sigma_vals = data[:, 1]

    # Fit model: sigma = M / (2 * lambda)
    # Rearranged: M = sigma * 2 * lambda
    # Using least squares: X = 1 / (2 * lambda), fit M such that sigma ≈ M * X
    X = 1 / (2 * lambda_vals)
    M = np.dot(X, sigma_vals) / np.dot(X, X)

    # Predict lambda for target: target = M / (2 * lambda)
    # lambda = M / (2 * target)
    return M / (2 * target_active_risk)


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
class DynamicRiskResult:
    """Result of dynamic risk-targeted optimization.

    Attributes:
        optimization_result: The final optimization result
        lambda_: Final risk aversion parameter used
        active_risk: Achieved active risk (tracking error)
        target_active_risk: Target active risk
        iterations: Number of iterations to converge
        history: List of (lambda, active_risk) from each iteration
    """

    optimization_result: OptimizationResult
    lambda_: float
    active_risk: float
    target_active_risk: float
    iterations: int
    history: list[tuple[float, float]] = field(default_factory=list)

    @property
    def weights(self) -> np.ndarray:
        """Optimal portfolio weights."""
        return self.optimization_result.weights

    @property
    def is_converged(self) -> bool:
        """Whether optimization converged to target risk."""
        return abs(self.active_risk - self.target_active_risk) <= 0.005


@dataclass
class DynamicRiskOptimizer:
    """Portfolio optimizer that targets a specific active risk level.

    Iteratively adjusts the risk aversion parameter (lambda) to achieve
    a target tracking error relative to a benchmark.

    Args:
        constraints: List of portfolio constraints
        solver: CVXPY solver to use
        solver_options: Additional solver options
        verbose: Whether to print solver output
        max_iterations: Maximum iterations for lambda search
        tolerance: Convergence tolerance for active risk
        initial_lambda: Starting value for lambda search
    """

    constraints: list[Constraint] = field(default_factory=list)
    solver: str | None = None
    solver_options: dict[str, Any] = field(default_factory=dict)
    verbose: bool = False
    max_iterations: int = 5
    tolerance: float = 0.005
    initial_lambda: float = 100.0

    def optimize(
        self,
        asset_ids: list[str],
        alphas: np.ndarray,
        covariance_matrix: np.ndarray,
        benchmark_weights: np.ndarray,
        date: dt.date,
        target_active_risk: float = 0.05,
        previous_weights: np.ndarray | None = None,
    ) -> DynamicRiskResult:
        """Run dynamic optimization targeting a specific active risk.

        Args:
            asset_ids: List of asset IDs
            alphas: Alpha signals (n_assets,)
            covariance_matrix: Covariance matrix (n_assets, n_assets)
            benchmark_weights: Benchmark portfolio weights (n_assets,)
            date: Optimization date
            target_active_risk: Target active risk (annualized, e.g., 0.05 for 5%)
            previous_weights: Previous period weights (for turnover constraints)

        Returns:
            DynamicRiskResult with optimal weights and convergence info
        """
        active_risk = float("inf")
        lambda_ = None
        history: list[tuple[float, float]] = []
        iterations = 0
        result = None

        while abs(active_risk - target_active_risk) > self.tolerance:
            if lambda_ is None:
                lambda_ = self.initial_lambda
            else:
                lambda_ = predict_lambda(history, target_active_risk)

            # Create optimizer with current lambda
            objective = MeanVariance(risk_aversion=lambda_)
            optimizer = PortfolioOptimizer(
                objective=objective,
                constraints=self.constraints.copy(),
                solver=self.solver,
                solver_options=self.solver_options,
                verbose=self.verbose,
            )

            result = optimizer.optimize(
                asset_ids=asset_ids,
                alphas=alphas,
                covariance_matrix=covariance_matrix,
                date=date,
                previous_weights=previous_weights,
            )

            # Compute active risk
            active_weights = compute_active_weights(result.weights, benchmark_weights)
            active_risk = compute_active_risk(active_weights, covariance_matrix)

            history.append((lambda_, active_risk))
            iterations += 1

            if iterations >= self.max_iterations:
                break

        return DynamicRiskResult(
            optimization_result=result,
            lambda_=lambda_,
            active_risk=active_risk,
            target_active_risk=target_active_risk,
            iterations=iterations,
            history=history,
        )
