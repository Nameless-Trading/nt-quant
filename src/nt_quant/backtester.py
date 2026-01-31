from dataclasses import dataclass
import datetime as dt
import logging
import os

# Suppress Ray GPU warning for CPU-only usage
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

import polars as pl
import dataframely as dy
import ray

from nt_quant.models import Alphas, Calendar, FactorLoadings, FactorCovariances, IdioVol
from nt_quant.objectives import ObjectiveFunction
from nt_quant.constraints import Constraint
from nt_quant.covariance import build_covariance_matrix
from nt_quant.optimizer import PortfolioOptimizer


logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration options for MVO backtest.

    Attributes:
        parallel: Use Ray for parallel execution
        fail_on_infeasible: Raise error if optimization fails
        log_progress: Log progress during backtest
    """

    parallel: bool = True
    fail_on_infeasible: bool = False
    log_progress: bool = True


@dataclass
class BacktestResult:
    """Results from MVO backtest.

    Attributes:
        weights: DataFrame with optimal weights for each date
        failed_dates: Dates where optimization failed
    """

    weights: pl.DataFrame
    failed_dates: list[dt.date]

    @property
    def n_periods(self) -> int:
        """Number of unique dates in results."""
        return self.weights.select("date").n_unique()

    @property
    def success_rate(self) -> float:
        """Fraction of periods with successful optimization."""
        total = self.n_periods + len(self.failed_dates)
        if total == 0:
            return 0.0
        return self.n_periods / total

    def get_weights_for_date(self, date: dt.date) -> dict[str, float]:
        """Get weights for a specific date as dict."""
        weights_df = self.weights.filter(pl.col("date") == date)
        return dict(
            zip(
                weights_df["asset_id"].to_list(),
                weights_df["weight"].to_list(),
            )
        )

    def to_wide_format(self) -> pl.DataFrame:
        """Convert weights to wide format (dates x assets)."""
        return self.weights.pivot(index="date", on="asset_id", values="weight").sort("date")


@ray.remote
def _backtest_step_parallel(
    date: dt.date,
    alphas: pl.DataFrame,
    factor_loadings: pl.DataFrame,
    factor_covariances: pl.DataFrame,
    idio_vol: pl.DataFrame,
    objective: ObjectiveFunction,
    constraints: list[Constraint],
) -> tuple[pl.DataFrame | None, dt.date | None]:
    """Run optimization for a single date (Ray remote function)."""
    # Filter alphas for this date
    alphas_slice = alphas.filter(pl.col("date") == date).sort("asset_id")
    asset_ids = alphas_slice["asset_id"].to_list()

    if len(asset_ids) == 0:
        return None, date

    # Build covariance matrix
    covariance_matrix = build_covariance_matrix(
        asset_ids, date, factor_loadings, factor_covariances, idio_vol
    )

    # Get alpha values
    alpha_values = alphas_slice["alpha"].to_numpy()

    # Run optimization
    optimizer = PortfolioOptimizer(objective=objective, constraints=constraints)
    result = optimizer.optimize(
        asset_ids=asset_ids,
        alphas=alpha_values,
        covariance_matrix=covariance_matrix,
        date=date,
    )

    if result.is_optimal:
        weights_df = pl.DataFrame({
            "date": [date] * len(asset_ids),
            "asset_id": asset_ids,
            "weight": result.weights.tolist(),
        })
        return weights_df, None
    else:
        return None, date


def _backtest_step_sequential(
    date: dt.date,
    alphas: pl.DataFrame,
    factor_loadings: pl.DataFrame,
    factor_covariances: pl.DataFrame,
    idio_vol: pl.DataFrame,
    objective: ObjectiveFunction,
    constraints: list[Constraint],
) -> tuple[pl.DataFrame | None, dt.date | None]:
    """Run optimization for a single date (sequential)."""
    # Filter alphas for this date
    alphas_slice = alphas.filter(pl.col("date") == date).sort("asset_id")
    asset_ids = alphas_slice["asset_id"].to_list()

    if len(asset_ids) == 0:
        return None, date

    # Build covariance matrix
    covariance_matrix = build_covariance_matrix(
        asset_ids, date, factor_loadings, factor_covariances, idio_vol
    )

    # Get alpha values
    alpha_values = alphas_slice["alpha"].to_numpy()

    # Run optimization
    optimizer = PortfolioOptimizer(objective=objective, constraints=constraints)
    result = optimizer.optimize(
        asset_ids=asset_ids,
        alphas=alpha_values,
        covariance_matrix=covariance_matrix,
        date=date,
    )

    if result.is_optimal:
        weights_df = pl.DataFrame({
            "date": [date] * len(asset_ids),
            "asset_id": asset_ids,
            "weight": result.weights.tolist(),
        })
        return weights_df, None
    else:
        return None, date


class MVOBacktester:
    """Mean-Variance Optimization backtester.

    Supports both sequential and Ray-parallel execution.

    Args:
        objective: Objective function to maximize
        constraints: List of portfolio constraints
        config: Backtest configuration options
    """

    def __init__(
        self,
        objective: ObjectiveFunction,
        constraints: list[Constraint],
        config: BacktestConfig | None = None,
    ):
        self.objective = objective
        self.constraints = constraints
        self.config = config or BacktestConfig()

    def run(
        self,
        start_date: dt.date,
        end_date: dt.date,
        calendar: dy.DataFrame[Calendar],
        alphas: dy.DataFrame[Alphas],
        factor_loadings: dy.DataFrame[FactorLoadings],
        factor_covariances: dy.DataFrame[FactorCovariances],
        idio_vol: dy.DataFrame[IdioVol],
    ) -> BacktestResult:
        """Run the MVO backtest.

        Args:
            start_date: First date to optimize
            end_date: Last date to optimize (inclusive)
            calendar: Trading calendar with valid dates
            alphas: Alpha signals for each asset/date
            factor_loadings: Factor loadings for covariance estimation
            factor_covariances: Factor covariance matrix for each date
            idio_vol: Idiosyncratic volatility for each asset/date

        Returns:
            BacktestResult with weights and failed dates
        """
        # Get trading dates
        dates = (
            calendar.filter(pl.col("date") >= start_date)
            .filter(pl.col("date") <= end_date)
            .sort("date")["date"]
            .to_list()
        )

        if self.config.log_progress:
            logger.info(f"Running backtest from {start_date} to {end_date}")
            logger.info(f"Number of trading days: {len(dates)}")

        if self.config.parallel:
            return self._run_parallel(
                dates, alphas, factor_loadings, factor_covariances, idio_vol
            )
        else:
            return self._run_sequential(
                dates, alphas, factor_loadings, factor_covariances, idio_vol
            )

    def _run_parallel(
        self,
        dates: list[dt.date],
        alphas: dy.DataFrame[Alphas],
        factor_loadings: dy.DataFrame[FactorLoadings],
        factor_covariances: dy.DataFrame[FactorCovariances],
        idio_vol: dy.DataFrame[IdioVol],
    ) -> BacktestResult:
        """Run backtest in parallel using Ray."""
        # Initialize Ray
        ray.init(
            ignore_reinit_error=True,
            num_cpus=os.cpu_count(),
        )

        # Put DataFrames in Ray's object store to avoid repeated serialization
        alphas_ref = ray.put(pl.DataFrame(alphas))
        factor_loadings_ref = ray.put(pl.DataFrame(factor_loadings))
        factor_covariances_ref = ray.put(pl.DataFrame(factor_covariances))
        idio_vol_ref = ray.put(pl.DataFrame(idio_vol))
        objective_ref = ray.put(self.objective)
        constraints_ref = ray.put(self.constraints)

        # Launch parallel tasks
        futures = [
            _backtest_step_parallel.remote(
                date,
                alphas_ref,
                factor_loadings_ref,
                factor_covariances_ref,
                idio_vol_ref,
                objective_ref,
                constraints_ref,
            )
            for date in dates
        ]

        # Collect results
        results = ray.get(futures)

        # Separate weights and failed dates
        weights_list = [r[0] for r in results if r[0] is not None]
        failed_dates = [r[1] for r in results if r[1] is not None]

        if weights_list:
            weights = pl.concat(weights_list).sort("date", "asset_id")
        else:
            weights = pl.DataFrame(
                schema={"date": pl.Date, "asset_id": pl.String, "weight": pl.Float64}
            )

        return BacktestResult(weights=weights, failed_dates=failed_dates)

    def _run_sequential(
        self,
        dates: list[dt.date],
        alphas: dy.DataFrame[Alphas],
        factor_loadings: dy.DataFrame[FactorLoadings],
        factor_covariances: dy.DataFrame[FactorCovariances],
        idio_vol: dy.DataFrame[IdioVol],
    ) -> BacktestResult:
        """Run backtest sequentially."""
        weights_list = []
        failed_dates = []

        for i, date in enumerate(dates):
            if self.config.log_progress and i % 50 == 0:
                logger.info(f"Processing date {i + 1}/{len(dates)}: {date}")

            weights_df, failed_date = _backtest_step_sequential(
                date,
                pl.DataFrame(alphas),
                pl.DataFrame(factor_loadings),
                pl.DataFrame(factor_covariances),
                pl.DataFrame(idio_vol),
                self.objective,
                self.constraints,
            )

            if weights_df is not None:
                weights_list.append(weights_df)
            if failed_date is not None:
                failed_dates.append(failed_date)

                if self.config.fail_on_infeasible:
                    raise ValueError(f"Optimization failed on {failed_date}")

        if weights_list:
            weights = pl.concat(weights_list).sort("date", "asset_id")
        else:
            weights = pl.DataFrame(
                schema={"date": pl.Date, "asset_id": pl.String, "weight": pl.Float64}
            )

        return BacktestResult(weights=weights, failed_dates=failed_dates)
