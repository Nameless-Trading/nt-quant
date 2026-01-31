# nt-quant: Mean-Variance Optimization Backtesting Framework

# Data Models / Schemas
from nt_quant.models import (
    Alphas,
    Betas,
    AssetCharacteristics,
    BacktestResults,
    Calendar,
    FactorCovariances,
    FactorLoadings,
    IdioVol,
    Weights,
)

# Objective Functions
from nt_quant.objectives import (
    ObjectiveFunction,
    MaximizeAlpha,
    MinimizeVariance,
    MeanVariance,
    MaximizeSharpe,
    RiskParity,
    MaximizeInfoRatio,
)

# Constraints
from nt_quant.constraints import (
    Constraint,
    # Basic constraints
    FullyInvested,
    DollarNeutral,
    GrossExposure,
    LongOnly,
    BoxConstraint,
    # Risk constraints
    MaxVariance,
    MaxVolatility,
    # Factor constraints
    BetaNeutral,
    TargetBeta,
    BetaBounds,
    # Sector constraints
    SectorNeutral,
    SectorBounds,
    # Turnover constraints
    MaxTurnover,
    # Cardinality
    MaxPositions,
    # Custom
    LinearConstraint,
)

# Covariance Matrix Functions
from nt_quant.covariance import (
    build_covariance_matrix,
    get_factor_loadings_matrix,
    get_factor_covariance_matrix,
    get_idio_vol_matrix,
)

# Optimizer
from nt_quant.optimizer import (
    PortfolioOptimizer,
    RobustOptimizer,
    OptimizationResult,
)

# Backtester
from nt_quant.backtester import (
    MVOBacktester,
    BacktestConfig,
    BacktestResult,
)


__all__ = [
    # Models
    "Alphas",
    "Betas",
    "AssetCharacteristics",
    "BacktestResults",
    "Calendar",
    "FactorCovariances",
    "FactorLoadings",
    "IdioVol",
    "Weights",
    # Objectives
    "ObjectiveFunction",
    "MaximizeAlpha",
    "MinimizeVariance",
    "MeanVariance",
    "MaximizeSharpe",
    "RiskParity",
    "MaximizeInfoRatio",
    # Constraints
    "Constraint",
    "FullyInvested",
    "DollarNeutral",
    "GrossExposure",
    "LongOnly",
    "BoxConstraint",
    "MaxVariance",
    "MaxVolatility",
    "BetaNeutral",
    "TargetBeta",
    "BetaBounds",
    "SectorNeutral",
    "SectorBounds",
    "MaxTurnover",
    "MaxPositions",
    "LinearConstraint",
    # Covariance
    "build_covariance_matrix",
    "get_factor_loadings_matrix",
    "get_factor_covariance_matrix",
    "get_idio_vol_matrix",
    # Optimizer
    "PortfolioOptimizer",
    "RobustOptimizer",
    "OptimizationResult",
    # Backtester
    "MVOBacktester",
    "BacktestConfig",
    "BacktestResult",
]
