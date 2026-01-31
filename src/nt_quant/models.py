import dataframely as dy


# =============================================================================
# Factor Model Schemas
# =============================================================================

class FactorLoadings(dy.Schema):
    """Factor loadings (betas) for each asset on each date.

    Shape: (n_dates * n_assets * n_factors) rows
    """
    date = dy.Date()
    asset_id = dy.String()
    factor = dy.String()
    loading = dy.Float64()


class FactorCovariances(dy.Schema):
    """Factor covariance matrix entries for each date.

    Shape: (n_dates * n_factors * n_factors) rows
    """
    date = dy.Date()
    factor_1 = dy.String()
    factor_2 = dy.String()
    covariance = dy.Float64()


class IdioVol(dy.Schema):
    """Idiosyncratic volatility for each asset on each date.

    Shape: (n_dates * n_assets) rows
    """
    date = dy.Date()
    asset_id = dy.String()
    idio_vol = dy.Float64()


# =============================================================================
# Alpha / Signal Schemas
# =============================================================================

class Alphas(dy.Schema):
    """Alpha signals for each asset on each date.

    Shape: (n_dates * n_assets) rows
    """
    date = dy.Date()
    asset_id = dy.String()
    alpha = dy.Float64()


# =============================================================================
# Asset Characteristic Schemas
# =============================================================================

class AssetCharacteristics(dy.Schema):
    """Generic asset characteristics for constraints (e.g., sector, market cap).

    Shape: (n_dates * n_assets) rows
    """
    date = dy.Date()
    asset_id = dy.String()
    characteristic = dy.String()
    value = dy.Float64()


class Betas(dy.Schema):
    """Market betas for each asset on each date.

    Shape: (n_dates * n_assets) rows
    """
    date = dy.Date()
    asset_id = dy.String()
    beta = dy.Float64()


# =============================================================================
# Output Schemas
# =============================================================================

class Weights(dy.Schema):
    """Optimal portfolio weights output.

    Shape: (n_dates * n_assets) rows
    """
    date = dy.Date()
    asset_id = dy.String()
    weight = dy.Float64()


class BacktestResults(dy.Schema):
    """Extended backtest results with diagnostics.

    Shape: (n_dates * n_assets) rows
    """
    date = dy.Date()
    asset_id = dy.String()
    weight = dy.Float64()
    alpha = dy.Float64()
    solve_status = dy.String()


# =============================================================================
# Calendar Schema
# =============================================================================

class Calendar(dy.Schema):
    """Trading calendar dates.

    Shape: (n_dates) rows
    """
    date = dy.Date()
