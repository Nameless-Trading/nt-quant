import datetime as dt

import numpy as np
import polars as pl
import dataframely as dy

from nt_quant.models import FactorLoadings, FactorCovariances, IdioVol


def get_factor_loadings_matrix(
    asset_ids: list[str], factor_loadings: pl.DataFrame
) -> np.ndarray:
    """Pivot factor loadings into matrix form (n_assets x n_factors)."""
    return (
        factor_loadings.filter(pl.col("asset_id").is_in(asset_ids))
        .sort("asset_id", "factor")
        .pivot(index="asset_id", on="factor", values="loading")
        .drop("asset_id")
        .to_numpy()
    )


def get_factor_covariance_matrix(factor_covariances: pl.DataFrame) -> np.ndarray:
    """Pivot factor covariances into matrix form (n_factors x n_factors)."""
    return (
        factor_covariances.sort("factor_1", "factor_2")
        .pivot(index="factor_1", on="factor_2", values="covariance")
        .drop("factor_1")
        .to_numpy()
    )


def get_idio_vol_matrix(asset_ids: list[str], idio_vol: pl.DataFrame) -> np.ndarray:
    """Build diagonal idiosyncratic volatility matrix (n_assets x n_assets)."""
    return np.diag(
        idio_vol.filter(pl.col("asset_id").is_in(asset_ids))
        .sort("asset_id")["idio_vol"]
        .to_numpy()
    )


def build_covariance_matrix(
    asset_ids: list[str],
    date: dt.date,
    factor_loadings: dy.DataFrame[FactorLoadings],
    factor_covariances: dy.DataFrame[FactorCovariances],
    idio_vol: dy.DataFrame[IdioVol],
) -> np.ndarray:
    """Build covariance matrix from factor model components.

    Covariance = B @ F @ B.T + D^2

    Where:
        B = factor loadings matrix (n_assets, n_factors)
        F = factor covariance matrix (n_factors, n_factors)
        D = diagonal idiosyncratic volatility matrix (n_assets, n_assets)

    Args:
        asset_ids: List of asset IDs (defines ordering of matrix)
        date: The date for which to build the covariance matrix
        factor_loadings: Factor loadings DataFrame
        factor_covariances: Factor covariances DataFrame
        idio_vol: Idiosyncratic volatility DataFrame

    Returns:
        Covariance matrix as numpy array (n_assets, n_assets)
    """
    # Filter data for the given date
    loadings_df = factor_loadings.filter(pl.col("date") == date)
    cov_df = factor_covariances.filter(pl.col("date") == date)
    idio_df = idio_vol.filter(pl.col("date") == date)

    # Build component matrices
    B = get_factor_loadings_matrix(asset_ids, loadings_df)
    F = get_factor_covariance_matrix(cov_df)
    D = get_idio_vol_matrix(asset_ids, idio_df)

    # Compute full covariance matrix: B @ F @ B.T + D^2
    return B @ F @ B.T + D**2
