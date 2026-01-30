import polars as pl
import dataframely as dy
import numpy as np
from nt_quant.models import FactorLoadings, FactorCovariances, IdioVol

def get_factor_loadings_matrix(
    factor_loadings: pl.DataFrame
) -> np.ndarray:
    return (
        factor_loadings
        .sort("asset_id", "factor")
        .pivot(index="asset_id", on="factor", values="loading")
        .drop("asset_id")
        .to_numpy()
    )


def get_factor_covariance_matrix(factor_covariances: pl.DataFrame) -> np.ndarray:
    return (
        factor_covariances.sort("factor_1", "factor_2")
        .pivot(index="factor_1", on="factor_2", values="covariance")
        .drop("factor_1")
        .to_numpy()
    )


def get_idio_vol_matrix(idio_vol: pl.DataFrame) -> np.ndarray:
    return np.diag(
        idio_vol
        .sort("asset_id")["idio_vol"]
        .to_numpy()
    )

def construct_covariance_matrix(
    asset_ids: list[str],
    factor_loadings: dy.DataFrame[FactorLoadings],
    factor_covariances: dy.DataFrame[FactorCovariances],
    idio_vol: dy.DataFrame[IdioVol]
) -> pl.DataFrame:
    # Filter to asset_ids
    factor_loadings = factor_loadings.filter(pl.col('asset_id').is_in(asset_ids))
    idio_vol = idio_vol.filter(pl.col('asset_id').is_in(asset_ids))

    # Construct covariance matrix components
    factor_loadings_matrix = get_factor_loadings_matrix(factor_loadings)
    factor_covariance_matrix = get_factor_covariance_matrix(factor_covariances)
    idio_vol_matrix = get_idio_vol_matrix(idio_vol)

    # Construct covariance matrix
    covariance_matrix_np = (
        factor_loadings_matrix @ factor_covariance_matrix @ factor_loadings_matrix.T
        + idio_vol_matrix**2
    )

    # Format covariance matrix
    covariance_matrix = pl.from_numpy(covariance_matrix_np)
    covariance_matrix.columns = asset_ids
    covariance_matrix = covariance_matrix.select(
        pl.Series(asset_ids).alias("asset_id"), *asset_ids
    )

    return covariance_matrix