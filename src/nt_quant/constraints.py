import cvxpy as cp
import numpy as np
import polars as pl
import dataframely as dy
import datetime as dt
from enum import Enum

class Alphas(dy.Schema):
    date = dy.Date
    asset_id = dy.String
    alpha = dy.Float64

class Betas(dy.Schema):
    date = dy.Date
    asset_id = dy.String
    beta = dy.Float64

class Weights(dy.Schema):
    date = dy.Date
    asset_id = dy.String
    weight = dy.Float64

class Calendar(dy.Schema):
    date = dy.Date

class AlphaConstructor:
    def __init__(self, alphas: dy.DataFrame[Alphas]) -> None:
        self.alphas = alphas

    def construct(self, date_: dt.date) -> dy.DataFrame[Alphas]:
        return (
            self.alphas
            .filter(pl.col('date').eq(date_))
            .sort('asset_id')
        )

class BetaConstructor:
    def __init__(self, betas: dy.DataFrame[Betas]) -> None:
        self.betas = betas

    def construct(self, date_: dt.date) -> dy.DataFrame[Betas]:
        return (
            self.alphas
            .filter(pl.col('date').eq(date_))
            .sort('asset_id')
        )

class UnitBetaConstructor:
    def __init__(self, beta_constructor: BetaConstructor) -> None:
        self.beta_constructor = beta_constructor

    def __call__(self, weights: cp.Variable, date_: dt.date) -> cp.Constraint:
        betas = self.beta_constructor.construct(date_)
        betas_np = betas['beta']

        return cp.sum(weights @ betas_np) == 1
    
class Constraint(Enum):
    UnitBeta = 'unit_beta'

CONSTRAINT_REGISTRY = {
    Constraint.UnitBeta: UnitBetaConstructor
}

def backtest(start: dt.date, end: dt.date, alphas: dy.DataFrame[Alphas], betas: dy.DataFrame[Betas], calendar: dy.DataFrame[Calendar], constraints = list[Constraint]) -> dy.DataFrame[Weights]:
    alpha_constructor = AlphaConstructor(alphas)
    beta_constructor = BetaConstructor(betas)
    dates = calendar.sort('date')['date'].to_list()

    for date_ in dates:
        _alphas = alpha_constructor.construct(date_)
        _betas = beta_constructor.construct(date_)
        _covariance_matrix = covariance_matrix_constructor.construct(date_)

        for constraint in constraints:
            