import dataframely as dy

class FactorLoadings(dy.Schema):
    date = dy.Date()
    asset_id = dy.String()
    factor = dy.String()
    loading = dy.Float64()

class FactorCovariances(dy.Schema):
    date = dy.Date()
    factor_1 = dy.String()
    factor_2 = dy.String()
    covariance = dy.Float64()

class IdioVol(dy.Schema):
    date = dy.Date()
    asset_id = dy.String()
    idio_vol = dy.Float64()

class Alphas(dy.Schema):
    date = dy.Date()
    asset_id = dy.String()
    alpha = dy.Float64()