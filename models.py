"""
Models implementations
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureDropper(BaseEstimator, TransformerMixin):
    """
    Preprocessor to drop selected columns
    Use case: after PolynomialFeatures, drop columns to force coefficient to 0
    """

    def __init__(self, features_to_drop):
        self.features_to_drop = features_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Assumes X is Pandas dataframe"""
        return X.drop(columns=self.features_to_drop)


def generate_model(degree=2, split_quartic=False, take_log=False):
    '''
    Generates a sklearn pipeline representing the model with the given specifications.

    degree: degree of polynomial expansion
    split_quartic: if true and degree==4, fits "a sl + b s^2 l + c sl^2 + d s^2 l^2" (s=vol, l=leverage)
    take_log: if true, transforms target by log(1+x) before fitting and e^x - 1 after predicting
    '''
    pipeline = []

    if degree >= 2:
        pipeline.append(("poly", PolynomialFeatures(degree=degree)))

    if degree == 4 and split_quartic:
        pipeline.append(
            (
                "dropper",
                FeatureDropper(
                    features_to_drop=[
                        "1",
                        "leverage",
                        "sigma",
                        "leverage^2",
                        # "leverage sigma",
                        "sigma^2",
                        "leverage^3",
                        # "leverage^2 sigma",
                        # "leverage sigma^2",
                        "sigma^3",
                        "leverage^4",
                        "leverage^3 sigma",
                        # "leverage^2 sigma^2",
                        "leverage sigma^3",
                        "sigma^4",
                    ]
                ),
            )
        )

    if take_log:
        pipeline.append(
            (
                "log_linear",
                TransformedTargetRegressor(
                    regressor=LinearRegression(fit_intercept=False),
                    func=np.log1p,
                    inverse_func=np.expm1,
                ),
            )
        )
    else:
        pipeline.append(("linear", LinearRegression(fit_intercept=False)))

    return Pipeline(pipeline)
