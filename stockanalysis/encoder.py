from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import pytz


class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """
        Extracts the day of week (dow), the hour, the month and the year from a time column.
        Returns a copy of the DataFrame X with only four columns: 'dow', 'hour', 'month', 'year'.
    """
    def __init__(self, time_column, time_zone_name='America/New_York'):
        self.time_column = time_column
        self.time_zone_name = time_zone_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X_ = X.copy()
        X_.index = pd.to_datetime(X[self.time_column])
        X_.index = X_.index.tz_localize(pytz.timezone(self.time_zone_name))
        X_.index = X_.index.tz_convert(self.time_zone_name)
        X_["weekday"] = X_.index.weekday
        X_["hour"] = X_.index.hour
        X_["month"] = X_.index.month
        X_["year"] = X_.index.year
        return X_[['weekday','month', 'year']]
