# -*- coding: utf-8 -*-

# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from cate_encoder import CategoricalEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# prepare data
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    if not os.path.exists(csv_path):
        fetch_housing_data()
    return pd.read_csv(csv_path)


def bird_view(df):
    df.head()
    df.info()
    df.describe()
    missing_percent = (df.isnull().sum() / df.isnull().count()).sort_values(
        ascending=False)
    missing_percent


# 3  Discover and visualize the data to gain insights
# omit
def insight(housing):
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing[
        "total_rooms"]
    housing["population_per_household"] = housing["population"] / housing[
        "households"]

    corr_matrix = housing.corr()
    print(
        'correlation with y:',
        corr_matrix["median_house_value"].sort_values(ascending=False))

# Hey, not bad! The new bedrooms_per_room attribute is much more correlated with the median house value
# than the total number of rooms or bedrooms. Apparently houses with a lower bedroom/room ratio tend to
# be more expensive.


def split_train_test(housing):
    # split train/test by stratified sampling
    # make sure train/test as same distribution of a feature
    housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    # Now you should remove the income_cat attribute so the data is back to its original state:
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    return strat_train_set, strat_test_set


# To use pipeline, create a class to select numerical or categorical columns
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


# column index
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


# combine features
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


def make_pipline(X_train):
    housing_num = X_train.drop("ocean_proximity", axis=1)
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('cat_encoder', CategoricalEncoder(encoding="onehot-dense")),
    ])

    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])
    return full_pipeline


# 5.1  Select a Performance Measure
def RMSE(y, y_pred):
    mse = mean_squared_error(y, y_pred)
    return np.sqrt(mse)


# 6  Fine-tune your model
def rmse_cv(model, X, y):
    scores = cross_val_score(
        model, X, y, scoring="neg_mean_squared_error", cv=10)
    return np.sqrt(-scores)


def validate(model, X_train, y_train):
    scores = rmse_cv(model, X_train, y_train)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# Evaluate Your System on the Test Set
def eval_test(model, test_set, full_pipeline):
    X_test = test_set.drop("median_house_value", axis=1)
    y_test = test_set["median_house_value"].copy()
    X_test_prepared = full_pipeline.transform(X_test)
    y_pred = model.predict(X_test_prepared)
    return RMSE(y_test, y_pred)


def RF_grid_search(X_train, y_train):
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestRegressor
    forest_reg = RandomForestRegressor(random_state=42)
    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {
            'n_estimators': [3, 10, 30],
            'max_features': [2, 4, 6, 8]
        },
        # then try 6 (2×3) combinations with bootstrap set as False
        {
            'bootstrap': [False],
            'n_estimators': [3, 10],
            'max_features': [2, 3, 4]
        },
    ]
    grid_search = GridSearchCV(
        forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    return grid_search
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training


def main():
    housing = load_housing_data()
    # bird_view(housing)
    strat_train_set, strat_test_set = split_train_test(housing)

    X_train = strat_train_set.drop(
        "median_house_value", axis=1)  # drop labels for training set
    y_train = strat_train_set["median_house_value"].copy()

    full_pipeline = make_pipline(X_train)
    X_train_prep = full_pipeline.fit_transform(X_train)

    # Select and train a model
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_prep, y_train)
    validate(lin_reg, X_train_prep, y_train)

    print('test_score:', eval_test(lin_reg, strat_test_set, full_pipeline))

    # grid_search = RF_grid_search(X_train_prep, y_train)
    # final_model = grid_search.best_estimator_


if __name__ == "__main__":
    main()
