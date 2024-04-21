

### Pipelines ###
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris()
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.compose import FeatureUnion
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


# Changing the return value of all transformers to return a dataframe instead of a NumPy array
from sklearn import set_config
set_config(transform_output="pandas")

# Custom Preprocessing Function
class MovingAverage(BaseEstimator, TransformerMixin):

    def __init__(self, window=30):
        self.window = window

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.rolling(window=self.window, min_perios=1, center=False).mean()

pipeline = Pipeline(
    steps=[
        ("ma", MovingAverage(window=30)),
        ("imputer", SimpleImputer()),
        ("scaler", MinMaxScaler()),
        ("regressor", LinearRegression())
    ]
)

X, y = load_iris(return_X_y=True)
columns = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

X_train, X_test, y_train, y_test = train_test_split(X, y)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Accessing parameters of the pipeline's elements
# <estimator>__<parameter>
# Setting parameters of the pipeline's elemeents
# pipeline.set_params(pipeline__ma_window=7)

try_transforming_target = False
if try_transforming_target:
    regressor = TransformedTargetRegressor(
        regressor=LinearRegression(),
        func=np.log1p,
        inverse_func=np.expm1
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer()),
            ("scaler", MinMaxScaler()),
            ("regressor", regressor)
        ]
    )
    X, y = load_iris(return_X_y=True)
    columns = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

try_combining_features = False
if try_combining_features:
    numerical_pipeline = Pipeline(
        steps=[
            ("imputation", SimpleImputer()),
            ("scaling", MinMaxScaler())
        ]
    )

    preprocessor = (
        FeatureUnion(
            [
                ("moving_Average", MovingAverage(window=30)),
                ("numerical", numerical_pipeline)
            ]
        ),
    )

    pipeline = Pipeline(steps=["preprocessing", preprocessor])



try_choosing_columns = False
if try_choosing_columns:
    categorical_transformer = ColumnTransformer(
        transformers=[("encode", OneHotEncoder())]
    )
    categorical_transformer = ColumnTransformer(
        transformers=[("encode", OneHotEncoder(), ["col_name"])],
        remainder="passthrough"
    )
    categorical_transformer = ColumnTransformer(
        transformers=[("encode", OneHotEncoder(), ["col_name"])],
        remainder=MinMaxScaler()
    )

    pipeline = Pipeline(
        steps=[
            ("categorical", categorical_transformer, ["col_name"])
        ]
    )


    categorical_transformer = Pipeline(steps=[("encode", OneHotEncoder)])
    numerical_transformer = Pipeline(
        steps=[("imputation", SimpleImputer()),
               ("scaling", MinMaxScaler())]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numerical_transformer),
            ("categoric", categorical_transformer, ["col_name"])
        ]
    )
    pipeline =Pipeline(steps=["preprocessing", preprocessor])


try_making_pipelines = False
if try_making_pipelines:
    pipeline = Pipeline(
        steps=[("imputer", SimpleImputer()),
               ("scaler", MinMaxScaler()),
               ("regression", LinearRegression())]
    )

    # Also, we can create a pipeline with `make_pipeline`
    pipeline = make_pipeline(
        steps=[SimpleImputer(), MinMaxScaler(), LinearRegression()]
    )


    X, y = load_iris(return_X_y=True)
    columns = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
### End Pipelines0 ###

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

import sys
sys.exit(0)

### Custom Transformers ###
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

data_iris = load_iris()["data"]
columns = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
data = pd.DataFrame(data_iris, columns=columns)


class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor

    def outlier_detector(self, X, y=None):
        X = pd.Series(X).copy()
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1
        self.lower_bound.append(q1 - (self.factor * iqr))
        self.upper_bound.append(q3 + (self.factor * iqr))

    def fit(self, X, y=None):
        self.lower_bound = []
        self.upper_bound = []
        X.apply(self.outlier_detector)
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for i in range(X.shape[1]):
            x = X.iloc[:, i].copy()
            condition = (x < self.lower_bound[i]) | (x > self.upper_bound[i])
            x[condition] = np.nan
            X.iloc[:, i] = x
        return X


try_outlier_remover_dummy = False
if try_outlier_remover_dummy:
    outlier_remover = OutlierRemover()
    test = pd.DataFrame(
        {"col1": [100, 200, 300, 999], "col2": [0, 0, 1, 2], "col3": [-10, 0, 1, 2]}
    )
    outlier_remover.fit(test)
    print(outlier_remover.transform(test))
    print(outlier_remover.fit_transform(test))

try_outlier_remover_iris = False
if try_outlier_remover_iris:
    outlier_remover = OutlierRemover()
    ct = ColumnTransformer(
        transformers=[
            ["outlier_remover", OutlierRemover(), list(range(data.shape[1]))]
        ],
        remainder="passthrough",
    )
    data_without_outliers = pd.DataFrame(ct.fit_transform(data), columns=data.columns)

    outliers = data.loc[data_without_outliers.isnull().sum(axis=1) > 0, "SepalWidthCm"]
    print(outliers)

try_creating_pipeline = True
if try_creating_pipeline:
    X = data.copy()
    y = load_iris()["target"].copy()
    ct = ColumnTransformer(
        transformers=[
            ["outlier_remover", OutlierRemover(), list(range(data.shape[1]))]
        ],
        remainder="passthrough",
    )
    pipeline = Pipeline(
        steps=[
            ["outlier_removal", ct],
            ["imputer", SimpleImputer()],
            ["regressor", LogisticRegression(max_iter=1000)],
        ]
    )

    param_grid = {
        "outlier_removal__outlier_remover__factor": [0, 1, 2, 3, 4],
        "imputer__strategy": ["mean", "median", "most_frequent"],
        "regressor__C": [0.01, 0.1, 1, 10, 100],
    }
    gs = GridSearchCV(
        estimator=pipeline, param_grid=param_grid, scoring="accuracy", cv=3
    )
    gs.fit(X, y)
    print(gs.best_params_)


def outlier_removal(X, factor):
    X = pd.DataFrame(X).copy()
    for i in range(X.shape[1]):
        x = pd.Series(X.iloc[:, i]).copy()
        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (factor * iqr)
        upper_bound = q3 + (factor * iqr)
        condition = (X.iloc[:, i] < lower_bound) | (X.iloc[:, i] > upper_bound)
        X.iloc[condition, i] = np.nan
    return X


try_using_function_transformer = False
if try_using_function_transformer:
    outlier_remover = FunctionTransformer(outlier_removal, kw_args={"factor": 1.5})
    test = pd.DataFrame(
        {"col1": [100, 200, 300, 999], "col2": [0, 0, 1, 2], "col3": [-10, 0, 1, 2]}
    )

    print(outlier_remover.fit_transform(test))

try_using_function_transformer_real_data = False
if try_using_function_transformer_real_data:
    X = data.copy()
    y = load_iris()["target"].copy()
    outlier_remover = FunctionTransformer(outlier_removal)


    pipeline = Pipeline(
        steps=[
            ["outlier_removal", outlier_remover],
            ["imputer", SimpleImputer()],
            ["regressor", LogisticRegression(max_iter=1000)],
        ]
    )

    param_grid = {
        "outlier_removal__kw_args": [
            {"factor": 0},
            {"factor": 1},
            {"factor": 2},
            {"factor": 3},
            {"factor": 4},
        ],
        "imputer__strategy": ["mean", "median", "most_frequent"],
        "regressor__C": [0.01, 0.1, 1, 10, 100],
    }


    gs = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring="accuracy", cv=3)
    gs.fit(X, y)
    print(gs.best_params_)
### End Custom Transformers ###
