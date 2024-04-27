### Custom Estimator ###
from sklearn.base import BaseEstimator, clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import VotingClassifier
from imblearn.pipeline import make_pipeline as make_imb_pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report, RocCurveDisplay, ConfusionMatrixDisplay
import matplotlib.pyplot as plt



class ResampledEnsemble(BaseEstimator):
    def __init__(self, base_estimator=DecisionTreeClassifier(),
                 n_estimators=100, max_depth=None, max_features=None, 
                 min_samples_split=2, min_samples_leaf=1):
        self._estimator_type = "classifier"
        self.base_etimator = base_estimator
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.estimators = self._generate_estimators()

        self.estimator = VotingClassifier(self.estimators, voting="soft")
        
    def _generate_estimators(self):
        estimators = []
        for i in range(self.n_estimators):
            est = clone(self.base_etimator)
            est.random_state = i 
            est.max_depth = self.max_depth
            est.max_features = self.max_features
            est.min_saples_split = self.min_samples_split
            est.min_samples_leaf = self.min_samples_leaf
            pipe = make_imb_pipeline(
                RandomUnderSampler(random_state=i, replacement=True), 
                est
            )
        return estimators
    
    def fit(self, X, y, sample_weight=None):
        return self.estimator.fit(X, y, sample_weight)
    
    def predict(self, X):
        return self.estimator.predict(X)
    
    def classes_(self):
        if self.estimator:
            return self.estimator.classes_
        
    def set_params(self, **params):
        if not params:
            return self 

        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value
        self.estimators = self._generate_estimators()
        self.estimator = VotingClassifier(self.estimators, voting="soft")
        return self 
    


def set_params(self, **params):
    if not params:
        return self

    for key, value in params.items():
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.kwargs[key] = value



data = load_breast_cancer(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(data.data, data.train, 
                                                    random_state=0)
res_ensemb = ResampledEnsemble()
res_ensemb.fit(X_train, y_train)
y_pred = res_ensemb.predict(X_test)
print(
    classification_report(y_test, y_pred)
)


fig, ax = plt.subplots()
RocCurveDisplay(res_ensemb, X_test, y_test, ax=ax)
fig.show()

fig, ax = plt.subplots()
ConfusionMatrixDisplay(res_ensemb, X_test, y_test, display_labels=[0, 1, 2],
                       cmap = plt.cm.GnBu, normalize=None, ax=ax)
fig.show()

fig, ax = plt.subplots()
ConfusionMatrixDisplay(res_ensemb, X_test, y_test, display_labels=[0, 1, 2],
                       cmap=plt.cm.GnBu, normalize="true", ax=ax
)
fig.show()

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import GridSearchCV
pipe = make_pipeline(SimpleImputer(missing_values=np.nan, strategy="mean"),
                     MinMaxScaler(), 
                     ResampledEnsemble(
                         max_features="auto", min_samples_split=0.01,
                         min_samples_leaf=0.0001, n_estimators=300
                         ),                                     
                    )
grid_params = {
    "resampledensemble__max_depth": np.linspace(5, 40, 3, 
                                                endpoint=True, dtype=int)
}
grid = GridSearchCV(pipe, grid_params, cv=4, return_train_score=True,
                    n_jobs=-1, scoring="f1_macro")
grid.fit(X_train, y_train)
best_score = grid.best_score_
best_params = grid.best_params_
print(best_score)
print(best_params)

### End Custom Estimator ###

### Pipelines ###
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


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
        return X.rolling(window=self.window, min_periods=1, center=False).mean()


"""
pipeline = Pipeline(
    steps=[
        ("ma", MovingAverage(window=30)),
        ("imputer", SimpleImputer()),
        ("scaler", MinMaxScaler()),
        ("regressor", LinearRegression())
    ]
)

data_iris = load_iris()
columns = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

df = pd.DataFrame(data_iris["data"], columns=columns)
pipeline.fit(df[columns], data_iris.target)
y_pred = pipeline.predict(df[columns])

# Accessing parameters of the pipeline's elements
# <estimator>__<parameter>
# Setting parameters of the pipeline's elemeents
# pipeline.set_params(pipeline__ma_window=7)

"""
try_transforming_target = False
if try_transforming_target:
    regressor = TransformedTargetRegressor(
        regressor=LinearRegression(), func=np.log1p, inverse_func=np.expm1
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer()),
            ("scaler", MinMaxScaler()),
            ("regressor", regressor),
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
        steps=[("imputation", SimpleImputer()), ("scaling", MinMaxScaler())]
    )

    preprocessor = (
        FeatureUnion(
            [
                ("moving_Average", MovingAverage(window=30)),
                ("numerical", numerical_pipeline),
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
        remainder="passthrough",
    )
    categorical_transformer = ColumnTransformer(
        transformers=[("encode", OneHotEncoder(), ["col_name"])],
        remainder=MinMaxScaler(),
    )

    pipeline = Pipeline(steps=[("categorical", categorical_transformer, ["col_name"])])

    categorical_transformer = Pipeline(steps=[("encode", OneHotEncoder)])
    numerical_transformer = Pipeline(
        steps=[("imputation", SimpleImputer()), ("scaling", MinMaxScaler())]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numerical_transformer),
            ("categoric", categorical_transformer, ["col_name"]),
        ]
    )
    pipeline = Pipeline(steps=["preprocessing", preprocessor])


try_making_pipelines = False
if try_making_pipelines:
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer()),
            ("scaler", MinMaxScaler()),
            ("regression", LinearRegression()),
        ]
    )

    # Also, we can create a pipeline with `make_pipeline`
    pipeline = make_pipeline(*[SimpleImputer(), MinMaxScaler(), LinearRegression()])

    X, y = load_iris(return_X_y=True)
    columns = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
### End Pipelines ###


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

    gs = GridSearchCV(
        estimator=pipeline, param_grid=param_grid, scoring="accuracy", cv=3
    )
    gs.fit(X, y)
    print(gs.best_params_)
### End Custom Transformers ###
