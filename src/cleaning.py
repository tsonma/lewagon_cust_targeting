from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from imblearn.pipeline import Pipeline  # <- changed to imblearn
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd

class CyclicalMonthEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column_name='month'):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        month_map = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
            'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
            'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        months = X[self.column_name].str.lower().str[:3].map(month_map)
        sin_month = np.sin(2 * np.pi * months / 12)
        cos_month = np.cos(2 * np.pi * months / 12)
        return np.c_[sin_month, cos_month]

    def get_feature_names_out(self, input_features=None):
        return [f"{self.column_name}_sin", f"{self.column_name}_cos"]


def build_preprocessor(X_train):
    categorical_features = ['contact','default','education','housing','job','loan','marital','poutcome']

    categorical_transformer = OneHotEncoder(
        drop='if_binary',
        sparse_output=False,
        handle_unknown='ignore'
    )

    all_numerical = make_column_selector(dtype_include='int64')(X_train)
    numerical_features = [col for col in all_numerical if col != 'month']

    num_transformer = Pipeline([
        ('robust_scaler', RobustScaler())
    ])

    month_transformer = CyclicalMonthEncoder(column_name='month')

    preprocessor = ColumnTransformer([
        ('categorical_features', categorical_transformer, categorical_features),
        ('num_transformer', num_transformer, numerical_features),
        ('month_transformer', month_transformer, ['month'])
    ], remainder='passthrough')

    return preprocessor


def build_pipeline_with_smote(X_train):
    preprocessor = build_preprocessor(X_train)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42))  # SMOTE step
    ])
    return pipeline


def fit_and_return_preprocessed_df(preprocessor, X, y=None):
    # If SMOTE is in the pipeline, y is required
    if y is not None:
        X_transformed, y_resampled = preprocessor.fit_resample(X, y)
        feature_names = preprocessor.named_steps['preprocessor'].get_feature_names_out()
        X_df = pd.DataFrame(X_transformed, columns=feature_names)
        return X_df, y_resampled
    else:
        X_transformed = preprocessor.fit_transform(X)
        feature_names = preprocessor.get_feature_names_out()
        X_df = pd.DataFrame(X_transformed, columns=feature_names)
        return X_df

# --- DELETED CODE ---
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.compose import ColumnTransformer, make_column_selector
# from sklearn.preprocessing import OneHotEncoder, RobustScaler
# from sklearn.pipeline import Pipeline
# import numpy as np
# import pandas as pd

# class CyclicalMonthEncoder(BaseEstimator, TransformerMixin):
#     def __init__(self, column_name='month'):
#         self.column_name = column_name

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         month_map = {
#             'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
#             'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
#             'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
#         }

#         months = X[self.column_name].str.lower().str[:3].map(month_map)
#         sin_month = np.sin(2 * np.pi * months / 12)
#         cos_month = np.cos(2 * np.pi * months / 12)
#         return np.c_[sin_month, cos_month]

#     def get_feature_names_out(self, input_features=None):
#         return [f"{self.column_name}_sin", f"{self.column_name}_cos"]


# def build_preprocessor(X_train):
#     """
#     Build and return a preprocessor ColumnTransformer based on the training data X_train.

#     Args:
#         X_train (pd.DataFrame): Training data used to identify features.

#     Returns:
#         preprocessor (ColumnTransformer): The composed preprocessor pipeline.
#     """
#     categorical_features = ['contact','default','education','housing','job','loan','marital','poutcome']

#     categorical_transformer = OneHotEncoder(
#         drop='if_binary',
#         sparse_output=False,
#         handle_unknown='ignore'
#     )

#     all_numerical = make_column_selector(dtype_include='int64')(X_train)
#     #change the numerical features to run without a for loop
#     numerical_features = [col for col in all_numerical if col != 'month']

#     num_transformer = Pipeline([
#         ('robust_scaler', RobustScaler())
#     ])

#     month_transformer = CyclicalMonthEncoder(column_name='month')

#     preprocessor = ColumnTransformer([
#         ('categorical_features', categorical_transformer, categorical_features),
#         ('num_transformer', num_transformer, numerical_features),
#         ('month_transformer', month_transformer, ['month'])
#     ], remainder='passthrough')


#     return preprocessor

# def fit_and_return_preprocessed_df(preprocessor, X):
#     """
#     Fit the preprocessor on X and return a preprocessed DataFrame.

#     Args:
#         preprocessor (ColumnTransformer): Unfitted preprocessor.
#         X (pd.DataFrame): Data to fit and transform.

#     Returns:
#         X_df (pd.DataFrame): Transformed DataFrame with feature names.
#     """


#     X_transformed = preprocessor.fit_transform(X)
#     feature_names = preprocessor.get_feature_names_out()
#     X_df = pd.DataFrame(X_transformed, columns=feature_names)

#     return X_df
