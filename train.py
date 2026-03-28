"""
Autoresearch classic ML experiment script.
This is the ONLY file the agent edits.

Usage: uv run train.py

Everything is fair game:
  - Feature engineering (imputation, encoding, transforms, new features)
  - Model selection (Ridge, RandomForest, XGBoost, LightGBM, stacking, ...)
  - Hyperparameter tuning (manual, GridSearch, Optuna, ...)
  - Cross-validation strategy

The only constraint: the script must run without crashing and call
evaluate(y_test, y_pred) using the fixed harness from prepare.py.
"""

import time
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV

from prepare import load_data, evaluate, METRIC

t_start = time.time()

# ---------------------------------------------------------------------------
# Load raw data
# ---------------------------------------------------------------------------

X_train, X_test, y_train, y_test = load_data()

# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def add_features(df):
    df = df.copy()
    df["house_age"]   = df["YrSold"] - df["YearBuilt"]
    df["remodel_age"] = df["YrSold"] - df["YearRemodAdd"]
    df["total_sf"]    = df["TotalBsmtSF"].fillna(0) + df["1stFlrSF"] + df["2ndFlrSF"]
    df["total_baths"] = (df["FullBath"] + 0.5 * df["HalfBath"]
                         + df.get("BsmtFullBath", pd.Series(0, index=df.index)).fillna(0)
                         + 0.5 * df.get("BsmtHalfBath", pd.Series(0, index=df.index)).fillna(0))
    df["qual_sf"]     = df["OverallQual"] * df["GrLivArea"]
    df["qual_overall"]= df["OverallQual"] * df["OverallCond"]
    df["total_porch"] = (df.get("OpenPorchSF", 0) + df.get("EnclosedPorch", 0)
                         + df.get("3SsnPorch", 0) + df.get("ScreenPorch", 0))
    df["has_garage"]  = (df.get("GarageArea", pd.Series(0, index=df.index)).fillna(0) > 0).astype(int)
    df["has_bsmt"]    = (df.get("TotalBsmtSF", pd.Series(0, index=df.index)).fillna(0) > 0).astype(int)
    df["has_pool"]    = (df.get("PoolArea", pd.Series(0, index=df.index)).fillna(0) > 0).astype(int)
    return df

X_train = add_features(X_train)
X_test  = add_features(X_test)

# Log-transform target
y_train_log = np.log1p(y_train)

# Identify column types
num_cols = X_train.select_dtypes(include="number").columns.tolist()
cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()

# Numeric: median imputation + standard scaling
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

# Categorical: constant imputation + one-hot encoding
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols),
])

# ---------------------------------------------------------------------------
# Model — Ridge with alpha grid search
# ---------------------------------------------------------------------------

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor",    Ridge()),
])

param_grid = {"regressor__alpha": [0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]}
grid_search = GridSearchCV(
    model, param_grid, cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
)

# ---------------------------------------------------------------------------
# Cross-validation + fit
# ---------------------------------------------------------------------------

grid_search.fit(X_train, y_train_log)
best_model = grid_search.best_estimator_
best_alpha = grid_search.best_params_["regressor__alpha"]
cv_rmse_log = -grid_search.best_score_

y_pred_log = best_model.predict(X_test)
y_pred     = np.expm1(y_pred_log)

val_rmse = evaluate(y_test, y_pred)

t_end = time.time()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

num_features_out = best_model.named_steps["preprocessor"].transform(X_train[:1]).shape[1]

print("---")
print(f"val_rmse:         {val_rmse:.6f}")
print(f"cv_rmse_log:      {cv_rmse_log:.6f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"num_features:     {num_features_out}")
print(f"model:            Ridge(alpha={best_alpha})")
print(f"train_rows:       {len(X_train)}")
print(f"test_rows:        {len(X_test)}")
