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
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score, KFold

from prepare import load_data, evaluate, METRIC

t_start = time.time()

# ---------------------------------------------------------------------------
# Load raw data
# ---------------------------------------------------------------------------

X_train, X_test, y_train, y_test = load_data()

# ---------------------------------------------------------------------------
# Outlier removal (training only) — remove both types of outliers
# ---------------------------------------------------------------------------

outlier_mask = ~(
    ((X_train["GrLivArea"] > 4000) & (y_train < 200000)) |
    (X_train["LotArea"] > 100000)
)
X_train = X_train[outlier_mask].reset_index(drop=True)
y_train = y_train[outlier_mask].reset_index(drop=True)

y_train_log = np.log1p(y_train)

# ---------------------------------------------------------------------------
# NA imputation — treat structural-absence NAs as "None"/0 before anything else
# ---------------------------------------------------------------------------

# Columns where NA means "no feature" (not missing data)
none_str_cols = [
    "PoolQC", "Alley", "Fence", "MiscFeature",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "FireplaceQu", "MasVnrType",
]
none_num_cols = [
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath",
    "GarageArea", "GarageCars", "GarageYrBlt",
    "MasVnrArea",
]

for df in [X_train, X_test]:
    for col in none_str_cols:
        if col in df.columns:
            df[col] = df[col].fillna("None")
    for col in none_num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

# LotFrontage: impute by Neighborhood median (better than global median)
for df in [X_train, X_test]:
    if "LotFrontage" in df.columns:
        df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(x.median())
        )

# Drop near-zero-variance features
drop_cols = ["Street", "Utilities", "GarageYrBlt"]
for df in [X_train, X_test]:
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# ---------------------------------------------------------------------------
# Target encoding for Neighborhood
# ---------------------------------------------------------------------------

def target_encode_oof(train_col, train_target, test_col, n_splits=5, smoothing=10):
    global_mean = train_target.mean()
    train_enc = np.zeros(len(train_col))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for _, (tr_idx, val_idx) in enumerate(kf.split(train_col)):
        fold_map = train_target.iloc[tr_idx].groupby(train_col.iloc[tr_idx]).agg(["mean", "count"])
        fold_map["smooth"] = (
            (fold_map["mean"] * fold_map["count"] + global_mean * smoothing)
            / (fold_map["count"] + smoothing)
        )
        train_enc[val_idx] = train_col.iloc[val_idx].map(fold_map["smooth"]).fillna(global_mean).values
    full_map = train_target.groupby(train_col).agg(["mean", "count"])
    full_map["smooth"] = (
        (full_map["mean"] * full_map["count"] + global_mean * smoothing)
        / (full_map["count"] + smoothing)
    )
    test_enc = test_col.map(full_map["smooth"]).fillna(global_mean).values
    return train_enc, test_enc

X_train = X_train.copy()
X_test  = X_test.copy()
X_train["Neighborhood_enc"], X_test["Neighborhood_enc"] = target_encode_oof(
    X_train["Neighborhood"], y_train_log, X_test["Neighborhood"]
)

# ---------------------------------------------------------------------------
# Ordinal encode quality/condition columns
# ---------------------------------------------------------------------------

QUAL_MAP = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}

qual_cols = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond",
    "HeatingQC", "KitchenQual", "FireplaceQu",
    "GarageQual", "GarageCond", "PoolQC",
]
bsmt_exp_map = {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0}

for col in qual_cols:
    for df in [X_train, X_test]:
        if col in df.columns:
            df[col] = df[col].map(QUAL_MAP).fillna(0)

for df in [X_train, X_test]:
    if "BsmtExposure" in df.columns:
        df["BsmtExposure"] = df["BsmtExposure"].map(bsmt_exp_map).fillna(0)

# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def add_features(df):
    df = df.copy()
    # Age
    df["house_age"]      = df["YrSold"] - df["YearBuilt"]
    df["remodel_age"]    = df["YrSold"] - df["YearRemodAdd"]

    # Area — FinishedSF is more informative than TotalSF
    bsmt_sf = df["TotalBsmtSF"]
    df["total_sf"]       = bsmt_sf + df["1stFlrSF"] + df["2ndFlrSF"]
    df["finished_sf"]    = df["total_sf"] - df["BsmtUnfSF"]

    # Bathrooms
    df["total_baths"]    = (df["FullBath"] + 0.5 * df["HalfBath"]
                            + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"])

    # Quality interactions
    df["qual_sf"]        = df["OverallQual"] * df["GrLivArea"]
    df["qual_total_sf"]  = df["OverallQual"] * df["total_sf"]
    df["qual2"]          = df["OverallQual"] ** 2
    df["age_qual"]       = df["house_age"] * df["OverallQual"]

    # Ratio features
    df["liv_lot_ratio"]  = df["GrLivArea"] / (df["LotArea"].clip(lower=1))
    df["sf_per_room"]    = df["GrLivArea"] / (df["TotRmsAbvGrd"].clip(lower=1))

    # Ordinal quality interactions
    df["kitchen_qual_sf"] = df["KitchenQual"] * df["GrLivArea"]
    df["exter_qual_sf"]   = df["ExterQual"] * df["GrLivArea"]
    df["bsmt_qual_sf"]    = df["BsmtQual"] * bsmt_sf

    # Porch
    porch = sum(df.get(c, pd.Series(0, index=df.index)) for c in
                ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"])
    df["total_porch"]    = porch

    return df

X_train = add_features(X_train)
X_test  = add_features(X_test)

num_cols = X_train.select_dtypes(include="number").columns.tolist()
cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()

# ---------------------------------------------------------------------------
# Preprocessing — RobustScaler for linear model
# ---------------------------------------------------------------------------

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  RobustScaler()),
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols),
])

# ---------------------------------------------------------------------------
# Model — Ridge alpha=10
# ---------------------------------------------------------------------------

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor",    Ridge(alpha=10.0)),
])

cv_scores = cross_val_score(
    model, X_train, y_train_log,
    cv=5, scoring="neg_root_mean_squared_error",
)
cv_rmse_log = -cv_scores.mean()

model.fit(X_train, y_train_log)

y_pred_log = model.predict(X_test)
y_pred     = np.expm1(y_pred_log)

val_rmse = evaluate(y_test, y_pred)

t_end = time.time()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

num_features_out = model.named_steps["preprocessor"].transform(X_train[:1]).shape[1]

print("---")
print(f"val_rmse:         {val_rmse:.6f}")
print(f"cv_rmse_log:      {cv_rmse_log:.6f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"num_features:     {num_features_out}")
print(f"model:            Ridge(alpha=10) + research-informed preprocessing")
print(f"train_rows:       {len(X_train)}")
print(f"test_rows:        {len(X_test)}")
