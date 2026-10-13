#!/usr/bin/env python3
"""
Class C Dx Cost Estimator - modeling pipeline (background-safe)

Implements the architecture document:
  1 load + validate  2 frozen split  3 features (20 inputs only)  4 model zoo (OOF)
  5 blend (hill-climb / stacking)  6 calibration  7 report + log  8 final holdout (--final)
  + optional ablation of inputs that are NOT in the production contract.

Usage
  pip install pandas numpy scikit-learn lightgbm xgboost catboost
  python cost_estimator_pipeline.py --data nc_cleaned_training_dataset_10499_rows_49_cols.csv

  --diagnostics         profile the data (zeros, duplicates / noise ceiling, one-hot checks) and exit
  --ablation            also test A0..A4 input variants (needs number_of_cus / rate_category / complexity cols)
  --final               score the untouched holdout ONCE (refuses a 2nd time unless --force-final)
  --budget-min 120      stop starting new models after this many minutes (blends what finished)
  --models lgbm,xgb     run a subset;  --seeds 42,7,2024  repeated CV for a stable estimate
  --split-mode          random | workorder_time | ids   (ids needs --holdout-ids file.csv)
  --fast                tiny models, for a smoke test only
  --no-cache            retrain everything (default reuses finished models for identical data/split)

Everything is written to ./outputs (override with --out):
  split.json, logs/run_<id>.log, run_history.csv, oof_predictions.csv, blend.json,
  report_by_type.csv, ablation.csv, diagnostics.txt, cache/, holdout_predictions.csv (--final)
"""
import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
SCRIPT_VERSION = "1.0"

# =============================================================================
# CONFIG (edit here or override on the command line)
# =============================================================================
CFG = dict(
    data_path="nc_cleaned_training_dataset_10499_rows_49_cols.csv",
    out_dir="outputs",
    target="adjusted_net_total",
    id_col="work_order_number",              # never a feature; used only for splits / reporting
    project_cost_col="project_cost_cleaned",  # TRAINING TARGET of the component model only
    investment_col="investment_dollars_cleaned",
    split_mode="random",                     # random | workorder_time | ids
    holdout_ids_file=None,
    holdout_frac=0.15,
    n_folds=5,
    seeds=[42],
    time_budget_min=120,
    n_jobs=-1,
    models=None,                             # None = every available model
    # ---- business KPI (agreed accuracy draft) ----
    hit_tol=0.50,        # within +/-50% of Class A
    zero_tol=500.0,      # zero actual: hit when absolute error < $500
    under_cut=0.10,      # secondary KPI: positives predicted more than 10% under
    under_penalty=0.25,  # objective = hit_rate - penalty * under_rate
    # ---- calibration ----
    min_cell=150,
    cal_grid=(0.85, 1.30, 0.01),
    cal_min_gain=0.002,  # cross-fitted objective gain needed to accept calibration
    # ---- specialists ----
    specialist_min_rows=150,
    fast=False,
)

# the 20 Power App inputs (canonical names; matching ignores spaces / dash style / case)
INPUTS = [
    "AWF", "exp_length_oh", "exp_length_ug", "exp_length_su",
    "num_of_work_locations",
    "amps_group_200A - 400A", "amps_group_Greater Than 400A", "amps_group_Less Than 200A",
    "is_overhead", "is_underground", "far_or_same_side", "new_or_existing_transformer",
    "number_of_poles_installed", "number_of_poles_removed",
    "sub_work_type_LE", "sub_work_type_PRI", "sub_work_type_SEC",
    "rate_category_group_Residential",
    "con_obj_z_phase_Single", "con_obj_z_phase_Three",
]
SUB = ["sub_work_type_LE", "sub_work_type_PRI", "sub_work_type_SEC"]
SUB_NAMES = ["LE", "PRI", "SEC"]
AMPS = ["amps_group_Less Than 200A", "amps_group_200A - 400A", "amps_group_Greater Than 400A"]
PHASE = ["con_obj_z_phase_Single", "con_obj_z_phase_Three"]

# columns NOT in the production contract; only used by --ablation to size their value
ABLATION_COLS = {
    "customers": ["number_of_cus"],
    "rate_category": ["rate_category"],
    "complexity": ["complexity_category", "complexity", "complexity_cat"],
}

MODEL_ORDER = ["ridge_poly", "knn", "hgb", "et", "rf", "lgbm", "lgbm_huber", "xgb", "xgb_tweedie",
               "hurdle_soft", "hurdle_hard", "spec_lgbm", "component", "svr", "mlp", "cat", "cat_mae"]
MODEL_VER = "m1"  # bump when model definitions change (invalidates cache)

LOG = logging.getLogger("pipe")


# =============================================================================
# logging / small helpers
# =============================================================================
def setup_logging(out_dir, run_id):
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(os.path.join(out_dir, "logs", f"run_{run_id}.log"), encoding="utf-8")):
        h.setFormatter(fmt)
        LOG.addHandler(h)


def inv(z):
    return np.clip(np.expm1(np.asarray(z, float)), 0, None)


def _norm(s):
    return re.sub(r"[\s\u2010-\u2015\u2212\-]+", "", str(s)).lower()


def clean_name(s):
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(s))


def data_hash(df, cols):
    h = pd.util.hash_pandas_object(df[cols], index=False).values.tobytes()
    return hashlib.sha1(h).hexdigest()[:12]


# =============================================================================
# 1. LOAD + VALIDATE
# =============================================================================
def load_and_validate(path):
    if not os.path.exists(path):
        raise SystemExit(f"Data file not found: {path}")
    raw = pd.read_csv(path, low_memory=False)
    LOG.info("loaded %s rows x %s cols from %s", *raw.shape, path)

    lookup = {}
    for c in raw.columns:
        lookup.setdefault(_norm(c), c)

    rename, missing = {}, []
    for want in INPUTS + [CFG["target"]]:
        got = lookup.get(_norm(want))
        if got is None:
            missing.append(want)
        else:
            rename[got] = want
    if missing:
        raise SystemExit("Missing required columns (check names in INPUTS / CFG['target']): %s\n"
                         "Columns present: %s" % (missing, list(raw.columns)))
    df = raw.rename(columns=rename).reset_index(drop=True)

    # resolve optional columns to the name that exists in df
    for key in ("id_col", "project_cost_col", "investment_col"):
        got = lookup.get(_norm(CFG[key]))
        CFG[key + "_resolved"] = rename.get(got, got) if got else None
        if got is None:
            LOG.info("optional column '%s' not found (%s)", CFG[key], key)

    # numeric coercion
    for c in INPUTS + [CFG["target"]]:
        before = df[c].isna().sum()
        df[c] = pd.to_numeric(df[c], errors="coerce")
        bad = df[c].isna().sum() - before
        if bad:
            LOG.info("WARNING %s: %d non-numeric values -> NaN", c, bad)
    nan_in = int(df[INPUTS].isna().sum().sum())
    if nan_in:
        LOG.info("WARNING %d NaN in inputs; filled with 0", nan_in)
        df[INPUTS] = df[INPUTS].fillna(0)
    if df[CFG["target"]].isna().any():
        n = int(df[CFG["target"]].isna().sum())
        LOG.info("WARNING dropping %d rows with missing target", n)
        df = df[df[CFG["target"]].notna()].reset_index(drop=True)
    neg = int((df[CFG["target"]] < 0).sum())
    if neg:
        LOG.info("WARNING %d negative targets clipped to 0", neg)
        df[CFG["target"]] = df[CFG["target"]].clip(lower=0)

    # consistency checks (reported, never silently fixed)
    for name, cols in (("sub_work_type", SUB), ("amps_group", AMPS), ("phase", PHASE)):
        bad = int((df[cols].sum(1) != 1).sum())
        LOG.info("one-hot check %-13s: %d rows do not sum to 1", name, bad)
    both = int(((df.is_overhead == 1) & (df.is_underground == 1)).sum())
    neither = int(((df.is_overhead == 0) & (df.is_underground == 0)).sum())
    LOG.info("overhead/underground: both=%d neither=%d", both, neither)
    const = [c for c in INPUTS if df[c].nunique() <= 1]
    if const:
        LOG.info("constant inputs (will be dropped from features): %s", const)

    y = df[CFG["target"]]
    LOG.info("target: zero share %.1f%%, median(y>0)=%.0f, p95=%.0f, max=%.0f",
             100 * (y == 0).mean(), y[y > 0].median(), y.quantile(0.95), y.max())
    sub = df[SUB].values.argmax(1)
    for i, nme in enumerate(SUB_NAMES):
        m = sub == i
        LOG.info("  %-3s rows=%5d zero=%5.1f%% median(y>0)=%.0f", nme, m.sum(),
                 100 * (y[m] == 0).mean() if m.any() else 0,
                 y[m & (y > 0)].median() if (m & (y > 0)).any() else 0)
    return df


# =============================================================================
# KPI (official rule) + objective
# =============================================================================
def kpi(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    pos = y > 0
    hit = np.zeros(len(y), bool)
    if pos.any():
        hit[pos] = np.abs(p[pos] - y[pos]) / y[pos] <= CFG["hit_tol"]
    if (~pos).any():
        hit[~pos] = np.abs(p[~pos] - y[~pos]) < CFG["zero_tol"]
    out = {"n": int(len(y)), "n_zero": int((~pos).sum()), "hit": float(hit.mean()) if len(y) else np.nan}
    out["hit_pos"] = float(hit[pos].mean()) if pos.any() else np.nan
    out["hit_zero"] = float(hit[~pos].mean()) if (~pos).any() else np.nan
    out["under"] = float(((p[pos] - y[pos]) / y[pos] < -CFG["under_cut"]).mean()) if pos.any() else 0.0
    out["mae"] = float(np.mean(np.abs(p - y))) if len(y) else np.nan
    return out


def objective(y, p):
    k = kpi(y, p)
    return k["hit"] - CFG["under_penalty"] * k["under"]


def kpi_by_type(sub, y, p):
    rows = {"ALL": kpi(y, p)}
    for i, nme in enumerate(SUB_NAMES):
        m = sub == i
        if m.any():
            rows[nme] = kpi(y[m], p[m])
    return pd.DataFrame(rows).T[["n", "n_zero", "hit", "hit_pos", "hit_zero", "under", "mae"]]


# =============================================================================
# 2. FROZEN SPLIT
# =============================================================================
def strat_label(df):
    y = df[CFG["target"]].values
    sub = np.array(SUB_NAMES)[df[SUB].values.argmax(1)]
    bucket = np.zeros(len(df), int)
    pos = y > 0
    if pos.sum() >= 8:
        bucket[pos] = 1 + pd.qcut(y[pos], 4, labels=False, duplicates="drop")
    lab = pd.Series(sub) + "_" + pd.Series(bucket).astype(str)
    vc = lab.value_counts()
    rare = vc[vc < 2 * CFG["n_folds"]].index
    lab = lab.where(~lab.isin(rare), "rare")
    if (lab == "rare").sum() < CFG["n_folds"]:
        lab = lab.where(lab != "rare", vc.index[0])
    return lab.values


def make_folds(lab, seed):
    fold = np.zeros(len(lab), int)
    skf = StratifiedKFold(CFG["n_folds"], shuffle=True, random_state=seed)
    for k, (_, va) in enumerate(skf.split(np.zeros(len(lab)), lab)):
        fold[va] = k
    return fold


def frozen_split(df, dhash, seeds, reset=False):
    path = os.path.join(CFG["out_dir"], "split.json")
    mode = CFG["split_mode"]
    sig = {"mode": mode, "data_hash": dhash, "holdout_frac": CFG["holdout_frac"], "n_folds": CFG["n_folds"],
           "ids_file": CFG["holdout_ids_file"]}
    store = None
    if os.path.exists(path) and not reset:
        store = json.load(open(path))
        old = {k: store.get(k) for k in sig}
        if old != sig:
            raise SystemExit("split.json was built for different data/settings.\n  saved: %s\n  now:   %s\n"
                             "Use --reset-split to rebuild (this invalidates comparisons with earlier runs)."
                             % (old, sig))
    if store is None:
        lab_all = strat_label(df)
        n = len(df)
        idx = np.arange(n)
        if mode == "random":
            tr, ho = train_test_split(idx, test_size=CFG["holdout_frac"], random_state=0, stratify=lab_all)
        elif mode == "workorder_time":
            col = CFG["id_col_resolved"]
            if col is None:
                raise SystemExit("split_mode=workorder_time needs the id column")
            order = np.argsort(pd.to_numeric(df[col], errors="coerce").fillna(-1).values, kind="stable")
            n_ho = int(round(n * CFG["holdout_frac"]))
            ho, tr = np.sort(order[-n_ho:]), np.sort(order[:-n_ho])
        elif mode == "ids":
            col = CFG["id_col_resolved"]
            if col is None or not CFG["holdout_ids_file"]:
                raise SystemExit("split_mode=ids needs the id column and --holdout-ids <csv>")
            ids = pd.read_csv(CFG["holdout_ids_file"])
            ids = ids[CFG["id_col"]] if CFG["id_col"] in ids.columns else ids.iloc[:, 0]
            ho_mask = df[col].astype(str).isin(set(ids.astype(str)))
            ho, tr = np.where(ho_mask)[0], np.where(~ho_mask)[0]
            LOG.info("holdout ids matched %d of %d ids", len(ho), len(set(ids)))
            if len(ho) == 0:
                raise SystemExit("no holdout ids matched")
        else:
            raise SystemExit(f"unknown split_mode {mode}")
        store = dict(sig, train_idx=[int(i) for i in tr], holdout_idx=[int(i) for i in ho], folds={})
        LOG.info("new frozen split (%s): train=%d holdout=%d", mode, len(tr), len(ho))
    tr = np.array(store["train_idx"])
    ho = np.array(store["holdout_idx"])
    lab_tr = strat_label(df.iloc[tr].reset_index(drop=True))
    changed = False
    for s in seeds:
        if str(s) not in store["folds"]:
            store["folds"][str(s)] = [int(v) for v in make_folds(lab_tr, s)]
            changed = True
    if changed or not os.path.exists(path):
        json.dump(store, open(path, "w"))
    folds = {s: np.array(store["folds"][str(s)]) for s in seeds}
    return tr, ho, folds


# =============================================================================
# 3. FEATURES (derived only from the 20 inputs)
# =============================================================================
def make_features(df):
    X = df[INPUTS].copy().astype(float)
    tot = X.exp_length_oh + X.exp_length_ug + X.exp_length_su
    X["total_exp_length"] = tot
    X["oh_share"] = X.exp_length_oh / (tot + 1e-6)
    X["ug_share"] = X.exp_length_ug / (tot + 1e-6)
    X["poles_total"] = X.number_of_poles_installed + X.number_of_poles_removed
    X["poles_net"] = X.number_of_poles_installed - X.number_of_poles_removed
    X["len_per_location"] = tot / X.num_of_work_locations.clip(lower=1)
    X["awf_x_len"] = X.AWF * tot
    X["awf_x_locations"] = X.AWF * X.num_of_work_locations
    X["log_len"] = np.log1p(tot.clip(lower=0))
    X["log_awf"] = np.log1p(X.AWF.clip(lower=0))
    X["ug_x_transformer"] = X.is_underground * X.new_or_existing_transformer
    X["poles_x_locations"] = X.number_of_poles_installed * X.num_of_work_locations
    X["sub_code"] = df[SUB].values.argmax(1).astype(float)
    X["amps_code"] = df[AMPS].values.argmax(1).astype(float)
    X["is_three_phase"] = df["con_obj_z_phase_Three"].astype(float)
    X["sub_x_len"] = X.sub_code * 1000 + X.log_len
    X.columns = [clean_name(c) for c in X.columns]
    return X.reset_index(drop=True)


def build_extras(df):
    """Optional columns that are NOT production inputs (for --ablation only)."""
    lookup = {_norm(c): c for c in df.columns}

    def find(cands):
        for c in cands:
            if _norm(c) in lookup:
                return lookup[_norm(c)]
        return None

    out = {}
    c = find(ABLATION_COLS["customers"])
    if c:
        v = pd.to_numeric(df[c], errors="coerce").fillna(0).clip(lower=0)
        out["customers"] = pd.DataFrame({"x_cus": v, "x_log_cus": np.log1p(v)})
    c = find(ABLATION_COLS["rate_category"])
    if c:
        s = df[c].astype(str)
        keep = s.value_counts()
        keep = keep[keep >= 50].index
        s = s.where(s.isin(keep), "OTHER")
        out["rate_category"] = pd.get_dummies(s, prefix="x_rate").astype(float)
    c = find(ABLATION_COLS["complexity"])
    if c:
        v = pd.to_numeric(df[c], errors="coerce")
        if v.isna().all():
            v = df[c].astype("category").cat.codes
        out["complexity"] = pd.DataFrame({"x_complexity": v.fillna(-1).astype(float)})
    for k, v in out.items():
        v.columns = [clean_name(x) for x in v.columns]
    return out


# =============================================================================
# 4. MODELS
# =============================================================================
class RawTarget:
    """Fit on raw dollars (e.g. Tweedie: handles $0 natively); exposes log1p predictions."""
    def __init__(self, est):
        self.est = est

    def fit(self, X, z):
        self.est.fit(X, np.expm1(z))
        return self

    def predict(self, X):
        return np.log1p(np.clip(self.est.predict(X), 0, None))


class Hurdle:
    """P(cost>0) classifier x amount regressor (positives only)."""
    def __init__(self, hard, seed, E):
        self.hard, self.seed, self.E = hard, seed, E

    def fit(self, X, z):
        pos = z > 0
        self.const = None if (pos.any() and (~pos).any()) else float(pos.all())
        if self.const is None:
            self.clf = HistGradientBoostingClassifier(max_iter=self.E(300), learning_rate=0.05,
                                                      random_state=self.seed).fit(X, pos)
        self.reg = HistGradientBoostingRegressor(max_iter=self.E(600), learning_rate=0.04,
                                                 random_state=self.seed).fit(X[pos], z[pos])
        return self

    def predict(self, X):
        p = np.full(len(X), self.const) if self.const is not None else self.clf.predict_proba(X)[:, 1]
        mu = np.expm1(self.reg.predict(X))
        out = np.where(p >= 0.5, mu, 0.0) if self.hard else p * mu
        return np.log1p(np.clip(out, 0, None))


class Specialist:
    """One model per sub-work-type (when enough rows) + a global fallback."""
    def __init__(self, base, min_rows):
        self.base, self.min_rows = base, min_rows

    def fit(self, X, z):
        sub = X["sub_code"].values
        self.glob = self.base()
        self.glob.fit(X, z)
        self.spec = {}
        for t in range(3):
            m = sub == t
            if m.sum() >= self.min_rows:
                mdl = self.base()
                mdl.fit(X[m], z[m])
                self.spec[t] = mdl
        return self

    def predict(self, X):
        out = self.glob.predict(X)
        sub = X["sub_code"].values
        for t, mdl in self.spec.items():
            m = sub == t
            if m.any():
                out[m] = mdl.predict(X[m])
        return out


class Component:
    """Predict project cost and investment separately; output their difference (>=0)."""
    def __init__(self, base, pc, inv_):
        self.base, self.pc, self.inv = base, pc, inv_

    def fit(self, X, z):
        idx = X.index.values
        self.m_pc, self.m_inv = self.base(), self.base()
        self.m_pc.fit(X, np.log1p(self.pc[idx]))
        self.m_inv.fit(X, np.log1p(self.inv[idx]))
        return self

    def predict(self, X):
        d = np.expm1(self.m_pc.predict(X)) - np.expm1(self.m_inv.predict(X))
        return np.log1p(np.clip(d, 0, None))


def get_models(seed, aux=None):
    nj = CFG["n_jobs"]
    E = (lambda n: max(40, n // 12)) if CFG["fast"] else (lambda n: n)
    m = {}
    m["ridge_poly"] = lambda: make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False),
                                            StandardScaler(), Ridge(alpha=30.0))
    m["knn"] = lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=15, weights="distance"))
    m["hgb"] = lambda: HistGradientBoostingRegressor(max_iter=E(600), learning_rate=0.04, max_leaf_nodes=24,
                                                     l2_regularization=1.0, random_state=seed)
    m["et"] = lambda: ExtraTreesRegressor(n_estimators=E(500), min_samples_leaf=2, max_features=0.7,
                                          n_jobs=nj, random_state=seed)
    m["rf"] = lambda: RandomForestRegressor(n_estimators=E(500), min_samples_leaf=2, max_features=0.5,
                                            n_jobs=nj, random_state=seed)
    base_gbm = m["hgb"]
    try:
        import lightgbm as lgb
        m["lgbm"] = lambda: lgb.LGBMRegressor(n_estimators=E(1500), learning_rate=0.02, num_leaves=31,
                                              min_child_samples=20, subsample=0.8, subsample_freq=1,
                                              colsample_bytree=0.8, reg_lambda=1.0, n_jobs=nj,
                                              random_state=seed, verbose=-1)
        m["lgbm_huber"] = lambda: lgb.LGBMRegressor(objective="huber", alpha=0.9, n_estimators=E(1500),
                                                    learning_rate=0.02, num_leaves=15, min_child_samples=30,
                                                    subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                                                    n_jobs=nj, random_state=seed, verbose=-1)
        small = lambda: lgb.LGBMRegressor(n_estimators=E(500), learning_rate=0.03, num_leaves=15,
                                          min_child_samples=15, subsample=0.8, subsample_freq=1,
                                          colsample_bytree=0.8, n_jobs=nj, random_state=seed, verbose=-1)
        base_gbm = small
    except ImportError:
        LOG.info("lightgbm not installed - skipping lgbm models (HGB used for specialists/component)")
    try:
        import xgboost as xgb
        m["xgb"] = lambda: xgb.XGBRegressor(n_estimators=E(1200), learning_rate=0.03, max_depth=6,
                                            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                                            reg_lambda=1.0, random_state=seed, n_jobs=nj)
        m["xgb_tweedie"] = lambda: RawTarget(xgb.XGBRegressor(
            objective="reg:tweedie", tweedie_variance_power=1.5, n_estimators=E(1000), learning_rate=0.03,
            max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=nj))
    except ImportError:
        LOG.info("xgboost not installed - skipping xgb models")
    try:
        from catboost import CatBoostRegressor
        tc = nj if nj and nj > 0 else -1
        m["cat"] = lambda: CatBoostRegressor(iterations=E(1500), learning_rate=0.03, depth=6,
                                             random_seed=seed, verbose=0, thread_count=tc)
        m["cat_mae"] = lambda: CatBoostRegressor(iterations=E(1500), learning_rate=0.03, depth=6,
                                                 loss_function="MAE", random_seed=seed, verbose=0, thread_count=tc)
    except ImportError:
        LOG.info("catboost not installed - skipping cat models")
    m["hurdle_soft"] = lambda: Hurdle(False, seed, E)
    m["hurdle_hard"] = lambda: Hurdle(True, seed, E)
    m["spec_lgbm"] = lambda: Specialist(base_gbm, CFG["specialist_min_rows"])
    if aux is not None:
        m["component"] = lambda: Component(base_gbm, aux[0], aux[1])
    m["svr"] = lambda: make_pipeline(StandardScaler(), SVR(C=3.0, epsilon=0.05, gamma="scale"))
    m["mlp"] = lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3,
                                                                    early_stopping=True, max_iter=500,
                                                                    random_state=seed))
    return m


def cache_path(key):
    d = os.path.join(CFG["out_dir"], "cache")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, hashlib.sha1(key.encode()).hexdigest()[:20] + ".npz")


def train_models(featset, X, Xho, z, folds, seed, names, ctx, deadline, aux=None):
    """Return {model: (oof_log, holdout_log)}; honours cache, time budget, per-model failures."""
    out, factories = {}, get_models(seed, aux)
    nf = CFG["n_folds"]
    for name in [n for n in MODEL_ORDER if n in names]:
        if name not in factories:
            LOG.info("[%s s%d] %-12s unavailable", featset, seed, name)
            continue
        cp = cache_path(f"{ctx}|{featset}|{seed}|{name}|{MODEL_VER}|{CFG['fast']}|{list(X.columns)}")
        if os.path.exists(cp) and not CFG.get("no_cache"):
            d = np.load(cp)
            out[name] = (d["oof"], d["hold"])
            LOG.info("[%s s%d] %-12s cached", featset, seed, name)
            continue
        if time.time() > deadline:
            LOG.info("[%s s%d] %-12s SKIPPED (time budget reached)", featset, seed, name)
            continue
        t0 = time.time()
        try:
            oof, hold = np.full(len(X), np.nan), np.zeros(len(Xho))
            for k in range(nf):
                tri, vai = np.where(folds != k)[0], np.where(folds == k)[0]
                mdl = factories[name]()
                mdl.fit(X.iloc[tri], z[tri])
                oof[vai] = mdl.predict(X.iloc[vai])
                hold += mdl.predict(Xho) / nf
            cap = np.log1p(1e7)
            oof = np.clip(np.nan_to_num(oof, nan=0.0, posinf=cap, neginf=0.0), 0, cap)
            hold = np.clip(np.nan_to_num(hold, nan=0.0, posinf=cap, neginf=0.0), 0, cap)
            np.savez(cp, oof=oof, hold=hold)
            out[name] = (oof, hold)
            LOG.info("[%s s%d] %-12s done in %.0fs", featset, seed, name, time.time() - t0)
        except Exception as e:  # one failing model never stops the others
            LOG.info("[%s s%d] %-12s FAILED: %s: %s", featset, seed, name, type(e).__name__, str(e)[:160])
    return out


# =============================================================================
# 5. BLENDING
# =============================================================================
def hill_climb(L, y, iters=50):
    names, chosen, best = list(L.columns), [], -1e9
    cur = np.zeros(len(L))
    for _ in range(iters):
        cand, cs = None, best
        for n in names:
            s = objective(y, inv((cur + L[n].values) / (len(chosen) + 1)))
            if s > cs + 1e-9:
                cs, cand = s, n
        if cand is None:
            break
        chosen.append(cand)
        cur += L[cand].values
        best = cs
    return pd.Series(chosen).value_counts(normalize=True).to_dict(), best


def blend_with(L, w):
    return sum(L[k].values * v for k, v in w.items())


def stack_nested(L, z, folds):
    pred = np.zeros(len(L))
    for k in range(CFG["n_folds"]):
        tri, vai = np.where(folds != k)[0], np.where(folds == k)[0]
        pred[vai] = LinearRegression(positive=True).fit(L.iloc[tri], z[tri]).predict(L.iloc[vai])
    full = LinearRegression(positive=True).fit(L, z)
    return pred, full


# =============================================================================
# 6. CALIBRATION (multiplicative, by sub-type x predicted quartile)
# =============================================================================
def cells(pred, sub, edges):
    return sub * 4 + np.digitize(pred, edges)


def fit_cal(pred, y, sub, edges):
    c = cells(pred, sub, edges)
    lo, hi, st = CFG["cal_grid"]
    grid = np.arange(lo, hi + 1e-9, st)
    fac = {}
    for cell in np.unique(c):
        m = c == cell
        if m.sum() < CFG["min_cell"]:
            continue
        best_f, best_s = 1.0, objective(y[m], pred[m])
        for f in grid:
            s = objective(y[m], pred[m] * f)
            if s > best_s + 0.003:
                best_f, best_s = float(f), s
        if best_f != 1.0:
            fac[int(cell)] = 1 + 0.5 * (best_f - 1)   # shrink halfway toward 1
    return fac


def apply_cal(pred, sub, edges, fac):
    c = cells(pred, sub, edges)
    f = np.array([fac.get(int(x), 1.0) for x in c])
    return pred * f


def calibrate_crossfit(pred, y, sub, folds):
    edges = np.quantile(pred, [0.25, 0.5, 0.75])
    out = np.zeros(len(pred))
    for k in range(CFG["n_folds"]):
        tri, vai = folds != k, folds == k
        fac = fit_cal(pred[tri], y[tri], sub[tri], edges)
        out[vai] = apply_cal(pred[vai], sub[vai], edges, fac)
    return out


# =============================================================================
# DIAGNOSTICS
# =============================================================================
def run_diagnostics(df):
    lines = []
    P = lines.append
    y = df[CFG["target"]].values
    sub = df[SUB].values.argmax(1)
    P("== target by sub-work-type ==")
    for i, nme in enumerate(SUB_NAMES):
        m = sub == i
        yy = y[m]
        if not m.any():
            continue
        q = np.quantile(yy, [0, .05, .25, .5, .75, .95, .99, 1])
        P(f"{nme}: n={m.sum()} zero={100*(yy==0).mean():.1f}% under$50={100*(yy<50).mean():.1f}% "
          f"quantiles(min,p5,p25,p50,p75,p95,p99,max)={np.round(q).astype(int).tolist()}")
    P("\n== duplicates on the 20 inputs (noise ceiling) ==")
    g = df.groupby(INPUTS, dropna=False)[CFG["target"]]
    size = g.transform("size")
    dup = size > 1
    P(f"rows in duplicate groups: {int(dup.sum())} ({100*dup.mean():.1f}%), groups={df[dup].groupby(INPUTS).ngroups}")
    if dup.any():
        d = df[dup].copy()
        grp = d.groupby(INPUTS, dropna=False)[CFG["target"]]
        spread = (grp.transform("max") - grp.transform("min")) / grp.transform("mean").replace(0, np.nan)
        P(f"median relative spread inside groups: {np.nanmedian(spread):.2f}")
        # leave-one-out group median as the prediction = best any model can do from these inputs
        s, c = grp.transform("sum"), grp.transform("size")
        loo_mean = (s - d[CFG["target"]]) / (c - 1)
        k = kpi(d[CFG["target"]].values, loo_mean.values)
        P(f"leave-one-out group-mean prediction on duplicates: hit={k['hit']:.3f} under={k['under']:.3f} "
          f"(a rough noise ceiling for these rows)")
        mix = (grp.transform("min") != grp.transform("max")).mean()
        P(f"share of duplicate rows whose group has differing targets: {100*mix:.1f}%")
    P("\n== sub-type median baseline (positives median per type) ==")
    base = np.zeros(len(y))
    for i in range(3):
        m = sub == i
        if m.any() and (y[m] > 0).any():
            base[m] = np.median(y[m][y[m] > 0])
    P(kpi_by_type(sub, y, base).round(3).to_string())
    col = CFG.get("id_col_resolved")
    P("\n== work order id ==")
    if col:
        v = pd.to_numeric(df[col], errors="coerce")
        P(f"unique={v.nunique()} of {len(v)}; sorted in file: {bool(v.is_monotonic_increasing)}; "
          f"spearman(row order, id)={v.reset_index(drop=True).corr(pd.Series(np.arange(len(v))), method='spearman'):.3f}")
        P("(a high correlation hints the id is a time proxy; confirm with the business before using workorder_time)")
    else:
        P("id column not found")
    txt = "\n".join(lines)
    LOG.info("\n%s", txt)
    open(os.path.join(CFG["out_dir"], "diagnostics.txt"), "w", encoding="utf-8").write(txt)


# =============================================================================
# MAIN
# =============================================================================
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data")
    ap.add_argument("--out")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--force-final", action="store_true")
    ap.add_argument("--diagnostics", action="store_true")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--models")
    ap.add_argument("--seeds")
    ap.add_argument("--budget-min", type=float)
    ap.add_argument("--split-mode", choices=["random", "workorder_time", "ids"])
    ap.add_argument("--holdout-ids")
    ap.add_argument("--reset-split", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--note", default="")
    return ap.parse_args()


def main():
    a = parse_args()
    t_start = time.time()
    if a.data: CFG["data_path"] = a.data
    if a.out: CFG["out_dir"] = a.out
    if a.models: CFG["models"] = a.models.split(",")
    if a.seeds: CFG["seeds"] = [int(s) for s in a.seeds.split(",")]
    if a.budget_min: CFG["time_budget_min"] = a.budget_min
    if a.split_mode: CFG["split_mode"] = a.split_mode
    if a.holdout_ids: CFG["holdout_ids_file"] = a.holdout_ids
    CFG["fast"] = a.fast or CFG["fast"]
    CFG["no_cache"] = a.no_cache
    os.makedirs(CFG["out_dir"], exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(CFG["out_dir"], run_id)
    LOG.info("pipeline v%s run %s | split=%s seeds=%s budget=%smin fast=%s",
             SCRIPT_VERSION, run_id, CFG["split_mode"], CFG["seeds"], CFG["time_budget_min"], CFG["fast"])
    deadline = t_start + CFG["time_budget_min"] * 60

    df = load_and_validate(CFG["data_path"])
    if a.diagnostics:
        run_diagnostics(df)
        return

    dh = data_hash(df, INPUTS + [CFG["target"]])
    seeds = CFG["seeds"]
    tr_idx, ho_idx, folds = frozen_split(df, dh, seeds, a.reset_split)
    d_tr, d_ho = df.iloc[tr_idx].reset_index(drop=True), df.iloc[ho_idx].reset_index(drop=True)
    X_full, Xho_full = make_features(d_tr), make_features(d_ho)
    keep = [c for c in X_full.columns if X_full[c].nunique() > 1 or c == "sub_code"]
    X, Xho = X_full[keep], Xho_full[keep]
    y = d_tr[CFG["target"]].values
    z = np.log1p(y)
    sub = d_tr[SUB].values.argmax(1)
    sub_ho = d_ho[SUB].values.argmax(1)
    LOG.info("features: %d columns; train=%d holdout=%d (holdout is NOT scored unless --final)",
             X.shape[1], len(X), len(Xho))

    aux = None
    pc_col, inv_col = CFG.get("project_cost_col_resolved"), CFG.get("investment_col_resolved")
    if pc_col and inv_col and pc_col in d_tr and inv_col in d_tr:
        aux = (pd.to_numeric(d_tr[pc_col], errors="coerce").fillna(0).clip(lower=0).values,
               pd.to_numeric(d_tr[inv_col], errors="coerce").fillna(0).clip(lower=0).values)
        LOG.info("component model enabled (project cost / investment used ONLY as training targets)")
    else:
        LOG.info("component model disabled (project cost / investment columns not found)")

    names = CFG["models"] or MODEL_ORDER
    ctx = f"{dh}|{CFG['split_mode']}|{len(tr_idx)}|{CFG['n_folds']}"

    # ---- baseline: sub-type median of positives (reference, not blended)
    base_oof = np.zeros(len(y))
    f0 = folds[seeds[0]]
    for k in range(CFG["n_folds"]):
        tri, vai = f0 != k, f0 == k
        for t in range(3):
            m_t = tri & (sub == t) & (y > 0)
            med = np.median(y[m_t]) if m_t.any() else 0.0
            base_oof[vai & (sub == t)] = med
    kb = kpi(y, base_oof)
    LOG.info("BASELINE sub-type median: hit=%.3f under=%.3f", kb["hit"], kb["under"])

    # ---- 4. model zoo
    per_seed = {}
    for s in seeds:
        per_seed[s] = train_models("base", X, Xho, z, folds[s], s, names, ctx, deadline, aux)
    done = [n for n in MODEL_ORDER if all(n in per_seed[s] for s in seeds) and n in names]
    if not done:
        raise SystemExit("no model finished (check the log above)")
    skipped = [n for n in names if n not in done]
    oof_log = pd.DataFrame({n: np.mean([per_seed[s][n][0] for s in seeds], axis=0) for n in done})
    hold_log = pd.DataFrame({n: np.mean([per_seed[s][n][1] for s in seeds], axis=0) for n in done})

    rows = []
    for n in done:
        k = kpi(y, inv(oof_log[n].values))
        seed_hits = [kpi(y, inv(per_seed[s][n][0]))["hit"] for s in seeds]
        rows.append(dict(model=n, hit=k["hit"], hit_pos=k["hit_pos"], hit_zero=k["hit_zero"], under=k["under"],
                         mae=k["mae"], hit_sd=float(np.std(seed_hits)) if len(seeds) > 1 else np.nan))
    tbl = pd.DataFrame(rows).sort_values("hit", ascending=False).reset_index(drop=True)
    LOG.info("\nSINGLE MODELS (OOF, official KPI: hit = within +/-50%%, zeros within $%d)\n%s",
             CFG["zero_tol"], tbl.round(4).to_string(index=False))
    best_single = tbl.iloc[0]

    # ---- 5. blend: hill-climb vs nested stacking
    w, _ = hill_climb(oof_log, y)
    p_hill = inv(blend_with(oof_log, w))
    stack_pred, stack_full = stack_nested(oof_log, z, folds[seeds[0]])
    p_stack = inv(stack_pred)
    s_hill, s_stack = objective(y, p_hill), objective(y, p_stack)
    if s_stack > s_hill + 1e-9:
        method, p_blend = "stack", p_stack
        coefs = dict(zip(oof_log.columns, np.round(stack_full.coef_, 4)))
    else:
        method, p_blend = "hill_climb", p_hill
        coefs = None
    LOG.info("BLEND hill-climb objective=%.4f | stack(nested) objective=%.4f -> using %s", s_hill, s_stack, method)
    LOG.info("weights: %s", {k: round(v, 3) for k, v in w.items()} if method == "hill_climb" else coefs)

    # ---- 6. calibration (accept only if cross-fitted gain)
    p_cal = calibrate_crossfit(p_blend, y, sub, folds[seeds[0]])
    gain = objective(y, p_cal) - objective(y, p_blend)
    use_cal = gain >= CFG["cal_min_gain"]
    LOG.info("CALIBRATION cross-fitted objective gain=%+.4f -> %s", gain, "APPLIED" if use_cal else "not applied")
    edges = np.quantile(p_blend, [0.25, 0.5, 0.75])
    fac = fit_cal(p_blend, y, sub, edges) if use_cal else {}
    p_final = apply_cal(p_blend, sub, edges, fac) if use_cal else p_blend

    kf = kpi(y, p_final)
    LOG.info("\nFINAL OOF PIPELINE: hit=%.4f (hit_pos=%.4f hit_zero=%.4f) under>10%%=%.4f  target >= 0.68  -> %s",
             kf["hit"], kf["hit_pos"], kf["hit_zero"], kf["under"], "MEETS" if kf["hit"] >= 0.68 else "below")
    rep = kpi_by_type(sub, y, p_final)
    LOG.info("\nBY SUB-WORK-TYPE (OOF)\n%s", rep.round(4).to_string())
    rep.to_csv(os.path.join(CFG["out_dir"], "report_by_type.csv"))
    if len(seeds) > 1:
        sh = []
        for s in seeds:
            Ls = pd.DataFrame({n: per_seed[s][n][0] for n in done})
            sh.append(kpi(y, inv(blend_with(Ls, w)))["hit"] if method == "hill_climb"
                      else kpi(y, inv(stack_full.predict(Ls)))["hit"])
        LOG.info("blend hit by seed: %s mean=%.4f sd=%.4f", np.round(sh, 4).tolist(), np.mean(sh), np.std(sh))

    # save artifacts
    res = pd.DataFrame({"actual": y, "sub": np.array(SUB_NAMES)[sub], "baseline": base_oof,
                        "blend": p_blend, "final": p_final})
    for n in done:
        res["oof_" + n] = inv(oof_log[n].values)
    if CFG["id_col_resolved"]:
        res.insert(0, CFG["id_col"], d_tr[CFG["id_col_resolved"]].values)
    res.to_csv(os.path.join(CFG["out_dir"], "oof_predictions.csv"), index=False)
    blend_doc = dict(method=method, hill_weights=w, stack_coef=coefs, calibration_applied=bool(use_cal),
                     calibration_factors={str(k): v for k, v in fac.items()}, calibration_edges=edges.tolist(),
                     models=done, skipped=skipped, features=keep, script_version=SCRIPT_VERSION)
    json.dump(blend_doc, open(os.path.join(CFG["out_dir"], "blend.json"), "w"), indent=2)

    # ---- ablation
    if a.ablation:
        run_ablation(df, d_tr, d_ho, X, Xho, z, y, folds, seeds[0], ctx, deadline, aux, sub)

    # ---- 8. final holdout (once)
    scored = False
    if a.final:
        guard = os.path.join(CFG["out_dir"], "final_runs.jsonl")
        if os.path.exists(guard) and not a.force_final:
            LOG.info("REFUSING --final: the holdout was already scored (see %s). Use --force-final to override.", guard)
        else:
            if method == "hill_climb":
                ph = inv(blend_with(hold_log, w))
            else:
                ph = inv(stack_full.predict(hold_log))
            if use_cal:
                ph = apply_cal(ph, sub_ho, edges, fac)
            yh = d_ho[CFG["target"]].values
            kh = kpi(yh, ph)
            LOG.info("\n*** FINAL HOLDOUT (n=%d): hit=%.4f hit_pos=%.4f hit_zero=%.4f under>10%%=%.4f -> %s ***",
                     len(yh), kh["hit"], kh["hit_pos"], kh["hit_zero"], kh["under"],
                     "MEETS 0.68" if kh["hit"] >= 0.68 else "below 0.68")
            LOG.info("\n%s", kpi_by_type(sub_ho, yh, ph).round(4).to_string())
            out = pd.DataFrame({"actual": yh, "pred": ph, "sub": np.array(SUB_NAMES)[sub_ho]})
            if CFG["id_col_resolved"]:
                out.insert(0, CFG["id_col"], d_ho[CFG["id_col_resolved"]].values)
            out.to_csv(os.path.join(CFG["out_dir"], "holdout_predictions.csv"), index=False)
            open(guard, "a").write(json.dumps(dict(run=run_id, hit=kh["hit"], n=len(yh))) + "\n")
            scored = True

    # ---- 7. run history
    hist = dict(run_id=run_id, ts=datetime.now().isoformat(timespec="seconds"), version=SCRIPT_VERSION,
                split=CFG["split_mode"], seeds="|".join(map(str, seeds)), fast=CFG["fast"],
                n_models=len(done), skipped="|".join(skipped), method=method, calibration=bool(use_cal),
                oof_hit=round(kf["hit"], 4), oof_hit_pos=round(kf["hit_pos"], 4), oof_under10=round(kf["under"], 4),
                baseline_hit=round(kb["hit"], 4), best_single=best_single["model"],
                best_single_hit=round(float(best_single["hit"]), 4),
                hit_LE=round(rep.loc["LE", "hit"], 4) if "LE" in rep.index else None,
                hit_PRI=round(rep.loc["PRI", "hit"], 4) if "PRI" in rep.index else None,
                hit_SEC=round(rep.loc["SEC", "hit"], 4) if "SEC" in rep.index else None,
                holdout_scored=scored, minutes=round((time.time() - t_start) / 60, 1), note=a.note)
    hp = os.path.join(CFG["out_dir"], "run_history.csv")
    pd.DataFrame([hist]).to_csv(hp, mode="a", header=not os.path.exists(hp), index=False)
    LOG.info("done in %.1f min. history: %s", (time.time() - t_start) / 60, hp)


def run_ablation(df, d_tr, d_ho, X, Xho, z, y, folds, seed, ctx, deadline, aux, sub):
    LOG.info("\n=== ABLATION (value of inputs that are NOT in the production contract) ===")
    extras_all = build_extras(df)
    if not extras_all:
        LOG.info("none of the optional columns were found; nothing to ablate")
        return
    tr_idx = d_tr.index  # d_tr is already positional
    tr_pos = np.array(json.load(open(os.path.join(CFG["out_dir"], "split.json")))["train_idx"])
    ho_pos = np.array(json.load(open(os.path.join(CFG["out_dir"], "split.json")))["holdout_idx"])
    variants = {"A0_contract_only": []}
    for k in extras_all:
        variants[f"A+{k}"] = [k]
    if len(extras_all) > 1:
        variants["A_all_extras"] = list(extras_all)
    abl_models = [m for m in ("lgbm", "et", "hgb") if m in get_models(seed)] or ["hgb"]
    rows = []
    for vname, ex in variants.items():
        Xa, Xha = X.copy(), Xho.copy()
        for k in ex:
            Xa = pd.concat([Xa, extras_all[k].iloc[tr_pos].reset_index(drop=True)], axis=1)
            Xha = pd.concat([Xha, extras_all[k].iloc[ho_pos].reset_index(drop=True)], axis=1)
        res = train_models(vname, Xa, Xha, z, folds[seed], seed, abl_models, ctx, deadline, aux=None)
        if not res:
            continue
        p = inv(np.mean([v[0] for v in res.values()], axis=0))
        k = kpi(y, p)
        rows.append(dict(variant=vname, models="+".join(res), hit=k["hit"], hit_pos=k["hit_pos"],
                         hit_zero=k["hit_zero"], under=k["under"], mae=k["mae"]))
    if rows:
        t = pd.DataFrame(rows)
        t["lift_vs_contract"] = t["hit"] - t.loc[0, "hit"]
        LOG.info("\n%s", t.round(4).to_string(index=False))
        t.to_csv(os.path.join(CFG["out_dir"], "ablation.csv"), index=False)
        LOG.info("Only fields confirmed to be known at estimate time may become production inputs.")


if __name__ == "__main__":
    main()
