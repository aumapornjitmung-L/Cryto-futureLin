"""crypto_ml.py — โหลด CSV รายวัน, ซ่อมจุดผิดปกติ, สร้างฟีเจอร์, เทียบโมเดลแบบ walk-forward และพยากรณ์"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # xgboost ไม่ได้ติดตั้ง -> ข้ามโมเดลนี้
    XGBClassifier = XGBRegressor = None

COINS = ["QUSDT", "BRUSDT", "GTCUSDT", "PHAUSDT"]  # ไฟล์ข้อมูลอยู่ที่ data/<SYMBOL>.csv
MIN_ROWS = 150

MODEL_LABELS = {
    "linear": "Logistic/Ridge (เส้นฐาน)",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "hist_gb": "HistGradientBoosting",
    "xgboost": "XGBoost",
}


class DataError(Exception):
    pass


# ============================================================ โหลดและทำความสะอาด
_ALIASES = {
    "time": ["event_date", "date", "time", "timestamp", "datetime", "snapped_at"],
    "close": ["close_price_usd", "close", "price", "price_usd"],
    "volume": ["volume_usd", "volume", "total_volume", "total_volumes"],
}


def load_coin_csv(src) -> tuple[pd.DataFrame, dict]:
    """อ่าน CSV รายวัน (รูปแบบ CoinGecko: event_date, close_price_usd, market_cap_usd, volume_usd)
    คืน (df[time, close, volume], info) — ตัดแถวที่ราคาปิดว่าง (วันปัจจุบันที่ยังไม่จบ) ออก"""
    raw = pd.read_csv(src)
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    pick = {k: next((c for c in v if c in raw.columns), None) for k, v in _ALIASES.items()}
    if pick["time"] is None or pick["close"] is None:
        raise DataError("CSV ต้องมีคอลัมน์วันที่ (event_date/date) และราคาปิด (close_price_usd/close/price)")
    df = pd.DataFrame({
        "time": pd.to_datetime(raw[pick["time"]].astype(str).str.replace(" UTC", "", regex=False), errors="coerce"),
        "close": pd.to_numeric(raw[pick["close"]], errors="coerce"),
        "volume": pd.to_numeric(raw[pick["volume"]], errors="coerce") if pick["volume"] else np.nan,
    })
    if hasattr(df["time"].dt, "tz") and df["time"].dt.tz is not None:
        df["time"] = df["time"].dt.tz_localize(None)
    n_raw = len(df)
    df = df.dropna(subset=["time"]).drop_duplicates("time").sort_values("time")
    n_incomplete = int(df["close"].isna().sum() + (df["close"] <= 0).sum())
    df = df[df["close"] > 0].reset_index(drop=True)
    df["volume"] = df["volume"].where(df["volume"] > 0)
    gaps = int((df["time"].diff().dt.days > 1).sum())
    if len(df) < MIN_ROWS:
        raise DataError(f"ข้อมูลน้อยเกินไป ({len(df)} แถว ต้องการอย่างน้อย {MIN_ROWS})")
    return df, {"rows_raw": n_raw, "dropped_incomplete": n_incomplete, "date_gaps": gaps}


def repair_spikes(close: pd.Series, max_run: int = 4, big: float = 0.5,
                  dev: float = 0.3, back: float = 0.3) -> tuple[pd.Series, np.ndarray]:
    """ซ่อมจุดราคาที่ "กระโดดแล้วกลับระดับเดิมภายใน max_run วัน" (มักเป็น bad tick ของผู้รวบรวมข้อมูล)
    - ช่วงที่กระโดดต้องเบี่ยงจากราคาก่อนหน้า > dev ทุกวัน และอย่างน้อย 1 วัน > big (ในหน่วย log)
    - ราคาหลังช่วงนั้นต้องกลับมาใกล้ราคาก่อนหน้า (ต่างไม่เกิน back)
    การเปลี่ยนระดับราคาแบบถาวรจึงไม่ถูกแตะ จุดที่ตรวจพบถูกแทนด้วยการประมาณค่าระหว่างจุดข้างเคียง"""
    lc = np.log(close.to_numpy(dtype=float))
    n, bad, i, a = len(lc), np.zeros(len(lc), bool), 1, 0  # a = ดัชนีของจุดล่าสุดที่ไม่ถูกซ่อม (จุดอ้างอิง)
    while i < n - 1:
        hit = False
        for k in range(1, max_run + 1):
            j = i + k
            if j >= n:
                break
            if abs(lc[j] - lc[a]) >= back:
                continue
            run = lc[i:j] - lc[a]
            if np.all(np.abs(run) > dev) and np.abs(run).max() > big and np.all(np.sign(run) == np.sign(run[0])):
                bad[i:j], i, hit = True, j, True
                break
        if not hit:
            a, i = i, i + 1
    fixed = pd.Series(np.where(bad, np.nan, lc), index=close.index).interpolate(limit_area="inside")
    return pd.Series(np.exp(fixed), index=close.index, name=close.name), bad


def prepare(src, repair: bool = True) -> tuple[pd.DataFrame, dict]:
    df, info = load_coin_csv(src)
    df["close_raw"] = df["close"]
    bad = np.zeros(len(df), bool)
    if repair:
        df["close"], bad = repair_spikes(df["close"])
    df["repaired"] = bad
    info["n_repaired"] = int(bad.sum())
    return df, info


# ============================================================ ฟีเจอร์
def _rsi(c: pd.Series, n: int = 14) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).where(dn != 0, 100.0)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    c, v = df["close"], df["volume"]
    lc = np.log(c)
    r = lc.diff()
    f = pd.DataFrame(index=df.index)
    for k in (1, 2, 3, 5, 7, 14, 30):
        f[f"ret_{k}"] = lc.diff(k)
    for w in (7, 14, 30):
        f[f"vol_{w}"] = r.rolling(w).std()
    f["rsi14"] = _rsi(c)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    for s in (7, 21, 50):
        f[f"ema{s}_ratio"] = c / c.ewm(span=s, adjust=False).mean() - 1
    sd20 = c.rolling(20).std().replace(0, np.nan)
    f["bb_pos"] = (c - c.rolling(20).mean()) / (2 * sd20)
    f["dd30"] = c / c.rolling(30).max() - 1
    f["up30"] = c / c.rolling(30).min() - 1
    lv = np.log(v)
    f["logvol_z20"] = (lv - lv.rolling(20).mean()) / lv.rolling(20).std().replace(0, np.nan)
    f["dlogvol_1"] = lv.diff(1)
    f["dlogvol_7"] = lv.diff(7)
    f["dow_sin"] = np.sin(2 * np.pi * df["time"].dt.dayofweek / 7)
    f["dow_cos"] = np.cos(2 * np.pi * df["time"].dt.dayofweek / 7)
    return f.replace([np.inf, -np.inf], np.nan)


def build_dataset(df: pd.DataFrame, horizon: int):
    X = make_features(df)
    y = np.log(df["close"].shift(-horizon) / df["close"])  # log-return ล่วงหน้า h วัน
    return X, y


def _train_xy(df: pd.DataFrame, horizon: int):
    X, y = build_dataset(df, horizon)
    mask = X.notna().all(axis=1) & y.notna()
    if mask.sum() < 100:
        raise DataError(f"ข้อมูลหลังสร้างฟีเจอร์เหลือน้อยเกินไป ({int(mask.sum())} แถว)")
    return X, y, mask


# ============================================================ โมเดล
def available_models() -> list[str]:
    names = ["linear", "random_forest", "extra_trees", "hist_gb"]
    return names + (["xgboost"] if XGBClassifier is not None else [])


def make_model(name: str, task: str, seed: int = 42):
    clf = task == "clf"
    if name == "linear":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000) if clf else Ridge(alpha=10.0))
    if name == "random_forest":
        cls = RandomForestClassifier if clf else RandomForestRegressor
        return cls(n_estimators=200, min_samples_leaf=5, max_features="sqrt", n_jobs=-1, random_state=seed)
    if name == "extra_trees":
        cls = ExtraTreesClassifier if clf else ExtraTreesRegressor
        return cls(n_estimators=200, min_samples_leaf=5, max_features="sqrt", n_jobs=-1, random_state=seed)
    if name == "hist_gb":
        cls = HistGradientBoostingClassifier if clf else HistGradientBoostingRegressor
        return cls(max_iter=150, learning_rate=0.05, max_depth=4, l2_regularization=1.0, random_state=seed)
    if name == "xgboost" and XGBClassifier is not None:
        kw = dict(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, random_state=seed, n_jobs=-1)
        return XGBClassifier(eval_metric="logloss", **kw) if clf else XGBRegressor(**kw)
    raise ValueError(f"ไม่รู้จักโมเดล: {name}")


def _prob_up(model, X) -> np.ndarray:
    classes = list(model.classes_) if hasattr(model, "classes_") else list(model[-1].classes_)
    return model.predict_proba(X)[:, classes.index(1)] if 1 in classes else np.zeros(len(X))


def _folds(n: int, horizon: int, n_folds: int = 4, start_frac: float = 0.5):
    """expanding window: ฝึกด้วยข้อมูล 50% แรก แล้วทดสอบทีละช่วงถัดไป (เว้น gap = horizon กันข้อมูลรั่ว)"""
    start = int(n * start_frac)
    block = (n - start) // n_folds
    for k in range(n_folds):
        te0 = start + k * block
        te1 = n if k == n_folds - 1 else te0 + block
        yield np.arange(0, te0 - horizon), np.arange(te0, te1)


def walk_forward(X: pd.DataFrame, y: pd.Series, name: str, horizon: int, with_reg: bool = True) -> dict:
    yd = (y > 0).astype(int)
    acc, auc, base, hit, mae_r, n_test = [], [], [], [], [], 0
    for tr, te in _folds(len(X), horizon):
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], yd.iloc[tr], yd.iloc[te]
        if ytr.nunique() < 2:
            continue
        p = _prob_up(make_model(name, "clf").fit(Xtr, ytr), Xte)
        acc.append(accuracy_score(yte, p > 0.5))
        auc.append(roc_auc_score(yte, p) if yte.nunique() == 2 else np.nan)
        base.append(float((yte == int(ytr.mean() >= 0.5)).mean()))
        n_test += len(te)
        if with_reg:
            pr = make_model(name, "reg").fit(Xtr, y.iloc[tr]).predict(Xte)
            yt = y.iloc[te].to_numpy()
            hit.append(float((np.sign(pr) == np.sign(yt)).mean()))
            mae_r.append(float(np.mean(np.abs(yt - pr)) / np.mean(np.abs(yt))))
    out = {"acc": float(np.mean(acc)), "auc": float(np.nanmean(auc)),
           "acc_baseline": float(np.mean(base)), "n_test": n_test}
    if with_reg:
        out.update({"reg_hit": float(np.mean(hit)), "mae_ratio": float(np.mean(mae_r))})
    return out


def forecast(df: pd.DataFrame, model_names: list[str], horizon: int = 1, evaluate: bool = True) -> dict:
    X, y, mask = _train_xy(df, horizon)
    x_last = X.iloc[[-1]]
    if x_last.isna().any(axis=None):
        raise DataError("แถวล่าสุดคำนวณฟีเจอร์ไม่ได้ (ข้อมูลสั้นเกินไป)")
    last_close = float(df["close"].iloc[-1])
    out = {}
    for name in model_names:
        clf = make_model(name, "clf").fit(X[mask], (y[mask] > 0).astype(int))
        reg = make_model(name, "reg").fit(X[mask], y[mask])
        ret = float(reg.predict(x_last)[0])
        out[name] = {
            "prob_up": float(_prob_up(clf, x_last)[0]),
            "pred_return": ret,
            "pred_price": last_close * float(np.exp(ret)),
            "metrics": walk_forward(X[mask], y[mask], name, horizon) if evaluate else None,
        }
    return out


def benchmark(data_dir="data", coins=COINS, horizon: int = 1, repair: bool = True, names=None):
    """เทียบโมเดลทุกตัวบนทุกเหรียญ คืน (ตารางละเอียด, ตารางสรุปเรียงตาม AUC เฉลี่ย)"""
    names = names or available_models()
    rows = []
    for s in coins:
        p = Path(data_dir) / f"{s}.csv"
        if not p.exists():
            print(f"[ข้าม] ไม่พบ {p}")
            continue
        df, info = prepare(p, repair)
        X, y, mask = _train_xy(df, horizon)
        print(f"{s}: {len(df)} วัน | ซ่อม {info['n_repaired']} จุด | ใช้ฝึก/ทดสอบ {int(mask.sum())} แถว")
        for n in names:
            rows.append({"symbol": s, "model": n, **walk_forward(X[mask], y[mask], n, horizon)})
    if not rows:
        raise DataError("ไม่พบไฟล์ข้อมูลของเหรียญใดเลย")
    res = pd.DataFrame(rows)
    summary = (res.groupby("model")[["auc", "acc", "acc_baseline", "reg_hit", "mae_ratio"]].mean()
               .sort_values(["auc", "acc"], ascending=False))
    return res, summary
