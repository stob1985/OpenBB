"""
Advanced Features modul (Fazis 3)
=================================
Profi szintu optimalizaciok a crypto_analyzer-hez:
  - Egysegesitett piaci regime detektalas (BULL/BEAR/SIDEWAYS/HIGH_VOL/NORMAL)
  - Regime-aware auto-optimalizacio (RSI kuszobok, lookback, hold period)
  - Advisor mode-ok (CONSERVATIVE/NORMAL/AGGRESSIVE)
  - Session-aware scoring (ASIA/EU/US)
  - Volatility forecast
  - Multi-horizon profit factor
  - Correlation tracking (modell vs valosag, Pearson)
  - Opcionalis: holdfazis (ephem nelkul, kozelitessel)

Minden fuggveny graceful: hianyzo adat eseten ertelmes alapertelmezessel ter vissza.
"""

import json
import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

CORR_DIR = "results/correlation"


# ---------------------------------------------------------------------------
# 1-2. MARKET REGIME + AUTO-OPTIMIZATION
# ---------------------------------------------------------------------------
REGIME_SETTINGS = {
    "BULL": {
        "rsi_ob": 75, "rsi_os": 30, "pivot_lookback": 5, "hold_period": 5,
        "min_expected_pf": 1.30, "preferred_strategy": "golden_long",
        "preferred_dir": "LONG", "max_position_size_pct": 1.0,
    },
    "BEAR": {
        "rsi_ob": 65, "rsi_os": 25, "pivot_lookback": 10, "hold_period": 5,
        "min_expected_pf": 1.25, "preferred_strategy": "inverse_golden",
        "preferred_dir": "SHORT", "max_position_size_pct": 1.0,
    },
    "SIDEWAYS": {
        "rsi_ob": 70, "rsi_os": 30, "pivot_lookback": 7, "hold_period": 3,
        "min_expected_pf": 1.15, "preferred_strategy": "breakout",
        "preferred_dir": None, "max_position_size_pct": 1.0,
    },
    "HIGH_VOL": {
        "rsi_ob": 80, "rsi_os": 20, "pivot_lookback": 3, "hold_period": 2,
        "min_expected_pf": 1.40, "preferred_strategy": "breakout",
        "preferred_dir": None, "max_position_size_pct": 0.5,
    },
    "NORMAL": {
        "rsi_ob": 70, "rsi_os": 30, "pivot_lookback": 7, "hold_period": 5,
        "min_expected_pf": 1.20, "preferred_strategy": "golden_long",
        "preferred_dir": None, "max_position_size_pct": 1.0,
    },
}


def detect_market_regime(df: pd.DataFrame) -> str:
    """Piaci rezsim a Fazis-3 keplet szerint.

    HIGH_VOL ha az ATR a 90 napos atlag 1.5x-e felett van;
    BULL ha 30d hozam > 10% es SMA50 > SMA200;
    BEAR ha 30d hozam < -10% VAGY SMA50 < SMA200;
    SIDEWAYS ha |30d hozam| < 5%; kulonben NORMAL.
    """
    if df is None or len(df) < 30:
        return "NORMAL"
    close = df["close"]
    price_now = float(close.iloc[-1])
    price_30 = float(close.iloc[-30])
    returns_30d = (price_now - price_30) / price_30 * 100 if price_30 else 0

    tr = (df["high"] - df["low"])
    atr = tr.rolling(14).mean()
    atr_now = float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else 0
    win = min(90, len(atr.dropna()))
    atr_avg = float(atr.iloc[-win:].mean()) if win > 0 else atr_now
    atr_ratio = atr_now / atr_avg if atr_avg > 0 else 1.0

    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(min(200, len(close) - 1)).mean().iloc[-1]
    sma_bull = pd.notna(sma50) and pd.notna(sma200) and sma50 > sma200
    sma_bear = pd.notna(sma50) and pd.notna(sma200) and sma50 < sma200

    if atr_ratio > 1.5:
        return "HIGH_VOL"
    if returns_30d > 10 and sma_bull:
        return "BULL"
    if returns_30d < -10 or sma_bear:
        return "BEAR"
    if abs(returns_30d) < 5:
        return "SIDEWAYS"
    return "NORMAL"


def regime_settings(regime: str) -> dict:
    return REGIME_SETTINGS.get(regime, REGIME_SETTINGS["NORMAL"])


# ---------------------------------------------------------------------------
# 3. ADAPTIV SULYOK (kategoriankent)
# ---------------------------------------------------------------------------
ADAPTIVE_WEIGHTS = {
    "pivot": 2.4, "rsi": 1.52, "streak": 1.5, "volume": 1.04,
    "calendar": 0.8, "session": 0.8, "moon": 0.96, "planets": 1.2,
}
_WEIGHT_MIN, _WEIGHT_MAX = 0.5, 3.0


def adjust_weight(category: str, pf: float) -> float:
    """Backteszt-alapu sulymodositas: PF>1.3 -> +0.1; PF<1.0 -> -0.1."""
    w = ADAPTIVE_WEIGHTS.get(category, 1.0)
    if pf > 1.3:
        w += 0.1
    elif pf < 1.0:
        w -= 0.1
    return round(max(_WEIGHT_MIN, min(_WEIGHT_MAX, w)), 2)


def weights_line() -> str:
    w = ADAPTIVE_WEIGHTS
    return (f"Piv:{w['pivot']} | RSI:{w['rsi']} | Strk:{w['streak']} | Vol:{w['volume']}\n"
            f"  Cal:{w['calendar']} | Moon:{w['moon']} | Ses:{w['session']} | Plan:{w['planets']}")


# ---------------------------------------------------------------------------
# 6. ADVISOR MODE
# ---------------------------------------------------------------------------
ADVISOR_MODES = {
    "CONSERVATIVE": {"min_score": 130, "min_data_quality": 90, "min_sample_size": 300,
                     "max_atr": 7, "min_rr": 2.5, "max_positions": 2},
    "NORMAL": {"min_score": 100, "min_data_quality": 70, "min_sample_size": 100,
               "max_atr": 10, "min_rr": 2.0, "max_positions": 3},
    "AGGRESSIVE": {"min_score": 80, "min_data_quality": 50, "min_sample_size": 50,
                   "max_atr": 15, "min_rr": 1.5, "max_positions": 5},
}


def advisor_settings(mode: str) -> dict:
    return ADVISOR_MODES.get((mode or "NORMAL").upper(), ADVISOR_MODES["NORMAL"])


# ---------------------------------------------------------------------------
# 7. SESSION-AWARE
# ---------------------------------------------------------------------------
def get_session(dt: datetime | None = None) -> str:
    """Aktualis kereskedesi szesszio UTC ora alapjan."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    h = dt.hour
    if 0 <= h < 8:
        return "ASIA"
    if 8 <= h < 16:
        return "EU"
    return "US"


# ---------------------------------------------------------------------------
# 5. VOLATILITY FORECAST
# ---------------------------------------------------------------------------
def volatility_forecast(atr_pct: float, hold_period: int = 5) -> float:
    """Varhato mozgas a hold period alatt: ATR% * sqrt(nap)."""
    return atr_pct * math.sqrt(max(1, hold_period))


# ---------------------------------------------------------------------------
# 4. MULTI-HORIZON PROFIT FACTOR
# ---------------------------------------------------------------------------
def multi_horizon_pf(df: pd.DataFrame) -> dict:
    """Piac szintu profit factor 1/3/5 napos holdra a teljes historian."""
    close = df["close"].values
    out = {}
    for h in (1, 3, 5):
        rets = [(close[i + h] - close[i]) / close[i] for i in range(len(close) - h)]
        arr = np.array(rets)
        gw = arr[arr > 0].sum()
        gl = abs(arr[arr < 0].sum())
        out[h] = round(gw / gl, 2) if gl > 0 else 0.0
    out["best_horizon"] = max((1, 3, 5), key=lambda h: out[h])
    return out


def pf_line(pf: dict) -> str:
    return f"PF[1d={pf.get(1, 0):.2f} 3d={pf.get(3, 0):.2f} 5d={pf.get(5, 0):.2f}]"


# ---------------------------------------------------------------------------
# 9. HOLDFAZIS (opcionalis, ephem nelkul kozelitessel)
# ---------------------------------------------------------------------------
def get_moon_phase(dt: datetime | None = None) -> tuple[str, float]:
    """Holdfazis megvilagitottsag %. Probal ephem-et, kulonben kozelit."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    try:
        import ephem
        m = ephem.Moon()
        m.compute(dt)
        illum = m.moon_phase * 100
    except Exception:
        # Synodikus honap kozelites (ref: 2000-01-06 18:14 UTC ujhold)
        ref = datetime(2000, 1, 6, 18, 14, tzinfo=timezone.utc)
        days = (dt - ref).total_seconds() / 86400
        syn = 29.530588853
        phase = (days % syn) / syn
        illum = (1 - math.cos(2 * math.pi * phase)) / 2 * 100
    if illum > 90:
        return "FULL", illum
    if illum < 10:
        return "NEW", illum
    return "NORMAL", illum


# ---------------------------------------------------------------------------
# 8. CORRELATION TRACKING (modell vs valosag, Pearson)
# ---------------------------------------------------------------------------
def _corr_path(symbol: str) -> str:
    return f"{CORR_DIR}/{symbol}_preds.json"


def _load_preds(symbol: str) -> list:
    p = _corr_path(symbol)
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            return []
    return []


def record_prediction(symbol: str, predicted_up_pct: float, price: float,
                      dt: datetime | None = None) -> None:
    """Predikcio naplozasa kesobbi korrelacio-szamitashoz."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    os.makedirs(CORR_DIR, exist_ok=True)
    preds = _load_preds(symbol)
    today = dt.strftime("%Y-%m-%d")
    if any(p.get("date") == today for p in preds):
        return  # naponta egy bejegyzes
    preds.append({"date": today, "predicted_up": round(predicted_up_pct, 1),
                  "price": price, "actual": None})
    with open(_corr_path(symbol), "w") as f:
        json.dump(preds[-500:], f, indent=2)


def resolve_and_correlate(symbol: str, df: pd.DataFrame,
                          horizon: int = 5) -> dict:
    """Megoldja a fuggo predikciokat (5 nap mulva) es Pearson korrelaciot szamol.

    Visszaad: corr (%), q (megoldott minta szam), label.
    """
    preds = _load_preds(symbol)
    if not preds or df is None or len(df) < horizon + 1:
        return {"corr": None, "q": 0, "label": "n/a"}

    # Datum -> zaroar map
    idx = pd.to_datetime(df.index)
    price_by_date = {d.strftime("%Y-%m-%d"): float(c)
                     for d, c in zip(idx, df["close"].values)}
    dates_sorted = [d.strftime("%Y-%m-%d") for d in idx]

    changed = False
    for p in preds:
        if p.get("actual") is not None:
            continue
        d = p["date"]
        if d in dates_sorted:
            i = dates_sorted.index(d)
            if i + horizon < len(dates_sorted):
                base = price_by_date[d]
                fut = price_by_date[dates_sorted[i + horizon]]
                p["actual"] = round((fut - base) / base * 100, 2) if base else 0
                changed = True
    if changed:
        with open(_corr_path(symbol), "w") as f:
            json.dump(preds, f, indent=2)

    resolved = [(p["predicted_up"], p["actual"]) for p in preds
                if p.get("actual") is not None]
    if len(resolved) < 5:
        return {"corr": None, "q": len(resolved), "label": "tul keves minta"}

    pred_arr = np.array([r[0] for r in resolved])
    act_arr = np.array([r[1] for r in resolved])
    if pred_arr.std() == 0 or act_arr.std() == 0:
        return {"corr": 0, "q": len(resolved), "label": "nincs varancia"}
    try:
        from scipy.stats import pearsonr
        corr = float(pearsonr(pred_arr, act_arr)[0]) * 100
    except Exception:
        corr = float(np.corrcoef(pred_arr, act_arr)[0, 1]) * 100

    if corr > 50:
        label = "kivalo modell"
    elif corr > 25:
        label = "jo modell"
    elif corr >= 0:
        label = "gyenge modell"
    else:
        label = "HIBAS modell - ne kereskedj"
    return {"corr": round(corr), "q": len(resolved), "label": label}


# ---------------------------------------------------------------------------
# 11. HEADER OUTPUT
# ---------------------------------------------------------------------------
def advanced_header_lines(symbol: str, regime: str, advisor_mode: str,
                          atr_pct: float, mtf_score: int = 0,
                          corr: dict | None = None, df=None) -> list:
    """Fazis-3 fejlec sorok (lista) az event forecast cime ala."""
    sett = regime_settings(regime)
    session = get_session()
    moon_label, moon_pct = get_moon_phase()
    hold = sett["hold_period"]
    vol = volatility_forecast(atr_pct, hold)

    mtf_label = ("STRONG BULL" if mtf_score >= 3 else "BULL" if mtf_score > 0 else
                 "STRONG BEAR" if mtf_score <= -3 else "BEAR" if mtf_score < 0 else "NEUTRAL")

    lines = [
        f"  ADV: {advisor_mode} | Regime: {regime} | MTF: {mtf_score:+d} ({mtf_label})",
        f"  Session: {session} | Moon: {moon_pct:.1f}% {moon_label}",
        f"  Expected Vol: ±{vol:.2f}% / {hold}d",
        f"  Auto-Opt: {regime} | RSI OB:{sett['rsi_ob']}/OS:{sett['rsi_os']} | H={hold}d",
    ]
    if df is not None:
        pf = multi_horizon_pf(df)
        lines.append(f"  {pf_line(pf)} | LB={len(df)}d")
    if corr and corr.get("corr") is not None:
        lines.append(f"  Correlation: Corr: {corr['corr']:+d}% | Q: {corr['q']} ({corr['label']})")
    lines.append("")
    lines.append("  Adaptive Weights:")
    lines.append(f"  {weights_line()}")
    return lines


def print_advanced_header(symbol: str, regime: str, advisor_mode: str,
                          atr_pct: float, mtf_score: int = 0,
                          corr: dict | None = None, df=None) -> None:
    """Fazis-3 fejlec kiiras (advanced_header_lines wrapper)."""
    for ln in advanced_header_lines(symbol, regime, advisor_mode, atr_pct,
                                    mtf_score, corr, df):
        print(ln)
