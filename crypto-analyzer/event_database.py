"""
Statistical Event Database modul
================================
Historikus eventek win rate trackelese. Minden eventre kiszamolja a
forward returnt (1d, 3d, 5d) es statisztikat vezet.

A coin OHLCV adatabol dolgozik (nincs kulon API az alapstatisztikahoz).
Composite scoring az aktiv eventekbol -> UP%/DN% elorejelzes.
"""

import json
import os
import numpy as np
import pandas as pd

EVENT_DB_DIR = "results/event_db"

# Adaptiv sulyok eventtipusonkent (backteszttel optimalizalhato)
ADAPTIVE_WEIGHTS = {
    "RSI_OS": 1.52, "RSI_OS_36": 1.52, "RSI_OB": 1.52, "RSI_OB_64": 1.52,
    "Vol_Spike": 1.04,
    "MACD_Bull_Cross": 1.3, "MACD_Bear_Cross": 1.3,
    "Golden_Cross": 2.4, "Death_Cross": 2.4,
    "BB_Break_Up": 1.2, "BB_Break_Down": 1.2,
    "Friday": 0.8, "Saturday": 0.8, "Sunday": 0.8, "Month_End": 0.8,
    "No_Streak": 1.5,
}


def detect_regime(df: pd.DataFrame) -> str:
    """Piaci rezsim: BULL / BEAR / VOLATILE / NORMAL."""
    if len(df) < 30:
        return "NORMAL"
    close = df["close"]
    tr = (df["high"] - df["low"])
    atr = tr.rolling(14).mean()
    if atr.iloc[-30:].mean() > 0 and atr.iloc[-1] > atr.iloc[-30:].mean() * 1.3:
        return "VOLATILE"
    chg30 = (close.iloc[-1] - close.iloc[-30]) / close.iloc[-30] * 100
    if chg30 < -10:
        return "BEAR"
    if chg30 > 10:
        return "BULL"
    return "NORMAL"


def _regime_rsi_thresholds(regime: str) -> tuple[float, float]:
    """RSI OB/OS kuszobok rezsim szerint."""
    if regime == "BEAR":
        return 65, 25
    if regime == "VOLATILE":
        return 75, 20
    return 70, 30  # NORMAL / BULL


def classify_edge(wr5: float, avg5: float) -> str:
    """BIAS vs EDGE megkulonboztetes: csak szignifikans elony szamit."""
    if wr5 >= 55 and avg5 > 0.5:
        return "UP"
    if wr5 <= 45 and avg5 < -0.5:
        return "DOWN"
    return "NEUTRAL"


def _forward_returns(close: pd.Series, idx: int, horizons=(1, 3, 5)) -> dict:
    """Forward return % az adott indextol."""
    out = {}
    base = close.iloc[idx]
    for h in horizons:
        if idx + h < len(close):
            out[h] = (close.iloc[idx + h] - base) / base * 100
        else:
            out[h] = None
    return out


def _detect_events(df: pd.DataFrame, idx: int) -> list[str]:
    """Megnezi melyik eventek aktivak az adott napon."""
    events = []
    last = df.iloc[idx]
    close = df["close"]

    rsi = last.get("rsi")
    if rsi is not None and pd.notna(rsi):
        if rsi < 30:
            events.append("RSI_OS")
        elif rsi < 36:
            events.append("RSI_OS_36")
        elif rsi > 70:
            events.append("RSI_OB")
        elif rsi > 64:
            events.append("RSI_OB_64")

    # Vol spike (RVOL > 2)
    vol = df["volume"].fillna(0)
    if idx >= 20:
        avg = vol.iloc[idx-20:idx].mean()
        if avg > 0 and vol.iloc[idx] > avg * 2:
            events.append("Vol_Spike")

    # MACD cross
    mh = df.get("macd_hist")
    if mh is not None and idx >= 1 and pd.notna(mh.iloc[idx]) and pd.notna(mh.iloc[idx-1]):
        if mh.iloc[idx] > 0 and mh.iloc[idx-1] <= 0:
            events.append("MACD_Bull_Cross")
        elif mh.iloc[idx] < 0 and mh.iloc[idx-1] >= 0:
            events.append("MACD_Bear_Cross")

    # Golden/Death cross
    sma50 = df.get("sma_50")
    sma200 = df.get("sma_200")
    if (sma50 is not None and sma200 is not None and idx >= 1 and
            pd.notna(sma50.iloc[idx]) and pd.notna(sma200.iloc[idx]) and
            pd.notna(sma50.iloc[idx-1]) and pd.notna(sma200.iloc[idx-1])):
        if sma50.iloc[idx] > sma200.iloc[idx] and sma50.iloc[idx-1] <= sma200.iloc[idx-1]:
            events.append("Golden_Cross")
        elif sma50.iloc[idx] < sma200.iloc[idx] and sma50.iloc[idx-1] >= sma200.iloc[idx-1]:
            events.append("Death_Cross")

    # BB squeeze break
    bu = df.get("bb_upper")
    bl = df.get("bb_lower")
    if bu is not None and bl is not None and pd.notna(bu.iloc[idx]):
        if close.iloc[idx] > bu.iloc[idx]:
            events.append("BB_Break_Up")
        elif close.iloc[idx] < bl.iloc[idx]:
            events.append("BB_Break_Down")

    # Naptari eventek
    date = df.index[idx]
    if hasattr(date, "weekday"):
        wd = date.weekday()
        if wd == 4:
            events.append("Friday")
        elif wd == 5:
            events.append("Saturday")
        elif wd == 6:
            events.append("Sunday")
        # Honap vege
        if date.day >= 28:
            events.append("Month_End")

    # No streak (semleges nap)
    if not events:
        events.append("No_Streak")

    return events


def build_event_stats(df: pd.DataFrame) -> dict:
    """Minden eventre kiszamolja a forward return statisztikat."""
    event_data = {}  # event -> list of forward returns

    n = len(df)
    for idx in range(50, n - 5):  # min 50 nap warmup, 5 nap forward
        events = _detect_events(df, idx)
        fwd = _forward_returns(df["close"], idx)
        for ev in events:
            if ev not in event_data:
                event_data[ev] = {1: [], 3: [], 5: []}
            for h in (1, 3, 5):
                if fwd[h] is not None:
                    event_data[ev][h].append(fwd[h])

    # Statisztika
    stats = {}
    for ev, horizons in event_data.items():
        ev_stat = {"n": len(horizons[5])}
        if ev_stat["n"] < 5:
            continue
        for h in (1, 3, 5):
            rets = horizons[h]
            if not rets:
                continue
            arr = np.array(rets)
            wins = (arr > 0).sum()
            wr = wins / len(arr) * 100
            avg = arr.mean()
            gross_win = arr[arr > 0].sum() if (arr > 0).any() else 0
            gross_loss = abs(arr[arr < 0].sum()) if (arr < 0).any() else 1
            pf = gross_win / gross_loss if gross_loss > 0 else 0
            ev_stat[f"WR_{h}d"] = round(wr, 1)
            ev_stat[f"AVG_{h}d"] = round(avg, 2)
            ev_stat[f"PF_{h}d"] = round(pf, 2)
            ev_stat[f"BEST_{h}d"] = round(arr.max(), 1)
            ev_stat[f"WORST_{h}d"] = round(arr.min(), 1)
        # Data quality: hany minta / 500 cel
        ev_stat["quality"] = min(100, round(ev_stat["n"] / 500 * 100))
        stats[ev] = ev_stat

    return stats


def save_event_stats(symbol: str, stats: dict) -> None:
    os.makedirs(EVENT_DB_DIR, exist_ok=True)
    with open(f"{EVENT_DB_DIR}/{symbol}_events.json", "w") as f:
        json.dump(stats, f, indent=2)


def load_event_stats(symbol: str) -> dict | None:
    path = f"{EVENT_DB_DIR}/{symbol}_events.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def analyze_events(df: pd.DataFrame, symbol: str, stats: dict | None = None) -> dict:
    """Aktiv eventek + composite forecast."""
    if stats is None:
        stats = build_event_stats(df)

    regime = detect_regime(df)
    # Aktiv eventek MA
    active = _detect_events(df, len(df) - 1)
    active_stats = []
    up_score = 0
    dn_score = 0
    total_weight = 0
    quality_sum = 0

    for ev in active:
        if ev not in stats:
            continue
        st = stats[ev]
        wr5 = st.get("WR_5d", 50)
        avg5 = st.get("AVG_5d", 0)
        n = st.get("n", 0)
        qual = st.get("quality", 0)
        # Csak 50%+ data quality esemenyek szamitanak a vegso scoringba
        edge = classify_edge(wr5, avg5)
        # Adaptiv suly: quality * eventtipus suly
        aw = ADAPTIVE_WEIGHTS.get(ev, 1.0)
        weight = (qual / 100) * aw
        # Csak ha EDGE szignifikans (nem csak BIAS)
        if edge == "UP":
            up_score += (wr5 - 50) * weight
        elif edge == "DOWN":
            dn_score += (50 - wr5) * weight
        total_weight += weight
        quality_sum += qual
        active_stats.append({
            "event": ev, "WR_5d": wr5, "AVG_5d": avg5,
            "n": n, "quality": qual, "edge": edge, "weight": round(aw, 2),
        })

    # Composite UP%/DN%
    if up_score + dn_score > 0:
        up_pct = up_score / (up_score + dn_score) * 100
        dn_pct = 100 - up_pct
    else:
        up_pct = dn_pct = 50

    if up_pct > 65:
        bias = "UP"
        strength = "VERY STRONG" if up_pct > 75 else "STRONG"
    elif dn_pct > 65:
        bias = "DOWN"
        strength = "VERY STRONG" if dn_pct > 75 else "STRONG"
    else:
        bias = "NEUTRAL"
        strength = "WEAK"

    avg_quality = quality_sum / len(active_stats) if active_stats else 0

    # Edge score: max +10 (irany + erosseg alapjan)
    edge = "NEUTRAL"
    edge_score = 0
    if bias == "UP":
        edge = "LONG"
        edge_score = 10 if strength == "VERY STRONG" else (7 if strength == "STRONG" else 3)
    elif bias == "DOWN":
        edge = "SHORT"
        edge_score = 10 if strength == "VERY STRONG" else (7 if strength == "STRONG" else 3)

    # Forecast confidence: csak akkor ervenyes ha eleg minoseg
    confidence = "HIGH" if avg_quality > 80 else ("MEDIUM" if avg_quality > 60 else "LOW")
    # Veto: ha alacsony a konfidencia, az edge_score felezodik
    if avg_quality < 60:
        edge_score = edge_score // 2

    return {
        "active_stats": active_stats, "up_pct": round(up_pct, 1),
        "dn_pct": round(dn_pct, 1), "bias": bias, "strength": strength,
        "avg_quality": round(avg_quality), "edge": edge, "edge_score": edge_score,
        "n_active": len(active_stats), "regime": regime, "confidence": confidence,
    }


def print_event_forecast(symbol: str, ev: dict, price: float = 0, atr_pct: float = 0) -> None:
    """Event forecast kiiras."""
    print(f"\n STATISTICAL EVENT FORECAST — {symbol}")
    print(f"{'-' * 60}")
    if not ev["active_stats"]:
        print("  Nincs eleg adat event elemzeshez.")
        return

    print(f"  Rezsim: {ev.get('regime', 'NORMAL')} | Confidence: {ev.get('confidence', '?')}")
    print(f"  ACTIVE EVENTS ({ev['n_active']}):")
    for st in ev["active_stats"]:
        sign = "+" if st["AVG_5d"] >= 0 else ""
        edge = st.get("edge", "?")
        w = st.get("weight", 1.0)
        print(f"    {st['event']:<16} WR-5d {st['WR_5d']:.1f}%, "
              f"AVG {sign}{st['AVG_5d']:.2f}% [{edge}, w{w}] (n={st['n']}, q{st['quality']}%)")

    print(f"\n  COMPOSITE: {ev['n_active']} active | Bias: {ev['bias']} ({ev['strength']}) | Q: {ev['avg_quality']}/100")
    print(f"  UP: {ev['up_pct']:.1f}% | DN: {ev['dn_pct']:.1f}%")

    if price > 0 and atr_pct > 0:
        move = price * atr_pct / 100 * 2.2  # ~5 napos varhato mozgas
        print(f"  FORECAST 5-day: UP ${price + move:,.4g} / DN ${price - move:,.4g}")

    edge_dir = "BULLISH" if ev["edge"] == "LONG" else ("BEARISH" if ev["edge"] == "SHORT" else "NEUTRAL")
    print(f"\n  EVENT-BASED EDGE: {edge_dir} ({ev['up_pct']:.1f}% vs {ev['dn_pct']:.1f}%) [+{ev['edge_score']} pont]")
