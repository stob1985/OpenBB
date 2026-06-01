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
    "BB_Break_Up": 1.2, "BB_Break_Down": 1.2, "BB_Squeeze_Break": 1.3,
    "Kumo_Breakout": 1.4, "Pivot_Touch": 2.4,
    "Win_Streak": 1.5, "Loss_Streak": 1.5, "No_Streak": 1.5,
    "Friday": 0.8, "Saturday": 0.8, "Sunday": 0.8, "Month_End": 0.8,
}

# Olvashato event nevek a tablazathoz
EVENT_LABELS = {
    "RSI_OS": "RSI OS", "RSI_OS_36": "RSI OS~", "RSI_OB": "RSI OB", "RSI_OB_64": "RSI OB~",
    "Vol_Spike": "Vol Spike", "MACD_Bull_Cross": "MACD Bull", "MACD_Bear_Cross": "MACD Bear",
    "Golden_Cross": "Golden Cross", "Death_Cross": "Death Cross",
    "BB_Break_Up": "BB Break Up", "BB_Break_Down": "BB Break Dn",
    "BB_Squeeze_Break": "BB Squeeze", "Kumo_Breakout": "Kumo Break",
    "Pivot_Touch": "Pivot", "Win_Streak": "Win Streak", "Loss_Streak": "Loss Streak",
    "No_Streak": "No Streak", "Friday": "Friday", "Saturday": "Saturday",
    "Sunday": "Sunday", "Month_End": "Month End",
}


def _ensure_event_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Kiegeszito oszlopok az event detektalashoz (Kumo, pivot, BB szelesseg).

    Egyszer szamoljuk a teljes mintara, hogy a per-nap detektalas gyors legyen.
    """
    df = df.copy()
    high, low, close = df["high"], df["low"], df["close"]

    # Ichimoku Kumo (felho) hatarok
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    df["kumo_top"] = pd.concat([span_a, span_b], axis=1).max(axis=1)
    df["kumo_bottom"] = pd.concat([span_a, span_b], axis=1).min(axis=1)

    # BB szelesseg (squeeze detektalashoz)
    if "bb_upper" in df.columns and "bb_lower" in df.columns:
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / close.replace(0, np.nan)

    # Pivot szintek (lookback=10)
    df["pivot_high"] = high.rolling(10).max()
    df["pivot_low"] = low.rolling(10).min()

    # Streak iranya
    df["_up_day"] = (close > close.shift()).astype(int)

    return df


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


def _detect_events(df: pd.DataFrame, idx: int,
                   rsi_ob: float = 70, rsi_os: float = 30) -> list[str]:
    """Megnezi melyik eventek aktivak az adott napon.

    rsi_ob / rsi_os: rezsim-adaptiv RSI kuszobok (_regime_rsi_thresholds).
    A "kozeli" savok (OS_36 / OB_64) a kuszob koruli 6 pontos savot jelolik.
    """
    events = []
    last = df.iloc[idx]
    close = df["close"]

    rsi = last.get("rsi")
    if rsi is not None and pd.notna(rsi):
        if rsi < rsi_os:
            events.append("RSI_OS")
        elif rsi < rsi_os + 6:
            events.append("RSI_OS_36")
        elif rsi > rsi_ob:
            events.append("RSI_OB")
        elif rsi > rsi_ob - 6:
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

    # BB break + squeeze break
    bu = df.get("bb_upper")
    bl = df.get("bb_lower")
    bw = df.get("bb_width")
    if bu is not None and bl is not None and pd.notna(bu.iloc[idx]):
        broke_up = close.iloc[idx] > bu.iloc[idx]
        broke_dn = close.iloc[idx] < bl.iloc[idx]
        if broke_up:
            events.append("BB_Break_Up")
        elif broke_dn:
            events.append("BB_Break_Down")
        # Squeeze break: a BB szelesseg 20 napos minimumon volt ES most kitort
        if bw is not None and idx >= 20 and (broke_up or broke_dn):
            wmin = bw.iloc[idx-20:idx].min()
            if pd.notna(wmin) and pd.notna(bw.iloc[idx-1]) and bw.iloc[idx-1] <= wmin * 1.05:
                events.append("BB_Squeeze_Break")

    # Kumo (Ichimoku felho) breakout: ar 3%+ attori a felho szelet
    kt = df.get("kumo_top")
    kb = df.get("kumo_bottom")
    if kt is not None and kb is not None and idx >= 1 and pd.notna(kt.iloc[idx]) and pd.notna(kb.iloc[idx]):
        c_now, c_prev = close.iloc[idx], close.iloc[idx-1]
        if c_prev <= kt.iloc[idx] and c_now > kt.iloc[idx] * 1.03:
            events.append("Kumo_Breakout")
        elif c_prev >= kb.iloc[idx] and c_now < kb.iloc[idx] * 0.97:
            events.append("Kumo_Breakout")

    # Pivot touch (lookback=10): ar a 10-napos pivot szinthez er
    ph = df.get("pivot_high")
    pl = df.get("pivot_low")
    if ph is not None and pl is not None and pd.notna(ph.iloc[idx]):
        c_now = close.iloc[idx]
        if abs(c_now - ph.iloc[idx]) / ph.iloc[idx] < 0.01 or \
           (pl.iloc[idx] > 0 and abs(c_now - pl.iloc[idx]) / pl.iloc[idx] < 0.01):
            events.append("Pivot_Touch")

    # Streak: 3+ egymas utani up/down nap
    ud = df.get("_up_day")
    if ud is not None and idx >= 3:
        last3 = ud.iloc[idx-2:idx+1].values
        if last3.sum() == 3:
            events.append("Win_Streak")
        elif last3.sum() == 0:
            events.append("Loss_Streak")

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

    # No streak (semleges nap - sem streak, sem mas erdemi event)
    if not any(e in events for e in (
            "Win_Streak", "Loss_Streak", "RSI_OS", "RSI_OB", "RSI_OS_36",
            "RSI_OB_64", "Vol_Spike", "MACD_Bull_Cross", "MACD_Bear_Cross",
            "Golden_Cross", "Death_Cross", "BB_Break_Up", "BB_Break_Down",
            "Kumo_Breakout", "Pivot_Touch")):
        events.append("No_Streak")

    return events


def _quality_score(n: int) -> int:
    """DB-QUAL: 100 ha n>=500; n/500*100 ha n<500; 0 ha n<30."""
    if n < 30:
        return 0
    return min(100, round(n / 500 * 100))


def _bias_label(wr5: float) -> str:
    """BIAS: UP ha WR_5d>53%, DOWN ha <47%, kulonben semleges."""
    if wr5 > 53:
        return "UP"
    if wr5 < 47:
        return "DOWN"
    return "—"


def build_event_stats(df: pd.DataFrame) -> dict:
    """Minden eventre kiszamolja a forward return statisztikat (max 500 sample)."""
    df = _ensure_event_columns(df)
    close = df["close"]
    event_data = {}  # event -> {1:[], 3:[], 5:[], "mdd3":[]}

    # Rezsim-adaptiv RSI kuszobok az egesz mintara
    regime = detect_regime(df)
    rsi_ob, rsi_os = _regime_rsi_thresholds(regime)

    n = len(df)
    for idx in range(50, n - 5):  # min 50 nap warmup, 5 nap forward
        events = _detect_events(df, idx, rsi_ob=rsi_ob, rsi_os=rsi_os)
        fwd = _forward_returns(close, idx)
        # 3 napos max drawdown (path-alapu)
        base = close.iloc[idx]
        window = close.iloc[idx+1:idx+4]
        mdd3 = float((window.min() - base) / base * 100) if len(window) and base else 0.0
        for ev in events:
            if ev not in event_data:
                event_data[ev] = {1: [], 3: [], 5: [], "mdd3": []}
            for h in (1, 3, 5):
                if fwd[h] is not None:
                    event_data[ev][h].append(fwd[h])
            event_data[ev]["mdd3"].append(mdd3)

    # Statisztika
    stats = {}
    for ev, horizons in event_data.items():
        # Max 500 legfrissebb sample (a lista kronologikus, a vege a legujabb)
        for key in (1, 3, 5, "mdd3"):
            horizons[key] = horizons[key][-500:]
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

        # MDD 3d (atlagos legrosszabb visszaeses)
        mdd_arr = np.array(horizons["mdd3"]) if horizons["mdd3"] else np.array([0.0])
        ev_stat["MDD_3d"] = round(float(mdd_arr.mean()), 2)

        # Legutobbi 5 jel WR + atlag (recency)
        recent5 = np.array(horizons[5][-5:]) if len(horizons[5]) >= 1 else np.array([0.0])
        ev_stat["last5_wr"] = round(float((recent5 > 0).sum() / len(recent5) * 100), 0)
        ev_stat["last5_avg"] = round(float(recent5.mean()), 2)

        # Long-term WR (a regebbi jelek, az utolso 5 nelkul)
        older = horizons[5][:-5] if len(horizons[5]) > 5 else horizons[5]
        older_arr = np.array(older)
        ev_stat["long_term_wr"] = round(float((older_arr > 0).sum() / len(older_arr) * 100), 0)

        # Data quality + bias + edge + expect return
        ev_stat["quality"] = _quality_score(ev_stat["n"])
        wr5 = ev_stat.get("WR_5d", 50)
        avg5 = ev_stat.get("AVG_5d", 0)
        ev_stat["bias"] = _bias_label(wr5)
        ev_stat["edge"] = classify_edge(wr5, avg5)
        ev_stat["expect_return"] = ev_stat.get("AVG_3d", 0)
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


def analyze_events(df: pd.DataFrame, symbol: str, stats: dict | None = None,
                   model_up: float | None = None) -> dict:
    """Aktiv eventek + blended composite forecast.

    model_up: opcionalis technikai modell UP% (0-100). Ha megadva,
    a blended forecast (model + DB + composite) / 3 keplettel szamol.
    """
    if stats is None:
        stats = build_event_stats(df)

    df = _ensure_event_columns(df)
    regime = detect_regime(df)
    rsi_ob, rsi_os = _regime_rsi_thresholds(regime)
    # Aktiv eventek MA
    active = _detect_events(df, len(df) - 1, rsi_ob=rsi_ob, rsi_os=rsi_os)
    active_stats = []
    up_score = 0
    dn_score = 0
    total_weight = 0
    quality_sum = 0
    wr5_list = []  # db_up komponenshez
    n_up_bias = 0
    n_dn_bias = 0

    for ev in active:
        if ev not in stats:
            continue
        st = stats[ev]
        wr5 = st.get("WR_5d", 50)
        avg5 = st.get("AVG_5d", 0)
        n = st.get("n", 0)
        qual = st.get("quality", 0)
        edge = classify_edge(wr5, avg5)
        bias_l = _bias_label(wr5)
        # Minden aktiv event megjelenik a tablazatban
        active_stats.append({
            "event": ev, "label": EVENT_LABELS.get(ev, ev),
            "WR_5d": wr5, "AVG_5d": avg5, "MDD_3d": st.get("MDD_3d", 0),
            "n": n, "quality": qual, "edge": edge, "bias": bias_l,
            "weight": round(ADAPTIVE_WEIGHTS.get(ev, 1.0), 2),
            "counted": qual >= 50,
        })
        # DE csak 50%+ data quality esemenyek szamitanak a scoringba
        if qual < 50:
            continue
        if bias_l == "UP":
            n_up_bias += 1
        elif bias_l == "DOWN":
            n_dn_bias += 1
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
        wr5_list.append(wr5)

    # Composite (edge-sulyozott) UP%/DN%
    if up_score + dn_score > 0:
        composite_up = up_score / (up_score + dn_score) * 100
    else:
        composite_up = 50

    # DB up: aktiv eventek atlagos WR_5d-je (nyers iranyvaloszinuseg)
    db_up = float(np.mean(wr5_list)) if wr5_list else 50

    # Blended model: (model + db + composite) / 3
    if model_up is not None:
        blended_up = (model_up + db_up + composite_up) / 3
    else:
        blended_up = (db_up + composite_up) / 2
    blended_dn = 100 - blended_up

    up_pct = blended_up
    dn_pct = blended_dn

    if up_pct > 65:
        bias = "UP"
        strength = "VERY STRONG" if up_pct > 75 else "STRONG"
    elif dn_pct > 65:
        bias = "DOWN"
        strength = "VERY STRONG" if dn_pct > 75 else "STRONG"
    elif up_pct > 53:
        bias = "UP"
        strength = "WEAK"
    elif dn_pct > 53:
        bias = "DOWN"
        strength = "WEAK"
    else:
        bias = "NEUTRAL"
        strength = "WEAK"

    n_counted = len(wr5_list)
    avg_quality = quality_sum / n_counted if n_counted else 0

    # Edge score: max +25 (irany + erosseg alapjan) — kombinalt dontesi motor
    # A spec szerinti eros trigger: UP/DN >=65% ES Q>=80 ES min 3 azonos bias
    edge = "NEUTRAL"
    edge_score = 0
    strong_trigger = (avg_quality >= 80 and
                      ((bias == "UP" and up_pct >= 65 and n_up_bias >= 3) or
                       (bias == "DOWN" and dn_pct >= 65 and n_dn_bias >= 3)))
    if bias == "UP":
        edge = "LONG"
        edge_score = 25 if strong_trigger else (18 if strength in ("STRONG", "VERY STRONG") else 8)
    elif bias == "DOWN":
        edge = "SHORT"
        edge_score = 25 if strong_trigger else (18 if strength in ("STRONG", "VERY STRONG") else 8)

    # Forecast confidence: label + numerikus %
    confidence = "HIGH" if avg_quality > 80 else ("MEDIUM" if avg_quality > 60 else "LOW")
    conf_pct = min(95, round(50 + (avg_quality / 100) * 25 + (abs(up_pct - 50) / 50) * 20))
    # Veto: ha alacsony a konfidencia, az edge_score felezodik
    if avg_quality < 60:
        edge_score = edge_score // 2

    return {
        "active_stats": active_stats, "up_pct": round(up_pct, 1),
        "dn_pct": round(dn_pct, 1), "bias": bias, "strength": strength,
        "avg_quality": round(avg_quality), "edge": edge, "edge_score": edge_score,
        "n_active": len(active_stats), "regime": regime, "confidence": confidence,
        "conf_pct": conf_pct, "db_up": round(db_up, 1),
        "composite_up": round(composite_up, 1),
        "model_up": round(model_up, 1) if model_up is not None else None,
        "n_up_bias": n_up_bias, "n_dn_bias": n_dn_bias,
        "strong_trigger": strong_trigger,
    }


def print_event_forecast(symbol: str, ev: dict, price: float = 0, atr_pct: float = 0,
                         adv_lines: list | None = None) -> None:
    """Event forecast kiiras (tablazatos).

    adv_lines: opcionalis Fazis-3 fejlec sorok a cim alatt (advisor/regime/session...).
    """
    print(f"\n STATISTICAL EVENT FORECAST — {symbol}")
    print(f"{'=' * 72}")
    if adv_lines:
        for ln in adv_lines:
            print(ln)
        print(f"{'-' * 72}")
    if not ev["active_stats"]:
        print("  Nincs eleg adat event elemzeshez.")
        return

    up_b, dn_b = ev.get("n_up_bias", 0), ev.get("n_dn_bias", 0)
    print(f"  Rezsim: {ev.get('regime', 'NORMAL')} | Confidence: {ev.get('confidence', '?')}")
    print(f"  ACTIVE EVENTS ({ev['n_active']} active · {up_b}↑ / {dn_b}↓ after regime):\n")

    # Fejlec
    print(f"  {'Event':<13}| {'n':>4} | {'WR-5d':>6} | {'AVG-5d':>7} | {'MDD':>7} | {'QUAL':>4} | BIAS")
    print(f"  {'-'*13}|{'-'*6}|{'-'*8}|{'-'*9}|{'-'*9}|{'-'*6}|{'-'*6}")
    for st in ev["active_stats"]:
        sign = "+" if st["AVG_5d"] >= 0 else ""
        label = st.get("label", st["event"])
        print(f"  {label:<13}| {st['n']:>4} | {st['WR_5d']:>5.1f}% | "
              f"{sign}{st['AVG_5d']:>5.2f}% | {st.get('MDD_3d', 0):>6.2f}% | "
              f"{st['quality']:>4} | {st.get('bias', '—')}")

    # Blended komponensek
    comp_parts = [f"DB {ev.get('db_up', 50):.0f}%", f"Composite {ev.get('composite_up', 50):.0f}%"]
    if ev.get("model_up") is not None:
        comp_parts.insert(0, f"Model {ev['model_up']:.0f}%")
    print(f"\n  COMPOSITE: {ev['n_active']} active  [{' + '.join(comp_parts)}]")
    print(f"  UP: {ev['up_pct']:.1f}% | DN: {ev['dn_pct']:.1f}% | "
          f"Bias: {ev['strength']} {ev['bias']} | Q: {ev['avg_quality']}")

    if price > 0 and atr_pct > 0:
        move = price * atr_pct / 100 * 2.2  # ~5 napos varhato mozgas
        print(f"\n  FORECAST 5-day @ {ev.get('conf_pct', 60)}% confidence:")
        print(f"    ▲ Target Up:   ${price + move:,.4g}")
        print(f"    ▼ Target Down: ${price - move:,.4g}")

    edge_dir = "BULLISH" if ev["edge"] == "LONG" else ("BEARISH" if ev["edge"] == "SHORT" else "NEUTRAL (no clear bias)")
    print(f"\n  EVENT-BASED EDGE: {edge_dir} ({ev['up_pct']:.1f}% vs {ev['dn_pct']:.1f}%) [+{ev['edge_score']} pont]")
