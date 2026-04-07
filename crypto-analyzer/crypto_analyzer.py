"""
Crypto Technical Analyzer - Swing Trading Edition
==================================================
OpenBB-alapu kriptovaluta technikai elemzo script.

Funkciok:
  - Tamasz/ellenallas detektalas (Pivot, lokalis extremumok, klaszterek)
  - Volumen elemzes (Volume Profile, OBV, VWAP, A/D, whale alert)
  - Swing trading jelzesek (ADX, Ichimoku, Fibonacci, Golden/Death cross)
  - Swing Score (0-100) osszefoglalo pontozas
  - Multi-coin scanner
  - Riasztasok

Hasznalat:
    python crypto_analyzer.py
    python crypto_analyzer.py --symbol ETH-USD --days 180
    python crypto_analyzer.py --symbols BTC-USD,ETH-USD,SOL-USD --days 180
"""

import argparse
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from openbb import obb


# ============================================================================
# 1. ADATLEKERDEZES
# ============================================================================
def fetch_crypto_data(
    symbol: str = "BTC-USD",
    days: int = 180,
    provider: str = "yfinance",
) -> pd.DataFrame:
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    print(f"  Adatok lekerese: {symbol} ({start_date} -> {end_date})")
    result = obb.crypto.price.historical(
        symbol=symbol, start_date=start_date, end_date=end_date,
        interval="1d", provider=provider,
    )
    df = result.to_df()
    if df.empty:
        raise ValueError(f"Nem erkezett adat: {symbol}")
    df.index = pd.to_datetime(df.index)
    return df


# ============================================================================
# 2. ALAPINDIKATOROK
# ============================================================================
def calc_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()


def calc_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def calc_rsi(df: pd.DataFrame, length: int = 14) -> pd.Series:
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
    ema_fast = calc_ema(df["close"], fast)
    ema_slow = calc_ema(df["close"], slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({
        "macd": macd_line, "signal": signal_line,
        "histogram": macd_line - signal_line,
    })


def calc_bollinger(df: pd.DataFrame, length=20, std_dev=2.0) -> pd.DataFrame:
    mid = df["close"].rolling(window=length).mean()
    std = df["close"].rolling(window=length).std()
    return pd.DataFrame({
        "bb_upper": mid + std_dev * std, "bb_middle": mid,
        "bb_lower": mid - std_dev * std,
    })


# ============================================================================
# 3. TAMASZ / ELLENALLAS DETEKTALAS
# ============================================================================
def calc_pivot_points(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    h, l, c = last["high"], last["low"], last["close"]
    pp = (h + l + c) / 3.0
    levels = {"PP": pp}
    # Klasszikus
    levels["S1"] = 2 * pp - h
    levels["R1"] = 2 * pp - l
    levels["S2"] = pp - (h - l)
    levels["R2"] = pp + (h - l)
    levels["S3"] = l - 2 * (h - pp)
    levels["R3"] = h + 2 * (pp - l)
    # Fibonacci
    diff = h - l
    levels["Fib_S1"] = pp - 0.382 * diff
    levels["Fib_S2"] = pp - 0.618 * diff
    levels["Fib_R1"] = pp + 0.382 * diff
    levels["Fib_R2"] = pp + 0.618 * diff
    # Camarilla
    levels["Cam_S1"] = c - diff * 1.1 / 12
    levels["Cam_R1"] = c + diff * 1.1 / 12
    levels["Cam_S2"] = c - diff * 1.1 / 6
    levels["Cam_R2"] = c + diff * 1.1 / 6
    levels["Cam_S3"] = c - diff * 1.1 / 4
    levels["Cam_R3"] = c + diff * 1.1 / 4
    return levels


def detect_local_extrema(df: pd.DataFrame, order: int = 10) -> list:
    close = df["close"].values
    local_max_idx = argrelextrema(close, np.greater_equal, order=order)[0]
    local_min_idx = argrelextrema(close, np.less_equal, order=order)[0]
    levels = []
    for i in local_max_idx:
        levels.append(close[i])
    for i in local_min_idx:
        levels.append(close[i])
    return levels


def cluster_levels(levels: list, threshold_pct: float = 0.02) -> list:
    if not levels:
        return []
    sorted_lvls = sorted(set(levels))
    clusters = []
    current_cluster = [sorted_lvls[0]]
    for lvl in sorted_lvls[1:]:
        if (lvl - current_cluster[0]) / current_cluster[0] <= threshold_pct:
            current_cluster.append(lvl)
        else:
            clusters.append(np.mean(current_cluster))
            current_cluster = [lvl]
    clusters.append(np.mean(current_cluster))
    return clusters


def get_sr_levels(df: pd.DataFrame) -> list:
    n = min(90, len(df))
    recent = df.iloc[-n:]
    pivot_levels = calc_pivot_points(recent)
    extrema_levels = detect_local_extrema(recent, order=max(5, n // 15))
    all_levels = list(pivot_levels.values()) + extrema_levels
    return cluster_levels(all_levels)


# ============================================================================
# 4. VOLUMEN ELEMZES
# ============================================================================
def calc_obv(df: pd.DataFrame) -> pd.Series:
    obv = (np.sign(df["close"].diff()) * df["volume"].fillna(0)).cumsum()
    return obv


def calc_vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (tp * df["volume"].fillna(0)).cumsum()
    cum_vol = df["volume"].fillna(0).cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


def calc_ad_line(df: pd.DataFrame) -> pd.Series:
    high, low, close, vol = df["high"], df["low"], df["close"], df["volume"].fillna(0)
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mfm = mfm.fillna(0)
    return (mfm * vol).cumsum()


def calc_volume_profile(df: pd.DataFrame, bins: int = 30) -> pd.DataFrame:
    price_min, price_max = df["close"].min(), df["close"].max()
    edges = np.linspace(price_min, price_max, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    vol_per_bin = np.zeros(bins)
    for i in range(bins):
        mask = (df["close"] >= edges[i]) & (df["close"] < edges[i + 1])
        vol_per_bin[i] = df.loc[mask, "volume"].fillna(0).sum()
    return pd.DataFrame({"price": centers, "volume": vol_per_bin})


def detect_volume_anomaly(df: pd.DataFrame, window: int = 20, multiplier: float = 2.0) -> bool:
    vol = df["volume"].fillna(0)
    if len(vol) < window + 1:
        return False
    avg_vol = vol.iloc[-(window + 1):-1].mean()
    return bool(avg_vol > 0 and vol.iloc[-1] > multiplier * avg_vol)


# ============================================================================
# 5. SWING TRADING INDIKATOROK
# ============================================================================
def calc_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, min_periods=period).mean()
    return pd.DataFrame({"adx": adx, "plus_di": plus_di, "minus_di": minus_di})


def calc_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    high, low = df["high"], df["low"]
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    chikou = df["close"].shift(-26)
    return pd.DataFrame({
        "tenkan": tenkan, "kijun": kijun,
        "senkou_a": senkou_a, "senkou_b": senkou_b, "chikou": chikou,
    })


def calc_fibonacci_retracement(df: pd.DataFrame) -> dict:
    close = df["close"]
    swing_high = close.max()
    swing_low = close.min()
    diff = swing_high - swing_low
    ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    levels = {}
    for r in ratios:
        levels[f"Fib_{r:.1%}"] = swing_high - r * diff
    return levels


def detect_golden_death_cross(df: pd.DataFrame) -> str:
    if len(df) < 201:
        return "N/A"
    sma50 = df["sma_50"]
    sma200 = df["sma_200"]
    if sma50.iloc[-1] > sma200.iloc[-1] and sma50.iloc[-2] <= sma200.iloc[-2]:
        return "GOLDEN CROSS"
    if sma50.iloc[-1] < sma200.iloc[-1] and sma50.iloc[-2] >= sma200.iloc[-2]:
        return "DEATH CROSS"
    if sma50.iloc[-1] > sma200.iloc[-1]:
        return "Bullish (SMA50 > SMA200)"
    return "Bearish (SMA50 < SMA200)"


# ============================================================================
# 6. SCORING RENDSZER
# ============================================================================
def calc_swing_score(df: pd.DataFrame, sr_levels: list) -> tuple:
    last = df.iloc[-1]
    score = 50.0  # start neutral

    # --- Trend (ADX) ---  max +/-20
    adx_val = last.get("adx", 0)
    plus_di = last.get("plus_di", 0)
    minus_di = last.get("minus_di", 0)
    if adx_val > 25:
        trend_strength = min(adx_val, 50) / 50 * 20
        score += trend_strength if plus_di > minus_di else -trend_strength
    # --- RSI ---  max +/-15
    rsi_val = last.get("rsi", 50)
    if rsi_val < 30:
        score += 15 * (30 - rsi_val) / 30
    elif rsi_val > 70:
        score -= 15 * (rsi_val - 70) / 30
    # --- MACD ---  max +/-15
    macd_val = last.get("macd", 0)
    macd_sig = last.get("macd_signal", 0)
    if macd_val > macd_sig:
        score += min(15, 15 * abs(macd_val - macd_sig) / (abs(macd_sig) + 1e-9))
    else:
        score -= min(15, 15 * abs(macd_val - macd_sig) / (abs(macd_sig) + 1e-9))
    # --- Volume trend ---  max +/-10
    vol = df["volume"].fillna(0)
    if len(vol) >= 21:
        vol_ratio = vol.iloc[-1] / vol.iloc[-21:-1].mean() if vol.iloc[-21:-1].mean() > 0 else 1
        if vol_ratio > 1.5:
            score += 10 * min(vol_ratio - 1, 1)
        elif vol_ratio < 0.5:
            score -= 5
    # --- S/R kozelség ---  max +/-10
    close = last["close"]
    if sr_levels:
        nearest_support = max([l for l in sr_levels if l <= close], default=None)
        nearest_resist = min([l for l in sr_levels if l > close], default=None)
        if nearest_support and (close - nearest_support) / close < 0.02:
            score += 10  # kozel a tamaszhoz -> potencialis pattanas
        if nearest_resist and (nearest_resist - close) / close < 0.02:
            score -= 10  # kozel az ellenallashoz

    score = max(0, min(100, score))
    if score >= 80:
        rec = "Eros vetel"
    elif score >= 60:
        rec = "Gyenge vetel"
    elif score >= 40:
        rec = "Semleges"
    elif score >= 20:
        rec = "Gyenge eladas"
    else:
        rec = "Eros eladas"
    return round(score, 1), rec


# ============================================================================
# 7. RIASZTASOK
# ============================================================================
def generate_alerts(df: pd.DataFrame, sr_levels: list) -> list:
    alerts = []
    last = df.iloc[-1]
    close = last["close"]

    # RSI
    rsi = last.get("rsi", 50)
    if rsi > 80:
        alerts.append(f"RSI TULVETT ({rsi:.1f}) - Extrem zona!")
    elif rsi < 20:
        alerts.append(f"RSI TULELADOTT ({rsi:.1f}) - Extrem zona!")

    # MACD crossover
    if len(df) >= 2:
        prev = df.iloc[-2]
        if last["macd"] > last["macd_signal"] and prev["macd"] <= prev["macd_signal"]:
            alerts.append("MACD BULLISH CROSSOVER - Veteli jelzes!")
        elif last["macd"] < last["macd_signal"] and prev["macd"] >= prev["macd_signal"]:
            alerts.append("MACD BEARISH CROSSOVER - Eladasi jelzes!")

    # Golden/Death cross
    cross = detect_golden_death_cross(df)
    if "GOLDEN CROSS" in cross:
        alerts.append("GOLDEN CROSS (SMA50 x SMA200) - Hosszu tavu veteli jelzes!")
    elif "DEATH CROSS" in cross:
        alerts.append("DEATH CROSS (SMA50 x SMA200) - Hosszu tavu eladasi jelzes!")

    # S/R kozelség
    for lvl in sr_levels:
        pct = abs(close - lvl) / close
        if pct < 0.02 and pct > 0.001:
            tag = "TAMASZ" if lvl < close else "ELLENALLAS"
            alerts.append(f"{tag} szint kozel: ${lvl:,.0f} ({pct:.1%} tavolsag)")

    # Volume spike
    if detect_volume_anomaly(df):
        vol_ratio = df["volume"].iloc[-1] / df["volume"].iloc[-21:-1].mean()
        alerts.append(f"VOLUME SPIKE ({vol_ratio:.1f}x atlag) - Whale gyanu!")

    return alerts


# ============================================================================
# 8. OSSZES INDIKATOR HOZZAADASA
# ============================================================================
def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # SMA
    df["sma_20"] = calc_sma(df["close"], 20)
    df["sma_50"] = calc_sma(df["close"], 50)
    df["sma_200"] = calc_sma(df["close"], 200)
    # RSI
    df["rsi"] = calc_rsi(df)
    # MACD
    macd = calc_macd(df)
    df["macd"] = macd["macd"]
    df["macd_signal"] = macd["signal"]
    df["macd_hist"] = macd["histogram"]
    # Bollinger
    bb = calc_bollinger(df)
    df["bb_upper"] = bb["bb_upper"]
    df["bb_middle"] = bb["bb_middle"]
    df["bb_lower"] = bb["bb_lower"]
    # Volume
    df["obv"] = calc_obv(df)
    df["vwap"] = calc_vwap(df)
    df["ad_line"] = calc_ad_line(df)
    # ADX
    adx = calc_adx(df)
    df["adx"] = adx["adx"]
    df["plus_di"] = adx["plus_di"]
    df["minus_di"] = adx["minus_di"]
    # Ichimoku
    ichi = calc_ichimoku(df)
    df["tenkan"] = ichi["tenkan"]
    df["kijun"] = ichi["kijun"]
    df["senkou_a"] = ichi["senkou_a"]
    df["senkou_b"] = ichi["senkou_b"]
    return df


# ============================================================================
# 9. MEGJELENITES - 6 PANELES CHART
# ============================================================================
def plot_chart(df: pd.DataFrame, symbol: str, sr_levels: list) -> None:
    fig = plt.figure(figsize=(18, 22))
    fig.suptitle(f"{symbol} — Swing Trading Technikai Elemzes", fontsize=16, fontweight="bold")
    gs = gridspec.GridSpec(6, 1, height_ratios=[4, 1.5, 1, 1, 1, 1], hspace=0.30)
    dates = df.index

    # --- Panel 1: Ar + BB + SMA + S/R + Ichimoku Cloud ---
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(dates, df["close"], lw=1.3, color="#1f77b4", label="Zaroar", zorder=5)
    ax1.plot(dates, df["sma_20"], lw=0.8, ls="--", color="#ff7f0e", label="SMA 20")
    ax1.plot(dates, df["sma_50"], lw=0.8, ls="--", color="#2ca02c", label="SMA 50")
    if df["sma_200"].notna().any():
        ax1.plot(dates, df["sma_200"], lw=0.8, ls="--", color="#d62728", label="SMA 200")
    # Bollinger
    ax1.fill_between(dates, df["bb_upper"], df["bb_lower"], alpha=0.08, color="blue", label="Bollinger")
    ax1.plot(dates, df["bb_upper"], lw=0.4, color="blue", alpha=0.4)
    ax1.plot(dates, df["bb_lower"], lw=0.4, color="blue", alpha=0.4)
    # Ichimoku Cloud
    sa = df["senkou_a"]
    sb = df["senkou_b"]
    ax1.fill_between(dates, sa, sb, where=sa >= sb, alpha=0.10, color="green", label="Ichimoku (bull)")
    ax1.fill_between(dates, sa, sb, where=sa < sb, alpha=0.10, color="red", label="Ichimoku (bear)")
    # S/R levels
    price_range = df["close"].max() - df["close"].min()
    for lvl in sr_levels:
        if df["close"].min() - price_range * 0.1 < lvl < df["close"].max() + price_range * 0.1:
            ax1.axhline(lvl, lw=0.7, ls=":", color="#e91e63", alpha=0.6)
            ax1.text(dates[-1], lvl, f" ${lvl:,.0f}", fontsize=6, color="#e91e63",
                     va="center", ha="left")
    ax1.set_xlim(dates[0], dates[-1])
    ax1.set_ylabel("Arfolyam (USD)")
    ax1.legend(loc="upper left", fontsize=7, ncol=3)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Arfolyam + Bollinger + SMA + Ichimoku + S/R szintek")

    # --- Panel 2: Volume + Volume Profile ---
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    vol_colors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(df["close"], df["open"])]
    ax2.bar(dates, df["volume"].fillna(0).astype(float), color=vol_colors, alpha=0.7, width=0.8)
    # Whale alert sav
    avg_vol = df["volume"].fillna(0).rolling(20).mean()
    ax2.plot(dates, avg_vol * 2, lw=0.7, ls="--", color="purple", alpha=0.5, label="2x atlag (whale)")
    # Volume Profile jobb oldalra (normalizalt)
    vp = calc_volume_profile(df, bins=25)
    ax2_vp = ax2.twinx()
    max_vp = vp["volume"].max()
    if max_vp > 0:
        vp_normalized = vp["volume"] / max_vp
        price_bin_height = (vp["price"].iloc[1] - vp["price"].iloc[0]) if len(vp) > 1 else 1
        ax2_vp.barh(vp["price"], vp_normalized, height=price_bin_height * 0.9,
                     alpha=0.15, color="blue")
    ax2_vp.set_ylim(df["close"].min() * 0.95, df["close"].max() * 1.05)
    ax2_vp.set_yticks([])
    ax2_vp.set_ylabel("Vol.Profile", fontsize=7)
    ax2.set_ylabel("Forgalom")
    ax2.legend(loc="upper left", fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Forgalom + Volume Profile")

    # --- Panel 3: RSI ---
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(dates, df["rsi"], lw=1, color="#ab47bc")
    ax3.axhline(80, lw=0.7, ls="--", color="red", alpha=0.6)
    ax3.axhline(20, lw=0.7, ls="--", color="green", alpha=0.6)
    ax3.axhline(50, lw=0.5, ls=":", color="gray", alpha=0.4)
    ax3.fill_between(dates, 80, 100, alpha=0.06, color="red")
    ax3.fill_between(dates, 0, 20, alpha=0.06, color="green")
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("RSI")
    ax3.grid(True, alpha=0.3)
    ax3.set_title("RSI (14)")

    # --- Panel 4: MACD ---
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(dates, df["macd"], lw=1, color="#1f77b4", label="MACD")
    ax4.plot(dates, df["macd_signal"], lw=1, color="#ff7f0e", label="Szignal")
    hist_c = ["#26a69a" if v >= 0 else "#ef5350" for v in df["macd_hist"]]
    ax4.bar(dates, df["macd_hist"], color=hist_c, alpha=0.5, width=0.8)
    ax4.axhline(0, lw=0.5, color="black", alpha=0.3)
    ax4.set_ylabel("MACD")
    ax4.legend(loc="upper left", fontsize=7)
    ax4.grid(True, alpha=0.3)
    ax4.set_title("MACD (12, 26, 9)")

    # --- Panel 5: OBV + A/D ---
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    ax5.plot(dates, df["obv"], lw=1, color="#0288d1", label="OBV")
    ax5.set_ylabel("OBV", color="#0288d1")
    ax5_ad = ax5.twinx()
    ax5_ad.plot(dates, df["ad_line"], lw=1, color="#e65100", label="A/D Line")
    ax5_ad.set_ylabel("A/D", color="#e65100", fontsize=8)
    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_ad.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7)
    ax5.grid(True, alpha=0.3)
    ax5.set_title("OBV + Accumulation/Distribution")

    # --- Panel 6: ADX ---
    ax6 = fig.add_subplot(gs[5], sharex=ax1)
    ax6.plot(dates, df["adx"], lw=1.2, color="#333333", label="ADX")
    ax6.plot(dates, df["plus_di"], lw=0.8, color="green", label="+DI")
    ax6.plot(dates, df["minus_di"], lw=0.8, color="red", label="-DI")
    ax6.axhline(25, lw=0.7, ls="--", color="orange", alpha=0.5)
    ax6.fill_between(dates, 0, 25, alpha=0.04, color="gray")
    ax6.set_ylabel("ADX")
    ax6.legend(loc="upper left", fontsize=7)
    ax6.grid(True, alpha=0.3)
    ax6.set_title("ADX Trend Erosseg")

    ax6.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax6.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax6.get_xticklabels(), rotation=45, ha="right")

    fname = f"{symbol.replace('/', '-')}_swing_analysis.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"  Chart elmentve: {fname}")
    plt.close(fig)


# ============================================================================
# 10. SZOVEGES OSSZEFOGLALO
# ============================================================================
def print_summary(df: pd.DataFrame, symbol: str, sr_levels: list) -> dict:
    last = df.iloc[-1]
    prev = df.iloc[-2]
    score, rec = calc_swing_score(df, sr_levels)
    alerts = generate_alerts(df, sr_levels)
    cross = detect_golden_death_cross(df)

    print("\n" + "=" * 70)
    print(f"  {symbol} — Swing Trading Osszefoglalo")
    print("=" * 70)
    print(f"  Datum:              {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Zaroar:             ${last['close']:,.2f}")
    chg = ((last['close'] / prev['close']) - 1) * 100
    print(f"  Valtozas (1 nap):   {chg:+.2f}%")
    print("-" * 70)
    print(f"  SMA 20/50/200:      ${last['sma_20']:,.2f} / ${last['sma_50']:,.2f}", end="")
    if pd.notna(last["sma_200"]):
        print(f" / ${last['sma_200']:,.2f}")
    else:
        print(" / N/A")
    print(f"  Bollinger:          ${last['bb_lower']:,.2f} - ${last['bb_upper']:,.2f}")
    bb_pct = (last["close"] - last["bb_lower"]) / (last["bb_upper"] - last["bb_lower"])
    print(f"  BB %B:              {bb_pct:.1%}")
    print("-" * 70)
    rsi = last["rsi"]
    rsi_tag = " TULVETT!" if rsi > 80 else (" TULELADOTT!" if rsi < 20 else "")
    print(f"  RSI (14):           {rsi:.1f}{rsi_tag}")
    print(f"  MACD / Szignal:     {last['macd']:.2f} / {last['macd_signal']:.2f}")
    print(f"  ADX:                {last['adx']:.1f}  (+DI: {last['plus_di']:.1f}  -DI: {last['minus_di']:.1f})")
    print(f"  SMA Cross:          {cross}")
    print(f"  VWAP:               ${last['vwap']:,.2f}")
    print("-" * 70)
    print(f"  SWING SCORE:        {score}/100  ->  {rec}")
    print("-" * 70)

    if alerts:
        print("  RIASZTASOK:")
        for a in alerts:
            print(f"    >> {a}")
    else:
        print("  Nincs aktiv riasztas.")
    print("=" * 70)

    return {
        "symbol": symbol, "close": last["close"], "change_pct": chg,
        "rsi": rsi, "adx": last["adx"], "macd": last["macd"],
        "score": score, "rec": rec, "alerts": len(alerts),
    }


# ============================================================================
# 11. MULTI-COIN SCANNER
# ============================================================================
def run_scanner(symbols: list, days: int, provider: str) -> None:
    results = []
    for sym in symbols:
        try:
            df = fetch_crypto_data(symbol=sym, days=days, provider=provider)
            df = add_all_indicators(df)
            sr = get_sr_levels(df)
            info = print_summary(df, sym, sr)
            plot_chart(df, sym, sr)
            results.append(info)
        except Exception as e:
            print(f"  HIBA ({sym}): {e}")

    if len(results) > 1:
        results.sort(key=lambda x: x["score"], reverse=True)
        print("\n\n" + "=" * 90)
        print("  MULTI-COIN SCANNER OSSZEFOGLALO (rendezve swing score szerint)")
        print("=" * 90)
        header = f"  {'Coin':<12}{'Ar':>12}{'Valt%':>8}{'RSI':>7}{'ADX':>7}{'MACD':>10}{'Score':>8}{'Jelzes':<16}{'Alert':>6}"
        print(header)
        print("-" * 90)
        for r in results:
            print(
                f"  {r['symbol']:<12}"
                f"${r['close']:>10,.2f}"
                f"{r['change_pct']:>+7.2f}%"
                f"{r['rsi']:>7.1f}"
                f"{r['adx']:>7.1f}"
                f"{r['macd']:>+10.2f}"
                f"{r['score']:>7.1f}"
                f"  {r['rec']:<14}"
                f"{r['alerts']:>4}"
            )
        print("=" * 90)


# ============================================================================
# 12. FOPROGRAM
# ============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crypto Swing Trading Analyzer (OpenBB)")
    parser.add_argument("--symbol", default=None,
                        help="Egyetlen kriptopar (pl. BTC-USD)")
    parser.add_argument("--symbols", default=None,
                        help="Tobb coin vesszoval: BTC-USD,ETH-USD,SOL-USD")
    parser.add_argument("--days", type=int, default=180,
                        help="Visszatekintesi idoszak napokban (alapert: 180)")
    parser.add_argument("--provider", default="yfinance",
                        help="Adatforras (yfinance, fmp, tiingo)")
    args = parser.parse_args()

    if args.symbols:
        coin_list = [s.strip() for s in args.symbols.split(",")]
    elif args.symbol:
        coin_list = [args.symbol]
    else:
        coin_list = ["BTC-USD"]

    print(f"\nCrypto Swing Trading Analyzer")
    print(f"Idoszak: {args.days} nap | Provider: {args.provider}")
    print(f"Coinok: {', '.join(coin_list)}\n")

    run_scanner(coin_list, args.days, args.provider)


if __name__ == "__main__":
    main()
