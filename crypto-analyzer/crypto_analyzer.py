"""
Crypto Technical Analyzer - Swing Trading Edition
==================================================
OpenBB + Binance kriptovaluta technikai elemzo script.

Funkciok:
  - Tamasz/ellenallas detektalas (Pivot, lokalis extremumok, klaszterek)
  - Volumen elemzes (Volume Profile, OBV, VWAP, A/D, whale alert)
  - Swing trading jelzesek (ADX, Ichimoku, Fibonacci, Golden/Death cross)
  - Swing Score (0-100) osszefoglalo pontozas
  - Multi-coin scanner
  - Riasztasok
  - Binance adatforras (publikus API, nincs API key szukseg)
  - Binance scanner (top 50 USDT par)

Hasznalat:
    python crypto_analyzer.py
    python crypto_analyzer.py --symbol ETH-USD --days 180
    python crypto_analyzer.py --symbols BTC-USD,ETH-USD,SOL-USD
    python crypto_analyzer.py --source binance --symbol BTCUSDT
    python crypto_analyzer.py --source binance --symbols BTCUSDT,ETHUSDT,SOLUSDT
    python crypto_analyzer.py --source binance --symbol BTC-USD --interval 4h
    python crypto_analyzer.py --source binance --scan-binance --days 180
    python crypto_analyzer.py --source alpha --symbol PLAY --days 90
    python crypto_analyzer.py --source alpha --symbols PLAY,ONDO,VIRTUAL
"""

import argparse
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import requests
from scipy.signal import argrelextrema

BINANCE_BASE_URL = "https://api.binance.us/api/v3"
GECKOTERMINAL_BASE = "https://api.geckoterminal.com/api/v2"

# Binance Alpha token cache (lazyload)
_alpha_token_cache = None


def _load_alpha_tokens() -> list[dict]:
    """Binance Alpha token lista betoltese (cached)."""
    global _alpha_token_cache
    if _alpha_token_cache is not None:
        return _alpha_token_cache
    url = "https://www.binance.com/bapi/defi/v1/public/wallet-direct/buw/wallet/cex/alpha/all/token/list"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    _alpha_token_cache = data.get("data", []) if data.get("success") else []
    return _alpha_token_cache


def _find_alpha_token(symbol: str) -> dict | None:
    """Binance Alpha token kereses szimbolum alapjan."""
    sym = symbol.upper().replace("-USD", "").replace("-USDT", "").replace("/", "")
    tokens = _load_alpha_tokens()
    # Pontos egyezes (largest marketcap first)
    matches = [t for t in tokens if t.get("symbol", "").upper() == sym]
    if matches:
        matches.sort(key=lambda t: float(t.get("marketCap") or 0), reverse=True)
        return matches[0]
    return None


# chain ID -> GeckoTerminal network slug
_CHAIN_MAP = {
    "1": "eth", "56": "bsc", "137": "polygon_pos", "8453": "base",
    "42161": "arbitrum", "10": "optimism", "43114": "avax",
    "CT_501": "solana",
}


def _get_gecko_network(chain_id) -> str:
    return _CHAIN_MAP.get(str(chain_id), "eth")


def fetch_alpha_data(symbol: str, days: int = 180, interval: str = "1d") -> pd.DataFrame:
    """Binance Alpha token adatok GeckoTerminal OHLCV API-n keresztul."""
    token_info = _find_alpha_token(symbol)
    if not token_info:
        raise ValueError(f"Binance Alpha token nem talalhato: {symbol}")

    contract = token_info["contractAddress"]
    chain_id = token_info.get("chainId", "8453")
    network = _get_gecko_network(chain_id)
    name = token_info.get("name", symbol)
    price = token_info.get("price", "?")

    print(f"  Adatok lekerese (Binance Alpha / GeckoTerminal):")
    print(f"    Token: {token_info.get('symbol')} ({name})")
    print(f"    Chain: {network} | Contract: {contract[:10]}...{contract[-6:]}")
    print(f"    Aktualis ar: ${float(price):,.6g}" if price else "")

    # DexScreener-ról megkeressük a pool cimet
    dex_resp = requests.get(
        f"https://api.dexscreener.com/latest/dex/tokens/{contract}", timeout=15)
    dex_data = dex_resp.json()
    pairs = dex_data.get("pairs", [])
    if not pairs:
        raise ValueError(f"Nem talalhato DEX par: {symbol} ({contract})")

    # Legnagyobb volume-u par kivalasztasa
    pairs.sort(key=lambda p: float(p.get("volume", {}).get("h24", 0) or 0), reverse=True)
    best_pair = pairs[0]
    pool_addr = best_pair.get("pairAddress", "")
    pair_chain = best_pair.get("chainId", network)
    dex_name = best_pair.get("dexId", "?")

    print(f"    DEX: {dex_name} | Pool: {pool_addr[:10]}...{pool_addr[-6:]}")

    # GeckoTerminal OHLCV
    gt_timeframe = {"1d": "day", "1h": "hour", "4h": "hour", "1w": "day"}.get(interval, "day")
    gt_aggregate = {"4h": 4, "1w": 7}.get(interval, 1)

    all_ohlcv = []
    page_token = None
    needed = days if gt_timeframe == "day" else days * (24 // gt_aggregate if gt_timeframe == "hour" else 1)

    while len(all_ohlcv) < needed:
        limit = min(1000, needed - len(all_ohlcv))
        gt_url = (f"{GECKOTERMINAL_BASE}/networks/{pair_chain}/pools/{pool_addr}"
                  f"/ohlcv/{gt_timeframe}")
        params = {"aggregate": gt_aggregate, "limit": limit, "currency": "usd"}
        if page_token:
            params["before_timestamp"] = page_token
        gr = requests.get(gt_url, timeout=15)
        if gr.status_code != 200:
            break
        gd = gr.json()
        ohlcv_list = gd.get("data", {}).get("attributes", {}).get("ohlcv_list", [])
        if not ohlcv_list:
            break
        all_ohlcv.extend(ohlcv_list)
        page_token = ohlcv_list[-1][0]
        if len(ohlcv_list) < limit:
            break

    if not all_ohlcv:
        raise ValueError(f"Nincs OHLCV adat: {symbol}")

    # Forditott sorrend (legrégebbi elol)
    all_ohlcv.sort(key=lambda x: x[0])

    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="s")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df.set_index("date")[["open", "high", "low", "close", "volume"]]
    df = df[~df.index.duplicated(keep="last")]

    print(f"    Betoltve: {len(df)} {gt_timeframe} gyertya ({df.index[0].date()} -> {df.index[-1].date()})")
    return df


def fetch_alpha_ticker(symbol: str) -> dict | None:
    """Binance Alpha token 24h adatok DexScreener-ról."""
    token_info = _find_alpha_token(symbol)
    if not token_info:
        return None
    contract = token_info["contractAddress"]
    dex_resp = requests.get(
        f"https://api.dexscreener.com/latest/dex/tokens/{contract}", timeout=15)
    pairs = dex_resp.json().get("pairs", [])
    if not pairs:
        return None
    pairs.sort(key=lambda p: float(p.get("volume", {}).get("h24", 0) or 0), reverse=True)
    p = pairs[0]
    price = float(p.get("priceUsd", 0) or 0)
    return {
        "symbol": token_info.get("symbol"),
        "display": f"{token_info.get('symbol')}/USDC (Binance Alpha)",
        "price": price,
        "change_pct": float(p.get("priceChange", {}).get("h24", 0) or 0),
        "high_24h": price,  # DexScreener nem ad high/low-t
        "low_24h": price,
        "volume_24h": float(p.get("volume", {}).get("h24", 0) or 0),
        "quote_volume_24h": float(p.get("volume", {}).get("h24", 0) or 0),
        "bid": price,
        "ask": price,
        "trades_24h": int(p.get("txns", {}).get("h24", {}).get("buys", 0) or 0)
                     + int(p.get("txns", {}).get("h24", {}).get("sells", 0) or 0),
        "liquidity": float(p.get("liquidity", {}).get("usd", 0) or 0),
        "dex": p.get("dexId", "?"),
        "chain": p.get("chainId", "?"),
        "market_cap": float(token_info.get("marketCap", 0) or 0),
    }


# ============================================================================
# 1. ADATLEKERDEZES
# ============================================================================
def _symbol_to_binance(symbol: str, quote: str = "USDT") -> str:
    """BTC-USD -> BTCUSDT, ETHUSDT marad ETHUSDT."""
    s = symbol.upper().replace("/", "").replace(" ", "")
    for suffix in ["-USD", "-USDT", "-BUSD", "-BTC", "-ETH"]:
        if s.endswith(suffix):
            base = s[: -len(suffix)]
            return base + quote
    if not any(s.endswith(q) for q in ["USDT", "BUSD", "BTC", "ETH"]):
        return s + quote
    return s


def _binance_display_name(binance_sym: str) -> str:
    """BTCUSDT -> BTC/USDT."""
    for q in ["USDT", "BUSD", "BTC", "ETH"]:
        if binance_sym.endswith(q):
            return binance_sym[: -len(q)] + "/" + q
    return binance_sym


def _binance_interval_ms(interval: str) -> int:
    units = {"m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}
    num = int(interval[:-1])
    return num * units.get(interval[-1], 86_400_000)


def fetch_binance_data(
    symbol: str, days: int = 180, interval: str = "1d", quote: str = "USDT",
) -> pd.DataFrame:
    """Binance publikus klines API-ról OHLCV adat lekerese."""
    bn_symbol = _symbol_to_binance(symbol, quote)
    display = _binance_display_name(bn_symbol)
    print(f"  Adatok lekerese (Binance): {display} ({interval}, {days} nap)")

    end_ms = int(datetime.now().timestamp() * 1000)
    start_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    all_rows = []
    current_start = start_ms
    limit = 1000

    while current_start < end_ms:
        params = {
            "symbol": bn_symbol, "interval": interval,
            "startTime": current_start, "endTime": end_ms, "limit": limit,
        }
        resp = requests.get(f"{BINANCE_BASE_URL}/klines", params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_rows.extend(data)
        last_close_time = data[-1][6]
        current_start = last_close_time + 1
        if len(data) < limit:
            break

    if not all_rows:
        raise ValueError(f"Nincs adat: {bn_symbol}")

    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_vol",
        "taker_buy_quote_vol", "ignore",
    ])
    df["date"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df.set_index("date")[["open", "high", "low", "close", "volume"]]
    df = df[~df.index.duplicated(keep="last")]
    return df


def fetch_binance_ticker_24h(symbol: str, quote: str = "USDT") -> dict:
    """24h ticker statisztikak (volume, change, bid/ask)."""
    bn = _symbol_to_binance(symbol, quote)
    resp = requests.get(f"{BINANCE_BASE_URL}/ticker/24hr",
                        params={"symbol": bn}, timeout=10)
    resp.raise_for_status()
    d = resp.json()
    return {
        "symbol": bn,
        "display": _binance_display_name(bn),
        "price": float(d.get("lastPrice", 0)),
        "change_pct": float(d.get("priceChangePercent", 0)),
        "high_24h": float(d.get("highPrice", 0)),
        "low_24h": float(d.get("lowPrice", 0)),
        "volume_24h": float(d.get("volume", 0)),
        "quote_volume_24h": float(d.get("quoteVolume", 0)),
        "bid": float(d.get("bidPrice", 0)),
        "ask": float(d.get("askPrice", 0)),
        "trades_24h": int(d.get("count", 0)),
    }


def fetch_binance_funding_rate(symbol: str, quote: str = "USDT") -> float | None:
    """Funding rate lekeres futures parhoz (ha elerheto)."""
    bn = _symbol_to_binance(symbol, quote)
    try:
        resp = requests.get(
            "https://fapi.binance.com/fapi/v1/premiumIndex",
            params={"symbol": bn}, timeout=10,
        )
        if resp.status_code == 200:
            d = resp.json()
            return float(d.get("lastFundingRate", 0))
    except Exception:
        pass
    return None


def scan_binance_top_pairs(
    quote: str = "USDT", min_volume_usd: float = 1_000_000,
) -> list[str]:
    """Top 50 USDT par lekerese Binance-ról, volume alapjan szurve."""
    resp = requests.get(f"{BINANCE_BASE_URL}/ticker/24hr", timeout=15)
    resp.raise_for_status()
    tickers = resp.json()
    usdt_pairs = []
    for t in tickers:
        sym = t["symbol"]
        if not sym.endswith(quote):
            continue
        qv = float(t.get("quoteVolume", 0))
        if qv < min_volume_usd:
            continue
        usdt_pairs.append((sym, qv))
    usdt_pairs.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in usdt_pairs[:50]]


def fetch_crypto_data(
    symbol: str = "BTC-USD",
    days: int = 180,
    source: str = "openbb",
    provider: str = "yfinance",
    interval: str = "1d",
    quote: str = "USDT",
) -> pd.DataFrame:
    """Adatlekerdezes source-tol fuggoen."""
    if source == "alpha":
        return fetch_alpha_data(symbol, days, interval)
    if source == "binance":
        return fetch_binance_data(symbol, days, interval, quote)
    # OpenBB
    from openbb import obb
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
    levels["S1"] = 2 * pp - h
    levels["R1"] = 2 * pp - l
    levels["S2"] = pp - (h - l)
    levels["R2"] = pp + (h - l)
    levels["S3"] = l - 2 * (h - pp)
    levels["R3"] = h + 2 * (pp - l)
    diff = h - l
    levels["Fib_S1"] = pp - 0.382 * diff
    levels["Fib_S2"] = pp - 0.618 * diff
    levels["Fib_R1"] = pp + 0.382 * diff
    levels["Fib_R2"] = pp + 0.618 * diff
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
    levels = [close[i] for i in local_max_idx] + [close[i] for i in local_min_idx]
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
    return (np.sign(df["close"].diff()) * df["volume"].fillna(0)).cumsum()


def calc_vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (tp * df["volume"].fillna(0)).cumsum()
    cum_vol = df["volume"].fillna(0).cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


def calc_ad_line(df: pd.DataFrame) -> pd.Series:
    high, low, close, vol = df["high"], df["low"], df["close"], df["volume"].fillna(0)
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    return (mfm.fillna(0) * vol).cumsum()


def calc_volume_profile(df: pd.DataFrame, bins: int = 30) -> pd.DataFrame:
    price_min, price_max = df["close"].min(), df["close"].max()
    edges = np.linspace(price_min, price_max, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    vol_per_bin = np.zeros(bins)
    for i in range(bins):
        mask = (df["close"] >= edges[i]) & (df["close"] < edges[i + 1])
        vol_per_bin[i] = df.loc[mask, "volume"].fillna(0).sum()
    return pd.DataFrame({"price": centers, "volume": vol_per_bin})


def detect_volume_anomaly(df: pd.DataFrame, window=20, multiplier=2.0) -> bool:
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
    return pd.DataFrame({
        "tenkan": tenkan, "kijun": kijun,
        "senkou_a": senkou_a, "senkou_b": senkou_b,
    })


def calc_fibonacci_retracement(df: pd.DataFrame) -> dict:
    swing_high, swing_low = df["close"].max(), df["close"].min()
    diff = swing_high - swing_low
    return {f"Fib_{r:.1%}": swing_high - r * diff
            for r in [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]}


def detect_golden_death_cross(df: pd.DataFrame) -> str:
    if len(df) < 201:
        return "N/A"
    sma50, sma200 = df["sma_50"], df["sma_200"]
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
    score = 50.0
    adx_val = last.get("adx", 0)
    plus_di = last.get("plus_di", 0)
    minus_di = last.get("minus_di", 0)
    if adx_val > 25:
        ts = min(adx_val, 50) / 50 * 20
        score += ts if plus_di > minus_di else -ts
    rsi_val = last.get("rsi", 50)
    if rsi_val < 30:
        score += 15 * (30 - rsi_val) / 30
    elif rsi_val > 70:
        score -= 15 * (rsi_val - 70) / 30
    macd_val = last.get("macd", 0)
    macd_sig = last.get("macd_signal", 0)
    diff = abs(macd_val - macd_sig)
    contribution = min(15, 15 * diff / (abs(macd_sig) + 1e-9))
    score += contribution if macd_val > macd_sig else -contribution
    vol = df["volume"].fillna(0)
    if len(vol) >= 21:
        avg = vol.iloc[-21:-1].mean()
        vr = vol.iloc[-1] / avg if avg > 0 else 1
        if vr > 1.5:
            score += 10 * min(vr - 1, 1)
        elif vr < 0.5:
            score -= 5
    close = last["close"]
    if sr_levels:
        ns = max([l for l in sr_levels if l <= close], default=None)
        nr = min([l for l in sr_levels if l > close], default=None)
        if ns and (close - ns) / close < 0.02:
            score += 10
        if nr and (nr - close) / close < 0.02:
            score -= 10
    score = max(0, min(100, score))
    labels = [(80, "Eros vetel"), (60, "Gyenge vetel"), (40, "Semleges"),
              (20, "Gyenge eladas"), (0, "Eros eladas")]
    rec = next(lb for th, lb in labels if score >= th)
    return round(score, 1), rec


# ============================================================================
# 7. RIASZTASOK
# ============================================================================
def generate_alerts(df: pd.DataFrame, sr_levels: list) -> list:
    alerts = []
    last = df.iloc[-1]
    close = last["close"]
    rsi = last.get("rsi", 50)
    if rsi > 80:
        alerts.append(f"RSI TULVETT ({rsi:.1f}) - Extrem zona!")
    elif rsi < 20:
        alerts.append(f"RSI TULELADOTT ({rsi:.1f}) - Extrem zona!")
    if len(df) >= 2:
        prev = df.iloc[-2]
        if last["macd"] > last["macd_signal"] and prev["macd"] <= prev["macd_signal"]:
            alerts.append("MACD BULLISH CROSSOVER - Veteli jelzes!")
        elif last["macd"] < last["macd_signal"] and prev["macd"] >= prev["macd_signal"]:
            alerts.append("MACD BEARISH CROSSOVER - Eladasi jelzes!")
    cross = detect_golden_death_cross(df)
    if "GOLDEN CROSS" in cross:
        alerts.append("GOLDEN CROSS (SMA50 x SMA200) - Hosszu tavu veteli jelzes!")
    elif "DEATH CROSS" in cross:
        alerts.append("DEATH CROSS (SMA50 x SMA200) - Hosszu tavu eladasi jelzes!")
    for lvl in sr_levels:
        pct = abs(close - lvl) / close
        if 0.001 < pct < 0.02:
            tag = "TAMASZ" if lvl < close else "ELLENALLAS"
            alerts.append(f"{tag} szint kozel: ${lvl:,.4g} ({pct:.1%} tavolsag)")
    if detect_volume_anomaly(df):
        vr = df["volume"].iloc[-1] / df["volume"].iloc[-21:-1].mean()
        alerts.append(f"VOLUME SPIKE ({vr:.1f}x atlag) - Whale gyanu!")
    return alerts


# ============================================================================
# 8. OSSZES INDIKATOR HOZZAADASA
# ============================================================================
def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["sma_20"] = calc_sma(df["close"], 20)
    df["sma_50"] = calc_sma(df["close"], 50)
    df["sma_200"] = calc_sma(df["close"], 200)
    df["rsi"] = calc_rsi(df)
    macd = calc_macd(df)
    df["macd"], df["macd_signal"], df["macd_hist"] = macd["macd"], macd["signal"], macd["histogram"]
    bb = calc_bollinger(df)
    df["bb_upper"], df["bb_middle"], df["bb_lower"] = bb["bb_upper"], bb["bb_middle"], bb["bb_lower"]
    df["obv"] = calc_obv(df)
    df["vwap"] = calc_vwap(df)
    df["ad_line"] = calc_ad_line(df)
    adx = calc_adx(df)
    df["adx"], df["plus_di"], df["minus_di"] = adx["adx"], adx["plus_di"], adx["minus_di"]
    ichi = calc_ichimoku(df)
    df["tenkan"], df["kijun"] = ichi["tenkan"], ichi["kijun"]
    df["senkou_a"], df["senkou_b"] = ichi["senkou_a"], ichi["senkou_b"]
    return df


# ============================================================================
# 9. MEGJELENITES - 6 PANELES CHART
# ============================================================================
def plot_chart(df: pd.DataFrame, symbol: str, sr_levels: list) -> None:
    fig = plt.figure(figsize=(18, 22))
    fig.suptitle(f"{symbol} — Swing Trading Technikai Elemzes", fontsize=16, fontweight="bold")
    gs = gridspec.GridSpec(6, 1, height_ratios=[4, 1.5, 1, 1, 1, 1], hspace=0.30)
    dates = df.index

    # --- Panel 1: Ar + BB + SMA + S/R + Ichimoku ---
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(dates, df["close"], lw=1.3, color="#1f77b4", label="Zaroar", zorder=5)
    ax1.plot(dates, df["sma_20"], lw=0.8, ls="--", color="#ff7f0e", label="SMA 20")
    ax1.plot(dates, df["sma_50"], lw=0.8, ls="--", color="#2ca02c", label="SMA 50")
    if df["sma_200"].notna().any():
        ax1.plot(dates, df["sma_200"], lw=0.8, ls="--", color="#d62728", label="SMA 200")
    ax1.fill_between(dates, df["bb_upper"], df["bb_lower"], alpha=0.08, color="blue", label="Bollinger")
    ax1.plot(dates, df["bb_upper"], lw=0.4, color="blue", alpha=0.4)
    ax1.plot(dates, df["bb_lower"], lw=0.4, color="blue", alpha=0.4)
    sa, sb = df["senkou_a"], df["senkou_b"]
    ax1.fill_between(dates, sa, sb, where=sa >= sb, alpha=0.10, color="green", label="Ichimoku (bull)")
    ax1.fill_between(dates, sa, sb, where=sa < sb, alpha=0.10, color="red", label="Ichimoku (bear)")
    price_range = df["close"].max() - df["close"].min()
    for lvl in sr_levels:
        if df["close"].min() - price_range * 0.1 < lvl < df["close"].max() + price_range * 0.1:
            ax1.axhline(lvl, lw=0.7, ls=":", color="#e91e63", alpha=0.6)
            ax1.text(dates[-1], lvl, f" ${lvl:,.4g}", fontsize=6, color="#e91e63", va="center", ha="left")
    ax1.set_xlim(dates[0], dates[-1])
    ax1.set_ylabel("Arfolyam (USD)")
    ax1.legend(loc="upper left", fontsize=7, ncol=3)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Arfolyam + Bollinger + SMA + Ichimoku + S/R szintek")

    # --- Panel 2: Volume + Volume Profile ---
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    vc = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(df["close"], df["open"])]
    ax2.bar(dates, df["volume"].fillna(0).astype(float), color=vc, alpha=0.7, width=0.8)
    avg_vol = df["volume"].fillna(0).rolling(20).mean()
    ax2.plot(dates, avg_vol * 2, lw=0.7, ls="--", color="purple", alpha=0.5, label="2x atlag (whale)")
    vp = calc_volume_profile(df, bins=25)
    ax2_vp = ax2.twinx()
    max_vp = vp["volume"].max()
    if max_vp > 0:
        vp_n = vp["volume"] / max_vp
        pbh = (vp["price"].iloc[1] - vp["price"].iloc[0]) if len(vp) > 1 else 1
        ax2_vp.barh(vp["price"], vp_n, height=pbh * 0.9, alpha=0.15, color="blue")
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
    hc = ["#26a69a" if v >= 0 else "#ef5350" for v in df["macd_hist"]]
    ax4.bar(dates, df["macd_hist"], color=hc, alpha=0.5, width=0.8)
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
    l1, lb1 = ax5.get_legend_handles_labels()
    l2, lb2 = ax5_ad.get_legend_handles_labels()
    ax5.legend(l1 + l2, lb1 + lb2, loc="upper left", fontsize=7)
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

    fname = f"{symbol.replace('/', '-').replace(' ', '')}_swing_analysis.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"  Chart elmentve: {fname}")
    plt.close(fig)


# ============================================================================
# 10. SZOVEGES OSSZEFOGLALO
# ============================================================================
def print_summary(df: pd.DataFrame, symbol: str, sr_levels: list,
                  binance_extra: dict | None = None) -> dict:
    last = df.iloc[-1]
    prev = df.iloc[-2]
    score, rec = calc_swing_score(df, sr_levels)
    alerts = generate_alerts(df, sr_levels)
    cross = detect_golden_death_cross(df)

    print("\n" + "=" * 70)
    print(f"  {symbol} — Swing Trading Osszefoglalo")
    print("=" * 70)
    print(f"  Datum:              {df.index[-1].strftime('%Y-%m-%d %H:%M')}")
    print(f"  Zaroar:             ${last['close']:,.4g}")
    chg = ((last['close'] / prev['close']) - 1) * 100
    print(f"  Valtozas (1 nap):   {chg:+.2f}%")

    if binance_extra:
        print("-" * 70)
        print(f"  Binance 24h adatok:")
        print(f"    24h High/Low:     ${binance_extra['high_24h']:,.4g} / ${binance_extra['low_24h']:,.4g}")
        print(f"    24h Volume:       {binance_extra['volume_24h']:,.2f} (${binance_extra['quote_volume_24h']:,.0f})")
        print(f"    24h Change:       {binance_extra['change_pct']:+.2f}%")
        print(f"    Bid/Ask:          ${binance_extra['bid']:,.4g} / ${binance_extra['ask']:,.4g}")
        spread = binance_extra['ask'] - binance_extra['bid']
        spread_pct = spread / binance_extra['ask'] * 100 if binance_extra['ask'] > 0 else 0
        print(f"    Spread:           ${spread:,.4g} ({spread_pct:.4f}%)")
        print(f"    Trades 24h:       {binance_extra['trades_24h']:,}")
        if binance_extra.get("funding_rate") is not None:
            print(f"    Funding Rate:     {binance_extra['funding_rate'] * 100:.4f}%")
        if binance_extra.get("liquidity"):
            print(f"    Liquidity:        ${binance_extra['liquidity']:,.0f}")
        if binance_extra.get("dex"):
            print(f"    DEX/Chain:        {binance_extra['dex']} / {binance_extra.get('chain', '?')}")
        if binance_extra.get("market_cap"):
            print(f"    Market Cap:       ${binance_extra['market_cap']:,.0f}")

    print("-" * 70)
    print(f"  SMA 20/50/200:      ${last['sma_20']:,.4g} / ${last['sma_50']:,.4g}", end="")
    if pd.notna(last["sma_200"]):
        print(f" / ${last['sma_200']:,.4g}")
    else:
        print(" / N/A")
    print(f"  Bollinger:          ${last['bb_lower']:,.4g} - ${last['bb_upper']:,.4g}")
    bb_pct = (last["close"] - last["bb_lower"]) / (last["bb_upper"] - last["bb_lower"])
    print(f"  BB %B:              {bb_pct:.1%}")
    print("-" * 70)
    rsi = last["rsi"]
    rsi_tag = " TULVETT!" if rsi > 80 else (" TULELADOTT!" if rsi < 20 else "")
    print(f"  RSI (14):           {rsi:.1f}{rsi_tag}")
    print(f"  MACD / Szignal:     {last['macd']:.4g} / {last['macd_signal']:.4g}")
    print(f"  ADX:                {last['adx']:.1f}  (+DI: {last['plus_di']:.1f}  -DI: {last['minus_di']:.1f})")
    print(f"  SMA Cross:          {cross}")
    print(f"  VWAP:               ${last['vwap']:,.4g}")
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
def run_scanner(symbols: list, days: int, source: str, provider: str,
                interval: str, quote: str) -> None:
    results = []
    for sym in symbols:
        try:
            df = fetch_crypto_data(sym, days, source, provider, interval, quote)
            df = add_all_indicators(df)
            sr = get_sr_levels(df)
            # Extra adatok forrastol fuggoen
            binance_extra = None
            if source == "binance":
                try:
                    binance_extra = fetch_binance_ticker_24h(sym, quote)
                    binance_extra["funding_rate"] = fetch_binance_funding_rate(sym, quote)
                except Exception:
                    pass
            elif source == "alpha":
                try:
                    binance_extra = fetch_alpha_ticker(sym)
                except Exception:
                    pass
            info = print_summary(df, sym, sr, binance_extra)
            plot_chart(df, sym, sr)
            results.append(info)
        except Exception as e:
            print(f"  HIBA ({sym}): {e}")

    if len(results) > 1:
        results.sort(key=lambda x: x["score"], reverse=True)
        print("\n\n" + "=" * 90)
        print("  MULTI-COIN SCANNER OSSZEFOGLALO (rendezve swing score szerint)")
        print("=" * 90)
        hdr = f"  {'Coin':<14}{'Ar':>14}{'Valt%':>8}{'RSI':>7}{'ADX':>7}{'MACD':>12}{'Score':>8}  {'Jelzes':<14}{'Alert':>6}"
        print(hdr)
        print("-" * 90)
        for r in results:
            print(
                f"  {r['symbol']:<14}"
                f"${r['close']:>12,.4g}"
                f"{r['change_pct']:>+7.2f}%"
                f"{r['rsi']:>7.1f}"
                f"{r['adx']:>7.1f}"
                f"{r['macd']:>+12.4g}"
                f"{r['score']:>7.1f}"
                f"  {r['rec']:<14}"
                f"{r['alerts']:>4}"
            )
        print("=" * 90)


def run_binance_scan(days: int, interval: str, quote: str,
                     min_volume: float) -> None:
    """Binance top 50 par scan es elemzes."""
    print(f"\n  Binance Scanner: top 50 {quote} par lekerese (min vol: ${min_volume:,.0f})...")
    top_symbols = scan_binance_top_pairs(quote, min_volume)
    print(f"  Talalt parok: {len(top_symbols)}\n")
    if not top_symbols:
        print("  Nincs elegendo par a szuresnek megfelelo.")
        return
    run_scanner(top_symbols, days, "binance", "yfinance", interval, quote)


# ============================================================================
# 12. FOPROGRAM
# ============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crypto Swing Trading Analyzer (OpenBB + Binance)")
    parser.add_argument("--symbol", default=None,
                        help="Egyetlen kriptopar (pl. BTC-USD vagy BTCUSDT)")
    parser.add_argument("--symbols", default=None,
                        help="Tobb coin vesszoval: BTC-USD,ETH-USD vagy BTCUSDT,ETHUSDT")
    parser.add_argument("--days", type=int, default=180,
                        help="Visszatekintesi idoszak napokban (alapert: 180)")
    parser.add_argument("--source", default="openbb", choices=["openbb", "binance", "alpha"],
                        help="Adatforras: openbb (default), binance, alpha (Binance Alpha/DEX)")
    parser.add_argument("--provider", default="yfinance",
                        help="OpenBB provider (yfinance, fmp, tiingo)")
    parser.add_argument("--interval", default="1d",
                        help="Idointervallum Binance-hoz (1h, 4h, 1d, 1w)")
    parser.add_argument("--quote", default="USDT",
                        help="Quote currency Binance-hoz (USDT, BUSD, BTC, ETH)")
    parser.add_argument("--scan-binance", action="store_true",
                        help="Binance top 50 USDT par scan")
    parser.add_argument("--min-volume", type=float, default=1_000_000,
                        help="Minimum 24h volume USD-ben (scan-binance-hoz)")
    args = parser.parse_args()

    source = args.source
    if args.scan_binance:
        source = "binance"

    print(f"\nCrypto Swing Trading Analyzer")
    print(f"Idoszak: {args.days} nap | Forras: {source}"
          + (f" | Interval: {args.interval}" if source in ("binance", "alpha") else
             f" | Provider: {args.provider}"))

    if args.scan_binance:
        run_binance_scan(args.days, args.interval, args.quote, args.min_volume)
        return

    if args.symbols:
        coin_list = [s.strip() for s in args.symbols.split(",")]
    elif args.symbol:
        coin_list = [args.symbol]
    else:
        defaults = {"openbb": "BTC-USD", "binance": "BTCUSDT", "alpha": "PLAY"}
        coin_list = [defaults.get(source, "BTC-USD")]

    print(f"Coinok: {', '.join(coin_list)}\n")
    run_scanner(coin_list, args.days, source, args.provider,
                args.interval, args.quote)


if __name__ == "__main__":
    main()
