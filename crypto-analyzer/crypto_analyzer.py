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
from colorama import Fore, Style, init as colorama_init
from scipy.signal import argrelextrema

colorama_init(autoreset=True)

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
    quiet: bool = False,
) -> pd.DataFrame:
    """Binance publikus klines API-ról OHLCV adat lekerese."""
    bn_symbol = _symbol_to_binance(symbol, quote)
    display = _binance_display_name(bn_symbol)
    if not quiet:
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
# 10. RESZLETES DONTES-TAMOGATO ELEMZES
# ============================================================================
_P = lambda v: f"${v:,.6g}"  # price formatter


def _calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


def _calc_max_drawdown(df: pd.DataFrame, window: int = 30) -> float:
    c = df["close"].iloc[-window:]
    peak = c.cummax()
    dd = (c - peak) / peak
    return float(dd.min()) * 100


def _detect_large_candles(df: pd.DataFrame, threshold: float = 0.03, lookback: int = 5) -> list:
    candles = []
    for i in range(-min(lookback, len(df)), 0):
        row = df.iloc[i]
        prev_close = df.iloc[i - 1]["close"] if i > -len(df) else row["open"]
        if prev_close == 0:
            continue
        pct = (row["close"] - prev_close) / prev_close
        if abs(pct) >= threshold:
            vol_avg = df["volume"].iloc[max(0, len(df) + i - 20):len(df) + i].mean()
            vol_ratio = row["volume"] / vol_avg if vol_avg > 0 else 0
            candles.append({
                "date": df.index[i].strftime("%m-%d"),
                "pct": pct * 100,
                "vol_ratio": vol_ratio,
            })
    return candles


def _build_entry_exit(close: float, sr_levels: list, atr: float, fib: dict) -> dict:
    supports = sorted([l for l in sr_levels if l < close], reverse=True)
    resists = sorted([l for l in sr_levels if l > close])
    fib_levels = sorted(fib.values())

    entry = supports[0] if supports else close - atr
    stop = (supports[1] if len(supports) > 1 else entry - atr) - atr * 0.2
    target1 = resists[0] if resists else close + atr * 2
    target2 = resists[1] if len(resists) > 1 else target1 + atr

    risk = close - stop
    reward1 = target1 - close
    rr1 = reward1 / risk if risk > 0 else 0

    return {
        "entry": entry, "stop": stop,
        "target1": target1, "target2": target2,
        "risk": risk, "reward1": reward1, "rr1": rr1,
        "nearest_support": supports[0] if supports else None,
        "nearest_resist": resists[0] if resists else None,
    }


def _build_context(df: pd.DataFrame, sr_levels: list) -> str:
    last = df.iloc[-1]
    rsi = last.get("rsi", 50)
    macd_bull = last.get("macd", 0) > last.get("macd_signal", 0)
    adx = last.get("adx", 0)
    plus_di = last.get("plus_di", 0)
    minus_di = last.get("minus_di", 0)
    vol = df["volume"].fillna(0)
    vol_trend = ""
    if len(vol) >= 6:
        recent_avg = vol.iloc[-5:].mean()
        older_avg = vol.iloc[-20:-5].mean() if len(vol) >= 20 else vol.iloc[:-5].mean()
        if older_avg > 0:
            if recent_avg > older_avg * 1.3:
                vol_trend = "novekvo"
            elif recent_avg < older_avg * 0.7:
                vol_trend = "csokken"
            else:
                vol_trend = "stabil"

    parts = []
    # RSI + MACD combo
    if rsi > 70 and macd_bull:
        parts.append(f"RSI {rsi:.0f} + MACD bullish = a momentum meg tart, DE kozel a tulvett zonahoz, ami korrekcios kockazatot jelent 1-3 napon belul.")
    elif rsi > 70 and not macd_bull:
        parts.append(f"RSI {rsi:.0f} tulvett + MACD bearish = bearish DIVERGENCIA. A momentum gyengul, korrekcio valoszinu.")
    elif rsi < 30 and not macd_bull:
        parts.append(f"RSI {rsi:.0f} tuleladott + MACD bearish = meg nem latszik fordulat, de a tulado zona kozel. Figyelj MACD crossoverre.")
    elif rsi < 30 and macd_bull:
        parts.append(f"RSI {rsi:.0f} tuleladott + MACD bullish cross = lehetseges fordulat/pattanas jelzes!")
    elif 45 <= rsi <= 55:
        parts.append(f"RSI {rsi:.0f} semleges zoneban - nincs egyertelmu momentum irany.")
    else:
        direction = "enyhen bullish" if rsi > 50 else "enyhen bearish"
        parts.append(f"RSI {rsi:.0f} ({direction}) {'+ MACD tamogatja.' if (rsi > 50) == macd_bull else '+ MACD nem erositi meg.'}")

    # Volume divergence
    if rsi > 65 and vol_trend == "csokken":
        parts.append("Volumen csokken mig ar magas = bearish divergencia, gyengulo felszallo nyomas.")
    elif rsi < 35 and vol_trend == "csokken":
        parts.append("Volumen csokken alacsony RSI mellett = az eladoi nyomas kimerulhet.")
    elif vol_trend == "novekvo":
        parts.append("Novekvo volumen erositi az aktualis mozgast.")

    # ADX trend
    if adx > 25:
        trend_dir = "felszallo" if plus_di > minus_di else "leszallo"
        parts.append(f"ADX {adx:.0f} = eros {trend_dir} trend. Trend-koveto strategia javasolt.")
    else:
        parts.append(f"ADX {adx:.0f} = gyenge trend / oldalazas. Range-trading lehetoseg.")

    return " ".join(parts)


def _build_scenarios(close: float, levels: dict, atr: float, adx: float) -> dict:
    nr = levels["nearest_resist"]
    ns = levels["nearest_support"]
    t1, t2 = levels["target1"], levels["target2"]

    bull = f"Ha attori a(z) {_P(nr)} ellenallast: kovetkezo celar {_P(t1)}"
    if t2 > t1:
        bull += f", majd {_P(t2)}."
    else:
        bull += "."
    bull_prob = 55 if adx > 25 else 40

    bear_target = ns - atr if ns else close - atr * 2
    bear = f"Ha elveszti a(z) {_P(ns)} tamaszt: kovetkezo support {_P(bear_target)}, "
    bear += f"varhato eses {abs(close - bear_target) / close * 100:.1f}%."
    bear_prob = 100 - bull_prob

    neutral = f"Konszolidacio {_P(ns or close - atr)} - {_P(nr or close + atr)} tartomanyban."

    return {
        "bull": bull, "bull_prob": bull_prob,
        "bear": bear, "bear_prob": bear_prob,
        "neutral": neutral,
    }


def _build_timing(df: pd.DataFrame, sr_levels: list, atr: float) -> str:
    last = df.iloc[-1]
    rsi = last.get("rsi", 50)
    macd_v = last.get("macd", 0)
    macd_s = last.get("macd_signal", 0)
    close = last["close"]

    signals_bullish = 0
    signals_bearish = 0
    if rsi < 35:
        signals_bullish += 1
    elif rsi > 65:
        signals_bearish += 1
    if macd_v > macd_s:
        signals_bullish += 1
    else:
        signals_bearish += 1
    if close > last.get("sma_20", close):
        signals_bullish += 1
    else:
        signals_bearish += 1

    if rsi > 70:
        return "VARJ belepessel. RSI tulvett zonahoz kozel - varj korrekciot vagy RSI 50 ala visszahuzodast."
    if rsi < 30 and macd_v > macd_s:
        return "MOST LEPJ BE (long). Tuleladott RSI + MACD bullish cross = fordulat jelzes."
    if signals_bullish >= 3:
        return "MOST LEPJ BE (long). Tobb indikator egyutt ad veteli jelzest."
    if signals_bearish >= 3:
        return "KERÜLD most. Tobb indikator bearish - varj stabilizalodasra."
    return "VARJ megerositesre. A jelzesek vegyesek, nincs egyertelmu belepo."


def _position_size(portfolio: float, close: float, atr: float, stop_dist: float,
                   vol_24h: float) -> dict:
    volatility_pct = atr / close * 100 if close > 0 else 10
    if volatility_pct > 8:
        max_risk_pct = 1.0
    elif volatility_pct > 4:
        max_risk_pct = 2.0
    else:
        max_risk_pct = 3.0

    if vol_24h < 100_000:
        max_risk_pct *= 0.5
        liquidity_note = "ALACSONY likviditas! Felezo poziciomeret!"
    elif vol_24h < 500_000:
        max_risk_pct *= 0.75
        liquidity_note = "Kozepes likviditas."
    else:
        liquidity_note = "Megfelelo likviditas."

    risk_usd = portfolio * max_risk_pct / 100
    stop_pct = stop_dist / close * 100 if close > 0 else 5
    position_usd = risk_usd / (stop_pct / 100) if stop_pct > 0 else 0
    position_usd = min(position_usd, portfolio * 0.2)

    return {
        "max_risk_pct": max_risk_pct,
        "risk_usd": risk_usd,
        "position_usd": position_usd,
        "position_pct": position_usd / portfolio * 100 if portfolio > 0 else 0,
        "volatility_pct": volatility_pct,
        "liquidity_note": liquidity_note,
    }


def print_summary(df: pd.DataFrame, symbol: str, sr_levels: list,
                  binance_extra: dict | None = None) -> dict:
    last = df.iloc[-1]
    prev = df.iloc[-2]
    close = last["close"]
    rsi = last.get("rsi", 50)
    chg = ((close / prev["close"]) - 1) * 100
    score, rec = calc_swing_score(df, sr_levels)
    alerts = generate_alerts(df, sr_levels)
    fib = calc_fibonacci_retracement(df)
    atr = _calc_atr(df) if len(df) >= 15 else abs(last["high"] - last["low"])
    levels = _build_entry_exit(close, sr_levels, atr, fib)
    context = _build_context(df, sr_levels)
    scenarios = _build_scenarios(close, levels, atr, last.get("adx", 0))
    timing = _build_timing(df, sr_levels, atr)
    vol_24h = binance_extra.get("quote_volume_24h", 0) if binance_extra else df["volume"].iloc[-1]
    pos = _position_size(10000, close, atr, levels["risk"], vol_24h)
    large_candles = _detect_large_candles(df)

    W = 70
    B = Fore.CYAN + Style.BRIGHT
    G = Fore.GREEN + Style.BRIGHT
    R = Fore.RED + Style.BRIGHT
    Y = Fore.YELLOW + Style.BRIGHT
    M = Fore.MAGENTA + Style.BRIGHT
    D = Style.RESET_ALL

    score_color = G if score >= 60 else (Y if score >= 40 else R)

    # ---- AKCIO TERV ----
    print(f"\n{B}{'=' * W}")
    print(f" AKCIO TERV: {symbol}")
    print(f"{'=' * W}{D}")

    action_icon = G + "VETEL" if score >= 60 else (R + "ELADAS" if score < 40 else Y + "VARJ")
    print(f" {action_icon}{D} | Score: {score_color}{score}/100 ({rec}){D}")
    print(f" {timing}")
    if levels["nearest_support"]:
        print(f"   Belepes: {G}{_P(levels['entry'])}{D} zona")
        print(f"   Stop:    {R}{_P(levels['stop'])}{D} | "
              f"Target: {G}{_P(levels['target1'])}{D} | "
              f"R:R = 1:{levels['rr1']:.1f}")
    print(f"{B}{'=' * W}{D}")

    # ---- TECHNIKAI ----
    print(f"\n{M} TECHNIKAI INDIKATOROK{D}")
    print(f"{'-' * W}")
    print(f"  Ar: {_P(close)} ({chg:+.2f}%) | VWAP: {_P(last['vwap'])}")
    print(f"  SMA 20/50/200: {_P(last['sma_20'])} / {_P(last['sma_50'])}", end="")
    print(f" / {_P(last['sma_200'])}" if pd.notna(last["sma_200"]) else " / N/A")
    print(f"  BB: {_P(last['bb_lower'])} - {_P(last['bb_upper'])} "
          f"(%B: {(close - last['bb_lower']) / (last['bb_upper'] - last['bb_lower']):.0%})" if pd.notna(last["bb_upper"]) else "")
    print(f"  RSI: {rsi:.1f} | MACD: {last['macd']:.4g} (sig: {last['macd_signal']:.4g})")
    adx_v = last.get('adx', 0)
    print(f"  ADX: {adx_v:.1f} (+DI: {last['plus_di']:.1f} -DI: {last['minus_di']:.1f}) | Cross: {detect_golden_death_cross(df)}")

    # ---- KONTEXTUS ----
    print(f"\n{M} KONTEXTUS ERTELMEZES{D}")
    print(f"{'-' * W}")
    # Word wrap context at W chars
    words = context.split()
    line = " "
    for w in words:
        if len(line) + len(w) + 1 > W:
            print(line)
            line = "  " + w
        else:
            line += " " + w
    if line.strip():
        print(line)

    # ---- SZINTEK ----
    print(f"\n{M} BELEPESI / KILEPESI SZINTEK{D}")
    print(f"{'-' * W}")
    if levels["nearest_support"]:
        print(f"  Legkozelebbi tamasz:     {G}{_P(levels['nearest_support'])}{D}")
    if levels["nearest_resist"]:
        print(f"  Legkozelebbi ellenallas: {R}{_P(levels['nearest_resist'])}{D}")
    print(f"  Optimalis belepes:       {_P(levels['entry'])}")
    print(f"  Stop-loss:               {R}{_P(levels['stop'])}{D} (kockazat: {levels['risk'] / close * 100:.1f}%)")
    print(f"  Take-profit #1:          {G}{_P(levels['target1'])}{D} (+{levels['reward1'] / close * 100:.1f}%)")
    print(f"  Take-profit #2:          {G}{_P(levels['target2'])}{D}")
    print(f"  Risk/Reward:             1:{levels['rr1']:.1f}")

    # Fibonacci
    print(f"  Fibonacci szintek:")
    for name, val in sorted(fib.items(), key=lambda x: x[1], reverse=True)[:5]:
        marker = " <<" if abs(close - val) / close < 0.02 else ""
        print(f"    {name:<12} {_P(val)}{Y}{marker}{D}")

    # ---- SZCENARIÓ ----
    print(f"\n{M} SZCENARIÓ ELEMZES{D}")
    print(f"{'-' * W}")
    print(f"  {G}BULLISH ({scenarios['bull_prob']}%):{D} {scenarios['bull']}")
    print(f"  {R}BEARISH ({scenarios['bear_prob']}%):{D} {scenarios['bear']}")
    print(f"  {Y}SEMLEGES:{D} {scenarios['neutral']}")

    # ---- WHALE / SMART MONEY ----
    print(f"\n{M} WHALE / SMART MONEY JELZESEK{D}")
    print(f"{'-' * W}")
    vol = df["volume"].fillna(0)
    if len(vol) >= 30:
        avg_30 = vol.iloc[-31:-1].mean()
        avg_5 = vol.iloc[-5:].mean()
        ratio_5d = avg_5 / avg_30 if avg_30 > 0 else 0
        vol_tag = G + "NOVEKVO" if ratio_5d > 1.3 else (R + "CSOKKEN" if ratio_5d < 0.7 else Y + "STABIL")
        print(f"  5 napos vol / 30 napos atlag: {ratio_5d:.2f}x ({vol_tag}{D})")
    vp = calc_volume_profile(df, bins=15)
    if not vp.empty:
        peak_idx = vp["volume"].idxmax()
        peak_price = vp.loc[peak_idx, "price"]
        print(f"  Legnagyobb akkumulacios zona: {_P(peak_price)}")
    if large_candles:
        print(f"  Nagy gyertyak (>3% mozgas, utolso 5 nap):")
        for lc in large_candles:
            direction = G + "FEL" if lc["pct"] > 0 else R + "LE"
            print(f"    {lc['date']}: {direction} {abs(lc['pct']):.1f}%{D} (vol: {lc['vol_ratio']:.1f}x atlag)")
    else:
        print(f"  Nincs kiemelkedo gyertya az elmult 5 napban.")
    if binance_extra:
        if binance_extra.get("liquidity"):
            print(f"  DEX Liquidity: ${binance_extra['liquidity']:,.0f}")
        if binance_extra.get("trades_24h"):
            print(f"  24h tranzakciok: {binance_extra['trades_24h']:,}")

    # ---- KOCKAZAT ----
    print(f"\n{M} KOCKAZAT ERTEKELES{D}")
    print(f"{'-' * W}")
    print(f"  ATR (14 nap):        {_P(atr)} ({pos['volatility_pct']:.1f}%)")
    dd30 = _calc_max_drawdown(df, 30) if len(df) >= 30 else 0
    dd90 = _calc_max_drawdown(df, min(90, len(df)))
    print(f"  Max drawdown 30 nap: {R}{dd30:.1f}%{D}")
    print(f"  Max drawdown 90 nap: {R}{dd90:.1f}%{D}")
    print(f"  Likviditas:          {pos['liquidity_note']}")
    # Position sizing for 10k portfolio
    print(f"  Poziciomeret ($10,000 portfolio):")
    print(f"    Max kockazat:      {pos['max_risk_pct']:.1f}% (${pos['risk_usd']:.0f})")
    print(f"    Javasolt pozicio:  ${pos['position_usd']:,.0f} ({pos['position_pct']:.0f}% portfolio)")
    if atr > 0:
        days_to_target = abs(levels["reward1"]) / atr
        print(f"  Becsult ido celarig: ~{days_to_target:.0f} nap (ATR alapu becses)")

    # ---- IDOZITES ----
    print(f"\n{M} IDOZITES{D}")
    print(f"{'-' * W}")
    print(f"  {Y}{timing}{D}")

    # ---- ALERTS ----
    if alerts:
        print(f"\n{M} RIASZTASOK ({len(alerts)}){D}")
        print(f"{'-' * W}")
        for a in alerts:
            print(f"  {Y}>>{D} {a}")

    print(f"\n{B}{'=' * W}{D}")

    return {
        "symbol": symbol, "close": close, "change_pct": chg,
        "rsi": rsi, "adx": last.get("adx", 0), "macd": last.get("macd", 0),
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
# 13. SHORT SCANNER
# ============================================================================
STABLECOINS = {"USDC", "USDT", "DAI", "TUSD", "BUSD", "FDUSD", "USDP",
               "PYUSD", "GUSD", "FRAX", "LUSD", "SUSD", "EUSD", "USDJ"}

_short_scan_cache = {}
_short_scan_cache_time = None


def _detect_bearish_divergence(df: pd.DataFrame) -> dict:
    """RSI, MACD, Volume bearish divergencia detektalas."""
    result = {"rsi_div": False, "macd_div": False, "vol_div": False}
    if len(df) < 30:
        return result
    close = df["close"]
    rsi = df.get("rsi")
    macd = df.get("macd")
    vol = df["volume"].fillna(0)

    # Utobbi 30 nap lokalis csucsai
    n = min(60, len(df))
    recent = df.iloc[-n:]
    c = recent["close"].values
    from scipy.signal import argrelextrema as _are
    peaks = _are(c, np.greater_equal, order=5)[0]

    if len(peaks) >= 2:
        p1, p2 = peaks[-2], peaks[-1]
        # RSI divergencia: ar magasabb csucs, RSI alacsonyabb
        if rsi is not None and c[p2] > c[p1]:
            r_vals = recent["rsi"].values
            if len(r_vals) > max(p1, p2) and r_vals[p2] < r_vals[p1]:
                result["rsi_div"] = True
        # MACD divergencia
        if macd is not None and c[p2] > c[p1]:
            m_vals = recent["macd"].values
            if len(m_vals) > max(p1, p2) and m_vals[p2] < m_vals[p1]:
                result["macd_div"] = True

    # Volume divergencia: 10-nap trend - ar emelkedik, volumen csokken
    if len(df) >= 20:
        recent_close = close.iloc[-10:].mean()
        older_close = close.iloc[-20:-10].mean()
        recent_vol = vol.iloc[-10:].mean()
        older_vol = vol.iloc[-20:-10].mean()
        if older_close > 0 and older_vol > 0:
            if recent_close > older_close and recent_vol < older_vol * 0.8:
                result["vol_div"] = True

    return result


def _count_consecutive_green(df: pd.DataFrame) -> int:
    """Egymast koveto zold gyertyak szama visszafele."""
    count = 0
    for i in range(len(df) - 1, 0, -1):
        if df["close"].iloc[i] > df["close"].iloc[i - 1]:
            count += 1
        else:
            break
    return count


def calc_short_score(df: pd.DataFrame, sr_levels: list,
                     funding_rate: float | None = None) -> tuple[float, list[str]]:
    """Short score (0-100) es indokok listaja."""
    score = 0.0
    reasons = []
    last = df.iloc[-1]
    close = last["close"]

    # --- RSI ---
    rsi = last.get("rsi", 50)
    if rsi > 80:
        score += 25
        reasons.append(f"RSI {rsi:.0f} extrem tulvett")
    elif rsi > 70:
        score += 15
        reasons.append(f"RSI {rsi:.0f} tulvett")

    # --- MACD bearish crossover ---
    if len(df) >= 2:
        prev = df.iloc[-2]
        if last["macd"] < last["macd_signal"] and prev["macd"] >= prev["macd_signal"]:
            score += 20
            reasons.append("MACD bearish crossover")
        elif last["macd"] < last["macd_signal"]:
            score += 5
            reasons.append("MACD bearish")

    # --- Ellenallas kozeleben ---
    resists = [l for l in sr_levels if l > close]
    if resists:
        nearest_r = min(resists)
        if (nearest_r - close) / close < 0.02:
            score += 15
            reasons.append(f"Ellenallas kozel ({_P(nearest_r)})")

    # --- Volume divergencia (csokken vol + emelkedo ar) ---
    vol = df["volume"].fillna(0)
    if len(vol) >= 20:
        recent_vol = vol.iloc[-5:].mean()
        older_vol = vol.iloc[-20:-5].mean()
        recent_price = df["close"].iloc[-5:].mean()
        older_price = df["close"].iloc[-20:-5].mean()
        if older_vol > 0 and recent_price > older_price and recent_vol < older_vol * 0.7:
            score += 15
            reasons.append("Vol. divergencia (ar fel, vol le)")

    # --- ADX trend ---
    adx = last.get("adx", 0)
    plus_di = last.get("plus_di", 0)
    minus_di = last.get("minus_di", 0)
    if adx > 25 and minus_di > plus_di:
        score += 10
        reasons.append(f"ADX {adx:.0f} bearish (-DI > +DI)")

    # --- Bollinger felso felett ---
    bb_upper = last.get("bb_upper", None)
    if bb_upper and pd.notna(bb_upper) and close > bb_upper:
        score += 10
        reasons.append("Ar Bollinger felso felett")

    # --- Death cross ---
    cross = detect_golden_death_cross(df)
    if "DEATH CROSS" in cross:
        score += 10
        reasons.append("Death Cross!")
    elif "Bearish" in cross:
        score += 3

    # --- Ichimoku bearish ---
    sa = last.get("senkou_a", None)
    sb = last.get("senkou_b", None)
    if sa and sb and pd.notna(sa) and pd.notna(sb):
        cloud_top = max(sa, sb)
        if close < cloud_top:
            score += 10
            reasons.append("Ar Ichimoku felho alatt")

    # --- Funding rate ---
    if funding_rate is not None and funding_rate > 0.0005:
        score += 10
        reasons.append(f"Funding rate magas ({funding_rate * 100:.3f}%)")

    # --- Consecutive green candles ---
    green_count = _count_consecutive_green(df)
    if green_count >= 5:
        score += 5
        reasons.append(f"{green_count} egymast koveto zold gyertya")

    # --- Bearish divergenciak ---
    divs = _detect_bearish_divergence(df)
    if divs["rsi_div"]:
        score += 15
        reasons.append("RSI bearish divergencia")
    if divs["macd_div"]:
        score += 10
        reasons.append("MACD bearish divergencia")
    if divs["vol_div"] and "Vol. divergencia" not in " ".join(reasons):
        score += 5

    return min(score, 100), reasons


def _prefilter_short_candidates(tickers: list, min_volume: float,
                                exclude_stablecoins: bool) -> list[dict]:
    """Eloszures: 24h change, volume, stablecoin filter."""
    candidates = []
    for t in tickers:
        sym = t["symbol"]
        if not sym.endswith("USDT"):
            continue
        base = sym[:-4]
        if exclude_stablecoins and base in STABLECOINS:
            continue
        qv = float(t.get("quoteVolume", 0))
        if qv < min_volume:
            continue
        change_pct = float(t.get("priceChangePercent", 0))
        price = float(t.get("lastPrice", 0))
        # Pre-filter: valoszinu short jeloltek - mar emelkedtek VAGY magas a change
        # De mindenkit atnezunk aki megfelel a volumefilternek
        candidates.append({
            "symbol": sym, "base": base, "price": price,
            "change_pct": change_pct, "quote_volume": qv,
            "volume_24h": float(t.get("volume", 0)),
            "high_24h": float(t.get("highPrice", 0)),
            "low_24h": float(t.get("lowPrice", 0)),
        })
    return candidates


def run_short_scanner(days: int, interval: str, quote: str,
                      min_volume: float, min_score: float,
                      exclude_stablecoins: bool) -> None:
    """Short opportunity scanner - batch lekerdezesekkel."""
    global _short_scan_cache, _short_scan_cache_time

    B = Fore.CYAN + Style.BRIGHT
    R = Fore.RED + Style.BRIGHT
    G = Fore.GREEN + Style.BRIGHT
    Y = Fore.YELLOW + Style.BRIGHT
    M = Fore.MAGENTA + Style.BRIGHT
    D = Style.RESET_ALL

    print(f"\n{R}{'=' * 70}")
    print(f" SHORT OPPORTUNITY SCANNER")
    print(f"{'=' * 70}{D}")
    print(f"  Min volume: ${min_volume:,.0f} | Min score: {min_score}")
    print(f"  Idoszak: {days} nap | Interval: {interval}")

    # 1. Osszes ticker lekerese
    print(f"\n  Tickers lekerese...", end="", flush=True)
    now = datetime.now()
    cache_valid = (_short_scan_cache_time and
                   (now - _short_scan_cache_time).total_seconds() < 300 and
                   _short_scan_cache)

    if cache_valid:
        tickers = _short_scan_cache.get("tickers", [])
        print(f" (cached, {len(tickers)} par)")
    else:
        resp = requests.get(f"{BINANCE_BASE_URL}/ticker/24hr", timeout=15)
        resp.raise_for_status()
        tickers = resp.json()
        _short_scan_cache["tickers"] = tickers
        _short_scan_cache_time = now
        print(f" {len(tickers)} par")

    # 2. Pre-filter
    candidates = _prefilter_short_candidates(tickers, min_volume,
                                              exclude_stablecoins)
    print(f"  Szurt jeloltek (vol > ${min_volume:,.0f}, USDT): {len(candidates)}")

    if not candidates:
        print(f"  {Y}Nincs jelolt a filternek megfelelo.{D}")
        return

    # 3. Mindegyikre technikai elemzes
    results = []
    total = len(candidates)
    import time

    for idx, cand in enumerate(candidates):
        sym = cand["symbol"]
        # Progress
        if (idx + 1) % 5 == 0 or idx == total - 1:
            print(f"\r  Szkenneles... {idx + 1}/{total} par elemezve", end="", flush=True)

        try:
            df = fetch_binance_data(sym, days=days, interval=interval,
                                   quote=quote, quiet=True)
            if len(df) < 20:
                continue
            df = add_all_indicators(df)
            sr = get_sr_levels(df)

            # Funding rate (proba)
            fr = fetch_binance_funding_rate(sym, quote)

            short_score, reasons = calc_short_score(df, sr, fr)

            if short_score >= min_score:
                last = df.iloc[-1]
                results.append({
                    "symbol": sym,
                    "price": last["close"],
                    "change_pct": cand["change_pct"],
                    "rsi": last.get("rsi", 50),
                    "adx": last.get("adx", 0),
                    "macd": last.get("macd", 0),
                    "short_score": short_score,
                    "reasons": reasons,
                    "volume_24h": cand["quote_volume"],
                    "df": df,
                    "sr_levels": sr,
                    "funding_rate": fr,
                })

            # Rate limit: max ~20 req/sec biztonsagosan
            time.sleep(0.15)

        except Exception:
            continue

    print(f"\r  Szkenneles... {total}/{total} par elemezve - KESZ!     ")

    if not results:
        print(f"\n  {Y}Nincs short jelolt score >= {min_score} felett.{D}")
        return

    # 4. Rangsolas
    results.sort(key=lambda x: x["short_score"], reverse=True)
    top20 = results[:20]

    # 5. Osszefoglalo tabla
    print(f"\n{R}{'=' * 95}")
    print(f" TOP {len(top20)} SHORT LEHETOSEG (score >= {min_score})")
    print(f"{'=' * 95}{D}")
    hdr = f"  {'#':<4}{'Coin':<12}{'Ar':>14}{'24h%':>8}{'RSI':>7}{'Score':>8}  {'Fo indok'}"
    print(hdr)
    print(f"{'-' * 95}")
    for i, r in enumerate(top20):
        main_reason = r["reasons"][0] if r["reasons"] else "-"
        score_color = R if r["short_score"] >= 70 else (Y if r["short_score"] >= 50 else D)
        print(
            f"  {i + 1:<4}"
            f"{r['symbol']:<12}"
            f"${r['price']:>12,.4g}"
            f"{r['change_pct']:>+7.2f}%"
            f"{r['rsi']:>7.1f}"
            f"  {score_color}{r['short_score']:>5.0f}{D}"
            f"  {main_reason}"
        )
    print(f"{R}{'=' * 95}{D}")

    # 6. Top 5 reszletes elemzes
    print(f"\n{R}{'=' * 70}")
    print(f" RESZLETES SHORT ELEMZESEK (Top 5)")
    print(f"{'=' * 70}{D}")

    for i, r in enumerate(results[:5]):
        df = r["df"]
        sr = r["sr_levels"]
        last = df.iloc[-1]
        close = last["close"]
        atr = _calc_atr(df) if len(df) >= 15 else abs(last["high"] - last["low"])
        fib = calc_fibonacci_retracement(df)

        # Short-specifikus szintek
        resists = sorted([l for l in sr if l > close])
        supports = sorted([l for l in sr if l < close], reverse=True)

        entry = close
        stop = (resists[0] if resists else close + atr) + atr * 0.2
        target1 = supports[0] if supports else close - atr * 2
        target2 = supports[1] if len(supports) > 1 else target1 - atr

        risk = stop - close
        reward = close - target1
        rr = reward / risk if risk > 0 else 0

        # Kockazat szint
        vol_24h = r["volume_24h"]
        if atr / close > 0.08 or vol_24h < 200_000:
            risk_level = "MAGAS"
            risk_color = R
        elif atr / close > 0.04 or vol_24h < 1_000_000:
            risk_level = "KOZEPES"
            risk_color = Y
        else:
            risk_level = "ALACSONY"
            risk_color = G

        dd30 = _calc_max_drawdown(df, min(30, len(df)))

        print(f"\n{R}{'=' * 70}")
        print(f" SHORT TERV: {r['symbol']} (#{i + 1} - Score: {r['short_score']:.0f}/100)")
        print(f"{'=' * 70}{D}")
        print(f"  {R}SHORT BELEPES:{D} {_P(entry)} kozeleben")
        print(f"    Stop-loss:  {R}{_P(stop)}{D} (ellenallas + ATR felett)")
        print(f"    Target 1:   {G}{_P(target1)}{D} (legkozelebbi tamasz)")
        print(f"    Target 2:   {G}{_P(target2)}{D}")
        print(f"    R:R = 1:{rr:.1f}")
        print(f"  Indokok: {', '.join(r['reasons'][:4])}")
        print(f"  Kockazat: {risk_color}{risk_level}{D}")
        print(f"{'-' * 70}")
        print(f"  Ar: {_P(close)} | 24h: {r['change_pct']:+.2f}%")
        print(f"  RSI: {r['rsi']:.1f} | ADX: {r['adx']:.1f} | MACD: {r['macd']:.4g}")
        if r["funding_rate"] is not None:
            print(f"  Funding Rate: {r['funding_rate'] * 100:.4f}%")
        print(f"  24h Volume: ${vol_24h:,.0f}")
        print(f"  ATR: {_P(atr)} ({atr / close * 100:.1f}%)")
        print(f"  Max DD 30 nap: {dd30:.1f}%")

        # Divergenciak
        divs = _detect_bearish_divergence(df)
        div_list = []
        if divs["rsi_div"]:
            div_list.append("RSI")
        if divs["macd_div"]:
            div_list.append("MACD")
        if divs["vol_div"]:
            div_list.append("Volume")
        if div_list:
            print(f"  Divergenciak: {Y}{', '.join(div_list)}{D}")

        print(f"{R}{'=' * 70}{D}")


# ============================================================================
# 14. FOPROGRAM
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
    parser.add_argument("--scan-shorts", action="store_true",
                        help="Short opportunity scanner - osszes USDT par")
    parser.add_argument("--min-volume", type=float, default=1_000_000,
                        help="Minimum 24h volume USD-ben")
    parser.add_argument("--min-short-score", type=float, default=60,
                        help="Minimum short score (0-100, alapert: 60)")
    parser.add_argument("--exclude-stablecoins", action="store_true", default=True,
                        help="Stablecoinok kiszurese (alapert: igen)")
    args = parser.parse_args()

    source = args.source
    if args.scan_binance or args.scan_shorts:
        source = "binance"

    print(f"\nCrypto Swing Trading Analyzer")
    print(f"Idoszak: {args.days} nap | Forras: {source}"
          + (f" | Interval: {args.interval}" if source in ("binance", "alpha") else
             f" | Provider: {args.provider}"))

    if args.scan_shorts:
        run_short_scanner(args.days, args.interval, args.quote,
                          args.min_volume, args.min_short_score,
                          args.exclude_stablecoins)
        return

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
