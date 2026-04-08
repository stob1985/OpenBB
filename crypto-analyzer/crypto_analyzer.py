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

BINANCE_BASE_URL = "https://data-api.binance.vision/api/v3"
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
    limit: int = 0,
) -> list[str]:
    """USDT parok lekerese Binance-ról, volume alapjan szurve.
    limit=0: osszes par (exchangeInfo-val), limit>0: top N."""
    # ExchangeInfo a valid szimbolumokhoz
    valid = _get_valid_usdt_symbols()
    resp = requests.get(f"{BINANCE_BASE_URL}/ticker/24hr", timeout=15)
    resp.raise_for_status()
    tickers = resp.json()
    usdt_pairs = []
    for t in tickers:
        sym = t["symbol"]
        if not sym.endswith(quote):
            continue
        if valid and sym not in valid:
            continue
        base = sym[:-len(quote)]
        if base in STABLECOINS or base in FIAT_BASES:
            continue
        if any(sub in base for sub in _BLACKLIST_SUBSTRINGS):
            continue
        if any(base.endswith(suf) for suf in LEVERAGED_SUFFIXES):
            continue
        qv = float(t.get("quoteVolume", 0))
        if qv < min_volume_usd:
            continue
        usdt_pairs.append((sym, qv))
    usdt_pairs.sort(key=lambda x: x[1], reverse=True)
    if limit > 0:
        return [p[0] for p in usdt_pairs[:limit]]
    return [p[0] for p in usdt_pairs]


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
# 6. SCORING RENDSZER + PUMP PENALTY
# ============================================================================
def calc_pump_penalty(df: pd.DataFrame) -> tuple[float, list[str]]:
    """Pump & dump szuro: penalty pontok es indokok."""
    penalty = 0.0
    flags = []
    close = df["close"]
    vol = df["volume"].fillna(0)

    # 1. Egynapos nagy mozgas (utolso 5 nap)
    max_daily_move = 0
    for i in range(-min(5, len(df) - 1), 0):
        prev_c = df["close"].iloc[i - 1]
        if prev_c > 0:
            move = abs(df["close"].iloc[i] - prev_c) / prev_c
            max_daily_move = max(max_daily_move, move)
    if max_daily_move > 0.30:
        penalty += 30
        flags.append(f"1 napos mozgas {max_daily_move:.0%}")
    elif max_daily_move > 0.20:
        penalty += 20
        flags.append(f"1 napos mozgas {max_daily_move:.0%}")

    # 2. ATR volatilitas
    if len(df) >= 15:
        h, l, c = df["high"], df["low"], df["close"]
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr_pct = float(tr.rolling(14).mean().iloc[-1] / close.iloc[-1] * 100)
        if atr_pct > 15:
            penalty += 20
            flags.append(f"ATR {atr_pct:.0f}% extrem")
        elif atr_pct > 10:
            penalty += 10
            flags.append(f"ATR {atr_pct:.0f}% magas")

    # 3. 7 napos emelkedes >50%
    if len(close) >= 8:
        week_change = (close.iloc[-1] - close.iloc[-8]) / close.iloc[-8]
        if week_change > 0.50:
            penalty += 25
            flags.append(f"7 nap +{week_change:.0%}")

    # 4. 90 napos max drawdown > -50%
    n90 = min(90, len(df))
    if n90 >= 10:
        peak = close.iloc[-n90:].cummax()
        dd = ((close.iloc[-n90:] - peak) / peak).min()
        if dd < -0.50:
            penalty += 15
            flags.append(f"90d DD {dd:.0%}")

    # 5. Volume spike >10x (utolso 3 nap)
    if len(vol) >= 21:
        avg_vol = vol.iloc[-21:-3].mean()
        if avg_vol > 0:
            recent_max_vol = vol.iloc[-3:].max()
            vol_spike = recent_max_vol / avg_vol
            if vol_spike > 10:
                penalty += 20
                flags.append(f"Vol spike {vol_spike:.0f}x")

    # 6. Penny coin
    if close.iloc[-1] < 0.01:
        penalty += 10
        flags.append("Penny coin")

    # 7. Organikus trend teszt (20 nap napi hozam szoras)
    if len(close) >= 21:
        daily_returns = close.pct_change().iloc[-20:]
        std_pct = float(daily_returns.std() * 100)
        if std_pct > 10:
            penalty += 15
            flags.append(f"Szoras {std_pct:.1f}% instabil")

    # 8. Minimum trend-kor (SMA20 felett/alatt hany napja)
    sma20 = df.get("sma_20")
    if sma20 is not None and sma20.notna().any():
        above = close > sma20
        trend_days = 0
        for i in range(len(above) - 1, -1, -1):
            if above.iloc[i] == above.iloc[-1]:
                trend_days += 1
            else:
                break
        if trend_days < 5:
            flags.append(f"Trend {trend_days} napos (friss)")
            # Nem penalty, de a trend-pontokat felezzuk (jelzes a callernek)

    return penalty, flags


def calc_swing_score(df: pd.DataFrame, sr_levels: list) -> tuple:
    last = df.iloc[-1]
    raw_score = 50.0
    adx_val = last.get("adx", 0)
    plus_di = last.get("plus_di", 0)
    minus_di = last.get("minus_di", 0)

    # Trend-kor check (SMA20 felett hany napja)
    sma20 = df.get("sma_20")
    trend_mult = 1.0
    if sma20 is not None and sma20.notna().any():
        above = df["close"] > sma20
        trend_days = 0
        for i in range(len(above) - 1, -1, -1):
            if above.iloc[i] == above.iloc[-1]:
                trend_days += 1
            else:
                break
        if trend_days < 5:
            trend_mult = 0.5

    if adx_val > 25:
        ts = min(adx_val, 50) / 50 * 20 * trend_mult
        raw_score += ts if plus_di > minus_di else -ts
    rsi_val = last.get("rsi", 50)
    if rsi_val < 30:
        raw_score += 15 * (30 - rsi_val) / 30
    elif rsi_val > 70:
        raw_score -= 15 * (rsi_val - 70) / 30
    macd_val = last.get("macd", 0)
    macd_sig = last.get("macd_signal", 0)
    diff = abs(macd_val - macd_sig)
    contribution = min(15, 15 * diff / (abs(macd_sig) + 1e-9))
    raw_score += contribution if macd_val > macd_sig else -contribution
    vol = df["volume"].fillna(0)
    if len(vol) >= 21:
        avg = vol.iloc[-21:-1].mean()
        vr = vol.iloc[-1] / avg if avg > 0 else 1
        if vr > 1.5:
            raw_score += 10 * min(vr - 1, 1)
        elif vr < 0.5:
            raw_score -= 5
    close = last["close"]
    if sr_levels:
        ns = max([l for l in sr_levels if l <= close], default=None)
        nr = min([l for l in sr_levels if l > close], default=None)
        if ns and (close - ns) / close < 0.02:
            raw_score += 10
        if nr and (nr - close) / close < 0.02:
            raw_score -= 10
    raw_score = max(0, min(100, raw_score))

    # Pump penalty
    penalty, pump_flags = calc_pump_penalty(df)
    final_score = max(0, min(100, raw_score - penalty))

    labels = [(80, "Eros vetel"), (60, "Gyenge vetel"), (40, "Semleges"),
              (20, "Gyenge eladas"), (0, "Eros eladas")]
    rec = next(lb for th, lb in labels if final_score >= th)
    return round(final_score, 1), rec, round(raw_score, 1), round(penalty, 1), pump_flags


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




# ============================================================================
# ICHIMOKU RESZLETES ELEMZES
# ============================================================================
def analyze_ichimoku(df: pd.DataFrame) -> dict:
    """Ichimoku 5-jelzes elemzes."""
    if len(df) < 52:
        return {"score": 0, "bull": 0, "signals": [], "summary": "Nincs eleg adat (min 52 nap)."}
    close = df["close"].iloc[-1]
    tenkan = df.get("tenkan")
    kijun = df.get("kijun")
    sa = df.get("senkou_a")
    sb = df.get("senkou_b")
    if tenkan is None or kijun is None:
        return {"score": 0, "bull": 0, "signals": [], "summary": "Ichimoku nem szamitott."}

    t = float(tenkan.iloc[-1])
    k = float(kijun.iloc[-1])
    s_a = float(sa.iloc[-1]) if sa is not None and pd.notna(sa.iloc[-1]) else close
    s_b = float(sb.iloc[-1]) if sb is not None and pd.notna(sb.iloc[-1]) else close
    kumo_top = max(s_a, s_b)
    kumo_bot = min(s_a, s_b)

    signals = []
    bull_count = 0

    # 1. TK cross
    if t > k:
        signals.append(f"TK cross: Tenkan ({_P(t)}) FELETT Kijun ({_P(k)}) -> bullish")
        bull_count += 1
    else:
        signals.append(f"TK cross: Tenkan ({_P(t)}) ALATT Kijun ({_P(k)}) -> bearish")

    # 2. Ar vs Kumo
    if close > kumo_top:
        signals.append(f"Ar ({_P(close)}) a Kumo FELETT -> bullish")
        bull_count += 1
    elif close < kumo_bot:
        signals.append(f"Ar ({_P(close)}) a Kumo ALATT -> bearish")
    else:
        signals.append(f"Ar ({_P(close)}) a Kumo-BAN -> semleges/atmeneti")

    # 3. Chikou (ar vs 26 nappal korabbi ar)
    if len(df) > 26:
        chikou_ref = df["close"].iloc[-27]
        if close > chikou_ref:
            signals.append(f"Chikou: jelenlegi ar > 26 nappal ezelotti ({_P(chikou_ref)}) -> bullish")
            bull_count += 1
        else:
            signals.append(f"Chikou: jelenlegi ar < 26 nappal ezelotti ({_P(chikou_ref)}) -> bearish")

    # 4. Kumo jovobelei alakja (senkou A vs B trend)
    if sa is not None and len(sa) > 5:
        recent_sa = [float(x) for x in sa.iloc[-5:] if pd.notna(x)]
        recent_sb = [float(x) for x in sb.iloc[-5:] if pd.notna(x)]
        if len(recent_sa) >= 2 and len(recent_sb) >= 2:
            sa_trend = recent_sa[-1] - recent_sa[0]
            sb_trend = recent_sb[-1] - recent_sb[0]
            if sa_trend > 0 and sb_trend > 0:
                signals.append("Kumo jovoben: mindketto emelkedik -> bullish")
                bull_count += 1
            elif sa_trend < 0 and sb_trend < 0:
                signals.append("Kumo jovoben: mindketto csokken -> bearish")
            else:
                # Twist kozelit
                if abs(recent_sa[-1] - recent_sb[-1]) < abs(recent_sa[0] - recent_sb[0]) * 0.3:
                    signals.append("Kumo TWIST kozelit! -> potencialis trendvaltas 1-5 napon belul")
                else:
                    signals.append("Kumo vegyes iranyu -> atmeneti")

    # 5. Kumo vastagsag
    thickness = abs(s_a - s_b)
    thick_pct = thickness / close * 100 if close > 0 else 0
    if thick_pct > 5:
        signals.append(f"Kumo vastag ({thick_pct:.1f}%) -> eros S/R zona, nehezen torheto at")
        bull_count += 1 if close > kumo_top else 0
    elif thick_pct < 1:
        signals.append(f"Kumo vekony ({thick_pct:.1f}%) -> gyenge S/R, konnyen attorheto")
    else:
        signals.append(f"Kumo kozepes ({thick_pct:.1f}%)")

    bear_count = len(signals) - bull_count
    if bull_count >= 4:
        summary = "Ichimoku: EROS BULLISH"
    elif bull_count >= 3:
        summary = "Ichimoku: BULLISH"
    elif bull_count >= 2:
        summary = "Ichimoku: ENYHE BULLISH"
    elif bear_count >= 4:
        summary = "Ichimoku: EROS BEARISH"
    elif bear_count >= 3:
        summary = "Ichimoku: BEARISH"
    else:
        summary = "Ichimoku: SEMLEGES"

    score_mod = (bull_count - 2.5) * 4  # -10 to +10

    return {"score": round(score_mod), "bull": bull_count, "bear": bear_count,
            "signals": signals, "summary": summary}


# ============================================================================
# BOLLINGER BANDS RESZLETES ELEMZES
# ============================================================================
def analyze_bollinger(df: pd.DataFrame) -> dict:
    """Bollinger Bands reszletes elemzes: squeeze, walk, %B."""
    if len(df) < 25 or "bb_upper" not in df or not pd.notna(df["bb_upper"].iloc[-1]):
        return {"score": 0, "signals": [], "summary": "Nincs eleg BB adat."}

    close = df["close"]
    upper = df["bb_upper"]
    lower = df["bb_lower"]
    mid = df["bb_middle"]

    c = close.iloc[-1]
    u = upper.iloc[-1]
    l = lower.iloc[-1]
    width = u - l

    signals = []
    score_mod = 0

    # %B
    pct_b = (c - l) / (u - l) if (u - l) > 0 else 0.5
    if pct_b > 1.0:
        signals.append(f"BB %B = {pct_b:.2f} — ar a FELSO sav FELETT -> tulvett, short jelzes")
        score_mod -= 10  # short bonus / long penalty
    elif pct_b < 0.0:
        signals.append(f"BB %B = {pct_b:.2f} — ar az ALSO sav ALATT -> tuleladott, long jelzes")
        score_mod += 10
    elif 0.4 <= pct_b <= 0.6:
        signals.append(f"BB %B = {pct_b:.2f} — kozepsav, semleges")
    else:
        tag = "felso fele" if pct_b > 0.5 else "also fele"
        signals.append(f"BB %B = {pct_b:.2f} — {tag}")

    # Squeeze detektalas
    if len(df) >= 25:
        widths = (upper - lower).iloc[-25:]
        current_width = widths.iloc[-1]
        pct_rank = (widths < current_width).sum() / len(widths)
        if pct_rank <= 0.2:
            squeeze_days = 0
            for i in range(len(widths) - 1, -1, -1):
                if widths.iloc[i] <= widths.quantile(0.25):
                    squeeze_days += 1
                else:
                    break
            signals.append(f"BB SQUEEZE aktiv! Szalagok {squeeze_days} napja szukek "
                           f"(legszukebb 20%) -> kitores varhato 1-5 napon belul")
            score_mod += 5
        elif pct_rank >= 0.8:
            signals.append("BB SZELES — magas volatilitas, trend folytatodik")

    # Bollinger Walk
    walk_up = 0
    walk_down = 0
    for i in range(-min(7, len(df)), 0):
        if close.iloc[i] >= upper.iloc[i] * 0.98:
            walk_up += 1
        elif close.iloc[i] <= lower.iloc[i] * 1.02:
            walk_down += 1
    if walk_up >= 3:
        signals.append(f"Bollinger Walk FELFELÉ ({walk_up} nap) — eros bullish trend, "
                       "NE shortold!")
        score_mod += 5
    elif walk_down >= 3:
        signals.append(f"Bollinger Walk LEFELÉ ({walk_down} nap) — eros bearish trend, "
                       "NE longold!")
        score_mod -= 5

    # Szelesseg trend
    if len(df) >= 10:
        w5 = (upper.iloc[-5:] - lower.iloc[-5:]).mean()
        w10 = (upper.iloc[-10:-5] - lower.iloc[-10:-5]).mean()
        if w10 > 0:
            if w5 < w10 * 0.8:
                signals.append("BB szukuloben -> alacsony volatilitas, kitores kozeleg")
            elif w5 > w10 * 1.2:
                signals.append("BB szelesedoben -> novekvo volatilitas")

    summary = f"BB %B: {pct_b:.0%}"
    if any("SQUEEZE" in s for s in signals):
        summary += " + SQUEEZE"
    if walk_up >= 3:
        summary += " + Walk UP"
    elif walk_down >= 3:
        summary += " + Walk DOWN"

    return {"score": score_mod, "pct_b": pct_b, "signals": signals, "summary": summary}


# ============================================================================
# ELLIOTT WAVE DETEKTALAS
# ============================================================================
def analyze_elliott(df: pd.DataFrame) -> dict:
    """Elliott Wave struktura automatikus detektalas."""
    if len(df) < 40:
        return {"score": 0, "wave": "?", "waves": [],
                "summary": "Nincs eleg adat Elliott elemzeshez."}

    close = df["close"].values
    from scipy.signal import argrelextrema as _are

    # Lokalis csucsok es melypontok
    order = max(5, len(df) // 15)
    max_idx = _are(close, np.greater_equal, order=order)[0]
    min_idx = _are(close, np.less_equal, order=order)[0]

    # Extremumok idorendben
    extrema = []
    for i in max_idx:
        extrema.append(("H", i, close[i]))
    for i in min_idx:
        extrema.append(("L", i, close[i]))
    extrema.sort(key=lambda x: x[1])

    if len(extrema) < 4:
        return {"score": 0, "wave": "?", "waves": [],
                "summary": "Nem talalhato elegendo extremum az EW elemzeshez."}

    # Alternalo H/L sorozat epitese
    filtered = [extrema[0]]
    for e in extrema[1:]:
        if e[0] != filtered[-1][0]:
            filtered.append(e)
        else:
            # Ugyanolyan tipus: a magasabb H-t vagy alacsonyabb L-t tartjuk
            if e[0] == "H" and e[2] > filtered[-1][2]:
                filtered[-1] = e
            elif e[0] == "L" and e[2] < filtered[-1][2]:
                filtered[-1] = e

    # Impulziv hullamok keresese (5 swing: H-L-H-L-H vagy L-H-L-H-L)
    waves = []
    current_wave = "?"
    wave_details = []

    # Megprobaljuk az utolso 5-7 extremumbol osszeallitani
    if len(filtered) >= 5:
        last5 = filtered[-5:]
        prices = [x[2] for x in last5]
        types = [x[0] for x in last5]
        indices = [x[1] for x in last5]

        # Emelkedo impulzus: L-H-L-H-L pattern ahol H-k emelkednek es L-k emelkednek
        if types[0] == "L" and types[-1] == "L":
            # Potencialis 1-2-3-4-5 (emelkedo)
            w1_start, w1_end = prices[0], prices[1]
            w2_end = prices[2]
            w3_end = prices[3]
            w4_end = prices[4] if len(prices) > 4 else close[-1]

            w1_size = w1_end - w1_start
            w3_size = w3_end - w2_end if len(prices) > 3 else 0
            valid = True
            reasons = []

            if w1_size <= 0:
                valid = False
            if w2_end < w1_start:
                valid = False
                reasons.append("Wave 2 az Wave 1 ala ment")
            if w3_size > 0 and w1_size > 0:
                if w3_size < w1_size * 0.5:
                    reasons.append("Wave 3 tul rovid")

            if valid and w1_size > 0:
                current_pos = close[-1]
                if current_pos > w3_end:
                    current_wave = "5"
                elif current_pos > w2_end:
                    current_wave = "3"
                elif current_pos > w1_start:
                    current_wave = "2 vagy 4"
                else:
                    current_wave = "A/B/C"

                # Fibonacci extensions
                w3_target = w2_end + w1_size * 1.618
                w5_target = w4_end + w1_size * 1.0 if len(prices) > 4 else w3_end + w1_size * 0.618

                for i, (t, idx, p) in enumerate(last5):
                    wave_details.append({
                        "num": i + 1, "type": t,
                        "price": p, "idx": idx,
                        "date": df.index[idx].strftime("%m-%d") if idx < len(df.index) else "?"
                    })
                waves = wave_details

        # Csökkeno impulzus: H-L-H-L-H
        elif types[0] == "H" and types[-1] == "H":
            w1_start, w1_end = prices[0], prices[1]
            w1_size = w1_start - w1_end  # Lefele
            if w1_size > 0:
                current_wave = "bearish_impulse"
                for i, (t, idx, p) in enumerate(last5):
                    wave_details.append({
                        "num": i + 1, "type": t,
                        "price": p, "idx": idx,
                        "date": df.index[idx].strftime("%m-%d") if idx < len(df.index) else "?"
                    })
                waves = wave_details

    # Score modosito
    score_mod = 0
    if current_wave == "3":
        score_mod = 10
    elif current_wave == "5":
        score_mod = -5
    elif current_wave in ("A/B/C", "bearish_impulse"):
        score_mod = -10
    elif current_wave == "2 vagy 4":
        score_mod = 5

    # Summary szoveg
    if current_wave == "?":
        summary = ("Az Elliott Wave struktura nem egyertelmu — az elmult idoszak "
                   "mozgasa nem mutat tiszta 5 hullamod mintat. Ez altalaban "
                   "oldalazó, range-bound piacra utal.")
    elif current_wave == "3":
        summary = "Elliott: az ar a WAVE 3-ban mozog — ez a legerosebb hullam."
    elif current_wave == "5":
        summary = "Elliott: az ar a WAVE 5-ben — az utolso hullam, csucs kozel lehet."
    elif current_wave == "2 vagy 4":
        summary = "Elliott: korrekcios hullam (Wave 2/4) — potencialis belepesi pont."
    elif current_wave == "bearish_impulse":
        summary = "Elliott: bearish impulzus — lefelemozgas dominansabb."
    elif current_wave == "A/B/C":
        summary = "Elliott: korrekcios A/B/C struktura — bearish retrace."
    else:
        summary = f"Elliott: Wave {current_wave}"

    return {"score": score_mod, "wave": current_wave, "waves": waves,
            "summary": summary}


# ============================================================================
# KONVERGENCIA ALERT
# ============================================================================
def analyze_convergence(ichi: dict, bb: dict, ew: dict, mtf: dict | None) -> dict:
    """Kombinalt jelzes — hany rendszer egyezik."""
    bull = 0
    bear = 0
    details = []

    # Ichimoku
    if ichi.get("bull", 0) >= 3:
        bull += 1
        details.append(f"Ichimoku: {ichi['bull']}/5 bullish")
    elif ichi.get("bear", 0) >= 3:
        bear += 1
        details.append(f"Ichimoku: {ichi.get('bear',0)}/5 bearish")

    # Bollinger
    bb_pct = bb.get("pct_b", 0.5)
    if bb_pct > 0.8:
        bear += 1
        details.append(f"Bollinger: %B {bb_pct:.0%} (tulvett zona)")
    elif bb_pct < 0.2:
        bull += 1
        details.append(f"Bollinger: %B {bb_pct:.0%} (tuleladott zona)")
    if any("SQUEEZE" in s for s in bb.get("signals", [])):
        details.append("Bollinger: SQUEEZE aktiv")

    # Elliott
    ew_wave = ew.get("wave", "?")
    if ew_wave in ("3",):
        bull += 1
        details.append(f"Elliott: Wave {ew_wave} (bullish)")
    elif ew_wave in ("5", "A/B/C", "bearish_impulse"):
        bear += 1
        details.append(f"Elliott: Wave {ew_wave} (bearish)")

    # MTF
    if mtf:
        if mtf.get("bull_count", 0) >= 3:
            bull += 1
            details.append(f"MTF: {mtf['bull_count']}/4 bullish")
        elif mtf.get("bear_count", 0) >= 3:
            bear += 1
            details.append(f"MTF: {mtf['bear_count']}/4 bearish")

    total = max(bull, bear)
    if total >= 3:
        signal = "KONVERGENCIA BULLISH" if bull > bear else "KONVERGENCIA BEARISH"
        score_mod = 15 if bull > bear else -15
    elif total == 2:
        signal = "RESZLEGES egyezes"
        score_mod = 5 if bull > bear else -5
    else:
        signal = "VEGYES — ne kereskedj"
        score_mod = 0

    return {
        "bull": bull, "bear": bear, "total": total,
        "signal": signal, "score": score_mod, "details": details,
    }


def _wrap(text: str, width: int = 68, indent: str = "  ") -> str:
    """Szoveg sorokra tordelese."""
    words = text.split()
    lines = []
    line = indent
    for w in words:
        if len(line) + len(w) + 1 > width:
            lines.append(line)
            line = indent + w
        else:
            line += (" " if line.strip() else "") + w
    if line.strip():
        lines.append(line)
    return "\n".join(lines)


def _build_narrative(df, last, close, rsi, atr, chg, sr_levels, fib, mtf_result,
                     large_candles, vol_24h) -> str:
    """Reszletes technikai narrativa generalasa."""
    parts = []
    # Arfolyam mozgas
    n7 = min(8, len(df))
    week_chg = (close - df["close"].iloc[-n7]) / df["close"].iloc[-n7] * 100
    n30 = min(31, len(df))
    month_chg = (close - df["close"].iloc[-n30]) / df["close"].iloc[-n30] * 100
    high_30 = df["high"].iloc[-n30:].max()
    low_30 = df["low"].iloc[-n30:].min()

    if week_chg > 5:
        parts.append(f"Az elmult 7 napban {week_chg:+.1f}%-ot emelkedett.")
    elif week_chg < -5:
        parts.append(f"Az elmult 7 napban {week_chg:+.1f}%-ot esett.")
    else:
        parts.append(f"Az elmult 7 napban {week_chg:+.1f}% valtozas — oldalazas.")

    parts.append(f"30 napos tartomany: {_P(low_30)} - {_P(high_30)} ({month_chg:+.1f}% honap).")

    # MACD tortenet
    macd_v = last.get("macd", 0)
    macd_s = last.get("macd_signal", 0)
    macd_h = last.get("macd_hist", 0)
    if len(df) >= 3:
        prev_h = df["macd_hist"].iloc[-2]
        if macd_v > macd_s:
            if prev_h < 0 and macd_h > 0:
                parts.append("A MACD epp most vegzett bullish crossovert — friss veteli jelzes.")
            elif macd_h > prev_h:
                parts.append("A MACD hisztogram egyre magasabban pozitiv — erosodo bullish momentum.")
            else:
                parts.append("A MACD bullish, de a hisztogram csokken — a momentum lassul.")
        else:
            if prev_h > 0 and macd_h < 0:
                parts.append("A MACD epp most vegzett bearish crossovert — friss eladasi jelzes.")
            elif macd_h < prev_h:
                parts.append("A MACD hisztogram egyre melyebben negativ — gyorsulo bearish momentum.")
            else:
                parts.append("A MACD bearish, de a hisztogram emelkedik — az eladoi nyomas csokkenhet.")

    # RSI
    if rsi > 75:
        parts.append(f"Az RSI {rsi:.0f} — erosen tulvett zona, korrekcio valoszinu 1-3 napon belul.")
    elif rsi > 65:
        parts.append(f"Az RSI {rsi:.0f} — enyhen tulvett, meg van mozgaster felfelé de ovatosan.")
    elif rsi < 25:
        parts.append(f"Az RSI {rsi:.0f} — erosen tuleladott, pattanas barmikor johet.")
    elif rsi < 35:
        parts.append(f"Az RSI {rsi:.0f} — tuleladott zona kozeleben, a bearish momentum kimerulhet.")
    else:
        parts.append(f"Az RSI {rsi:.0f} — semleges tartomanyban, nincs extrem jelzes.")

    # Volume
    vol = df["volume"].fillna(0)
    if len(vol) >= 20:
        avg_5 = vol.iloc[-5:].mean()
        avg_30 = vol.iloc[-min(30,len(vol)):].mean()
        if avg_30 > 0:
            vr = avg_5 / avg_30
            if vr > 1.5:
                parts.append(f"A volume az elmult 5 napban {(vr-1)*100:.0f}%-kal MAGASABB mint a 30 napos atlag — novekvo erdeklodes.")
            elif vr < 0.6:
                parts.append(f"A volume az elmult 5 napban {(1-vr)*100:.0f}%-kal ALACSONYABB mint a 30 napos atlag — nincs erdeklodes.")
            else:
                parts.append("A volume stabil az atlag korul.")

    # MTF
    if mtf_result:
        bc = mtf_result.get("bull_count", 0)
        brc = mtf_result.get("bear_count", 0)
        sig = mtf_result.get("signal", "")
        if bc == 4:
            parts.append("Mind a 4 timeframe (1h, 4h, 1d, 1w) bullish — ez ritka es nagyon eros jelzes.")
        elif brc == 4:
            parts.append("Mind a 4 timeframe (1h, 4h, 1d, 1w) bearish — ez ritka es nagyon eros jelzes. Az Ichimoku felho alatt van az ar minden TF-en.")
        elif "WEAK" in sig:
            parts.append(f"A timeframe-ek megosztottak ({bc}/4 bull, {brc}/4 bear) — nincs egyertelmu irany, nagyobb kockazat.")

    # Tamasz/ellenallas
    supports = sorted([l for l in sr_levels if l < close], reverse=True)
    resists = sorted([l for l in sr_levels if l > close])
    if supports:
        parts.append(f"Legkozelebbi tamasz: {_P(supports[0])} ({(close-supports[0])/close*100:.1f}% tavolsag).")
        if len(supports) > 1:
            parts.append(f"Ha ezt elveszti, a kovetkezo szint {_P(supports[1])}.")
    if resists:
        parts.append(f"Legkozelebbi ellenallas: {_P(resists[0])} ({(resists[0]-close)/close*100:.1f}% tavolsag).")

    # Likviditas
    if vol_24h > 10_000_000:
        parts.append(f"A 24h volume ${vol_24h/1e6:.1f}M — kivaloan likvid, nagy poziciok is kezelhetok.")
    elif vol_24h > 1_000_000:
        parts.append(f"A 24h volume ${vol_24h/1e6:.1f}M — elfogadhato likviditas.")
    elif vol_24h > 100_000:
        parts.append(f"A 24h volume ${vol_24h/1e3:.0f}K — alacsony likviditas, spread figyelese fontos!")
    else:
        parts.append(f"A 24h volume ${vol_24h/1e3:.0f}K — NAGYON alacsony, csuszas kockazat!")

    return " ".join(parts)


def _build_detailed_scenarios(df, close, sr_levels, atr, adx, rsi, mtf_result, fib) -> str:
    """Reszletes szcenarió elemzes."""
    G = Fore.GREEN + Style.BRIGHT
    R = Fore.RED + Style.BRIGHT
    Y = Fore.YELLOW + Style.BRIGHT
    D = Style.RESET_ALL

    supports = sorted([l for l in sr_levels if l < close], reverse=True)
    resists = sorted([l for l in sr_levels if l > close])
    s1 = supports[0] if supports else close - atr * 2
    s2 = supports[1] if len(supports) > 1 else s1 - atr
    r1 = resists[0] if resists else close + atr * 2
    r2 = resists[1] if len(resists) > 1 else r1 + atr
    sma50 = df.get("sma_50")
    sma50_v = float(sma50.iloc[-1]) if sma50 is not None and sma50.notna().iloc[-1] else close

    mtf_bull = mtf_result.get("bull_count", 2) if mtf_result else 2
    bull_prob = 55 if mtf_bull >= 3 else (40 if adx > 25 else 35)
    bear_prob = 100 - bull_prob - 15
    neutral_prob = 15

    lines = []
    lines.append(f"\n  {G}BULLISH SZCENARIÓ ({bull_prob}% esely):{D}")
    lines.append(_wrap(f"Ha az ar attori a(z) {_P(r1)} ellenallast volumennel, a kovetkezo celszint {_P(r2)}. "
        f"Ehhez az RSI-nek 50 felett kell maradnia es a MACD-nak bullish-nek. "
        f"Az SMA50 ({_P(sma50_v)}) fontos kozeptavu tamasz — amig felette van az ar, a long setup ervenyes. "
        f"{'A 4/4 bullish MTF tamogatja ezt a szcenáriot.' if mtf_bull==4 else 'Az MTF nem teljes megerosites, ovatosan.' if mtf_bull<3 else 'A 3/4 MTF tamogatja a bullish iranyt.'}"))

    lines.append(f"\n  {R}BEARISH SZCENARIÓ ({bear_prob}% esely):{D}")
    lines.append(_wrap(f"Ha elveszti a(z) {_P(s1)} tamaszt, a kovetkezo support {_P(s2)}, "
        f"ami {(close-s2)/close*100:.1f}%-os esest jelent. "
        f"{'A 4/4 bearish MTF erositi ezt a szcenáriot — az eladoi nyomas minden idotavon jelen van.' if mtf_bull==0 else ''} "
        f"Ha a volume megnovekedik az eses soran, az panik-eladasra utal es gyorsithatja a mozgast. "
        f"Stop-loss legyen {_P(s1)} kozeleben long, vagy {_P(r1)} kozeleben short pozicionál."))

    lines.append(f"\n  {Y}KONSZOLIDACIOS SZCENARIÓ ({neutral_prob}% esely):{D}")
    lines.append(_wrap(f"Az ar {_P(s1)} - {_P(r1)} tartomanyban ragad. "
        f"Ez swing tradinghez a legrosszabb, mert a toke le van kotve mozgas nelkul. "
        f"Ha {'7' if atr/close > 0.05 else '10'} nap utan nincs elmozdulas, zard a poziciot. "
        f"Az ATR ({atr/close*100:.1f}%) alapjan naponta atlagosan ennyit mozog az ar."))

    return "\n".join(lines)


def _build_detailed_entry(close, sr_levels, atr, rsi, fib, sma20) -> str:
    """Reszletes belepesi strategia."""
    G = Fore.GREEN + Style.BRIGHT
    D = Style.RESET_ALL
    supports = sorted([l for l in sr_levels if l < close], reverse=True)
    resists = sorted([l for l in sr_levels if l > close])

    lines = []
    # a) Azonnali
    lines.append(f"  a) AZONNALI belepes: {_P(close)}")
    lines.append(f"     Pro: nem maradsz le ha folytatodik a mozgas")
    lines.append(f"     Kontra: nincs megerosites, kicsit magasabb kockazat")

    # b) Pullback / visszateszt
    if supports:
        pb = supports[0]
        lines.append(f"  b) PULLBACK belepes: {_P(pb)} zona (tamasz visszateszt)")
        lines.append(f"     Pro: jobb ar, teszteli a tamaszt, alacsonyabb kockazat")
        lines.append(f"     Kontra: lehet hogy nem jon vissza ide")
    elif resists:
        pb = resists[0]
        lines.append(f"  b) VISSZATESZT belepes: {_P(pb)} zona")
        lines.append(f"     Pro: megerositi az ellenallast tamaszként (short: tamaszként)")
        lines.append(f"     Kontra: rosszabb ar ha nem jon vissza")

    # c) Breakdown/breakout
    if resists:
        lines.append(f"  c) BREAKOUT belepes: {_P(resists[0])} folott (long) vagy alatta (short)")
        lines.append(f"     Pro: megerositett mozgas, eros momentum")
        lines.append(f"     Kontra: rosszabb entry, kisebb R:R")

    # Ajánlás
    if rsi > 70:
        lines.append(f"\n  AJÁNLÁS: (b) opció — RSI {rsi:.0f} tulvett, varj pullback-re.")
    elif rsi < 30:
        lines.append(f"\n  AJÁNLÁS: (a) opció — RSI {rsi:.0f} tuleladott, azonnali belepes indokolt.")
    else:
        lines.append(f"\n  AJÁNLÁS: (b) opció a legoptimálisabb swing tradinghez.")

    return "\n".join(lines)


def _build_detailed_exit(close, sr_levels, atr, fib) -> str:
    """Reszletes kilepesi strategia."""
    G = Fore.GREEN + Style.BRIGHT
    R = Fore.RED + Style.BRIGHT
    D = Style.RESET_ALL
    supports = sorted([l for l in sr_levels if l < close], reverse=True)
    resists = sorted([l for l in sr_levels if l > close])
    fib_sorted = sorted(fib.items(), key=lambda x: x[1])

    lines = []
    targets = []
    if resists:
        targets.append((resists[0], "legkozelebbi ellenallas"))
    if len(resists) > 1:
        targets.append((resists[1], "masodik ellenallas"))
    for fn, fv in fib_sorted:
        if fv > close * 1.05 and len(targets) < 3:
            targets.append((fv, fn))

    if targets:
        pct_alloc = [33, 33, 34] if len(targets) >= 3 else [50, 50] if len(targets) == 2 else [100]
        for i, (t, reason) in enumerate(targets[:3]):
            pct_gain = (t - close) / close * 100
            lines.append(f"  Target {i+1}: {_P(t)} (+{pct_gain:.1f}%) — zard a pozicio {pct_alloc[i]}%-at")
            lines.append(f"    Miert itt: {reason}")

    lines.append(f"\n  TRAILING STOP:")
    lines.append(f"    Target 1 elerese utan huzd be a stopot az entry arra (breakeven).")
    lines.append(f"    Igy a maradek pozicion mar nem tudsz vesziteni.")

    days_est = abs(targets[0][0] - close) / atr if targets and atr > 0 else 0
    lines.append(f"\n  IDOZITES:")
    lines.append(f"    ATR ({atr/close*100:.1f}%) alapjan Target 1 ~{days_est:.0f} nap alatt erheto el.")
    lines.append(f"    Ha {max(10, int(days_est*2))} nap utan nincs elmozdulas, zard a poziciot.")

    return "\n".join(lines)


def _build_detailed_risk(df, close, atr, vol_24h, rsi, mtf_result,
                         penalty, pump_flags) -> str:
    """Reszletes kockazati faktorok."""
    G = Fore.GREEN + Style.BRIGHT
    R = Fore.RED + Style.BRIGHT
    Y = Fore.YELLOW + Style.BRIGHT
    D = Style.RESET_ALL

    factors = []
    # Positiv
    if mtf_result:
        bc = mtf_result.get("bull_count", 0)
        brc = mtf_result.get("bear_count", 0)
        if bc >= 3:
            factors.append((True, f"{bc}/4 bullish MTF — eros megerosites"))
        elif brc >= 3:
            factors.append((True, f"{brc}/4 bearish MTF — eros megerosites"))
        else:
            factors.append((None, f"MTF vegyes ({bc}/4 bull) — nincs egyertelmu irany"))

    macd_v = df["macd"].iloc[-1] if "macd" in df else 0
    macd_s = df["macd_signal"].iloc[-1] if "macd_signal" in df else 0
    if abs(macd_v - macd_s) > 0:
        factors.append((True, "MACD es jelzes egyeznek"))

    if not pump_flags:
        factors.append((True, "Nincs pump penalty — organikus mozgas"))
    else:
        factors.append((False, f"Pump penalty: {', '.join(pump_flags)}"))

    # Figyelmeztetesek
    atr_pct = atr / close * 100
    if atr_pct > 10:
        factors.append((False, f"ATR {atr_pct:.1f}% — extrem volatilitas"))
    elif atr_pct > 6:
        factors.append((None, f"ATR {atr_pct:.1f}% — kozepes volatilitas"))
    else:
        factors.append((True, f"ATR {atr_pct:.1f}% — elfogadhato volatilitas"))

    if vol_24h > 5_000_000:
        factors.append((True, f"24h volume ${vol_24h/1e6:.1f}M — jo likviditas"))
    elif vol_24h > 500_000:
        factors.append((None, f"24h volume ${vol_24h/1e3:.0f}K — elfogadhato"))
    else:
        factors.append((False, f"24h volume ${vol_24h/1e3:.0f}K — alacsony likviditas"))

    dd90 = _calc_max_drawdown(df, min(90, len(df)))
    if dd90 < -50:
        factors.append((False, f"Max drawdown 90 nap: {dd90:.0f}% — magas kockazat"))
    elif dd90 < -30:
        factors.append((None, f"Max drawdown 90 nap: {dd90:.0f}%"))
    else:
        factors.append((True, f"Max drawdown 90 nap: {dd90:.0f}% — elfogadhato"))

    lines = []
    for is_good, text in factors:
        icon = f"{G}V{D}" if is_good is True else (f"{R}X{D}" if is_good is False else f"{Y}!{D}")
        lines.append(f"  {icon} {text}")

    # Osszesites
    bads = sum(1 for g, _ in factors if g is False)
    goods = sum(1 for g, _ in factors if g is True)
    if bads >= 3:
        level = f"{R}MAGAS{D}"
    elif bads >= 2 or goods < 2:
        level = f"{Y}KOZEPES{D}"
    else:
        level = f"{G}ALACSONY{D}"
    lines.append(f"\n  OSSZESITETT KOCKAZAT: {level}")

    return "\n".join(lines)


def print_summary(df: pd.DataFrame, symbol: str, sr_levels: list,
                  binance_extra: dict | None = None,
                  mtf_result: dict | None = None) -> dict:
    last = df.iloc[-1]
    prev = df.iloc[-2]
    close = last["close"]
    rsi = last.get("rsi", 50)
    chg = ((close / prev["close"]) - 1) * 100
    score, rec, raw_score, penalty, pump_flags = calc_swing_score(df, sr_levels)
    mtf_mod = 0
    if mtf_result:
        mtf_mod = mtf_score_modifier(mtf_result)
        score = max(0, min(100, score + mtf_mod))
        labels = [(80, "Eros vetel"), (60, "Gyenge vetel"), (40, "Semleges"),
                  (20, "Gyenge eladas"), (0, "Eros eladas")]
        rec = next(lb for th, lb in labels if score >= th)
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

    # Advanced modules
    ichi_analysis = analyze_ichimoku(df)
    bb_analysis = analyze_bollinger(df)
    ew_analysis = analyze_elliott(df)
    conv_analysis = analyze_convergence(ichi_analysis, bb_analysis, ew_analysis, mtf_result)

    # Apply advanced score modifiers
    adv_mod = ichi_analysis["score"] + bb_analysis["score"] + ew_analysis["score"] + conv_analysis["score"]
    score = max(0, min(100, score + adv_mod))
    labels_r = [(80, "Eros vetel"), (60, "Gyenge vetel"), (40, "Semleges"),
                (20, "Gyenge eladas"), (0, "Eros eladas")]
    rec = next(lb for th, lb in labels_r if score >= th)

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

    pump_warn = penalty >= 20
    action_icon = G + "VETEL" if score >= 60 else (R + "ELADAS" if score < 40 else Y + "VARJ")
    if pump_warn:
        action_icon = R + "PUMP GYANU"
    score_str = f"{score}/100"
    parts = []
    if penalty > 0:
        parts.append(f"penalty: -{penalty}")
    if mtf_mod != 0:
        parts.append(f"MTF: {mtf_mod:+d}")
    if parts:
        score_str += f" (nyers: {raw_score}, {', '.join(parts)})"
    print(f" {action_icon}{D} | Score: {score_color}{score_str}{D} ({rec})")
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

    # ---- RESZLETES TECHNIKAI NARRATIVA ----
    print(f"\n{M} TECHNIKAI NARRATIVA{D}")
    print(f"{'-' * W}")
    narrative = _build_narrative(df, last, close, rsi, atr, chg, sr_levels,
                                 fib, mtf_result, large_candles, vol_24h)
    print(_wrap(narrative, W))

    # ---- SZINTEK (tomor) ----
    print(f"\n{M} SZINTEK{D}")
    print(f"{'-' * W}")
    if levels["nearest_support"]:
        print(f"  Tamasz:       {G}{_P(levels['nearest_support'])}{D}")
    if levels["nearest_resist"]:
        print(f"  Ellenallas:   {R}{_P(levels['nearest_resist'])}{D}")
    print(f"  Fibonacci:")
    for name, val in sorted(fib.items(), key=lambda x: x[1], reverse=True)[:5]:
        marker = " <<" if abs(close - val) / close < 0.02 else ""
        print(f"    {name:<12} {_P(val)}{Y}{marker}{D}")

    # ---- BELEPESI STRATEGIA ----
    print(f"\n{M} BELEPESI STRATEGIA{D}")
    print(f"{'-' * W}")
    sma20_v = float(last.get("sma_20", close))
    print(_build_detailed_entry(close, sr_levels, atr, rsi, fib, sma20_v))

    # ---- KILEPESI STRATEGIA ----
    print(f"\n{M} KILEPESI STRATEGIA{D}")
    print(f"{'-' * W}")
    print(_build_detailed_exit(close, sr_levels, atr, fib))

    # ---- SZCENARIÓ ELEMZES ----
    print(f"\n{M} SZCENARIÓ ELEMZES{D}")
    print(f"{'-' * W}")
    print(_build_detailed_scenarios(df, close, sr_levels, atr,
                                     last.get("adx", 0), rsi, mtf_result, fib))

    # ---- ICHIMOKU RESZLETES ----
    print(f"\n{M} ICHIMOKU FELHO ELEMZES ({ichi_analysis['summary']}){D}")
    print(f"{'-' * W}")
    for sig in ichi_analysis["signals"]:
        print(f"  {sig}")

    # ---- BOLLINGER RESZLETES ----
    print(f"\n{M} BOLLINGER BANDS ELEMZES ({bb_analysis['summary']}){D}")
    print(f"{'-' * W}")
    for sig in bb_analysis["signals"]:
        print(f"  {sig}")

    # ---- ELLIOTT WAVE ----
    print(f"\n{M} ELLIOTT WAVE ({ew_analysis['summary']}){D}")
    print(f"{'-' * W}")
    if ew_analysis["waves"]:
        for w in ew_analysis["waves"]:
            print(f"  Wave {w['num']} ({w['type']}): {_P(w['price'])} [{w['date']}]")
    else:
        print(f"  {ew_analysis['summary']}")

    # ---- KONVERGENCIA ----
    conv_sig = conv_analysis["signal"]
    conv_color = G if "BULL" in conv_sig else (R if "BEAR" in conv_sig else Y)
    print(f"\n{M} KONVERGENCIA ALERT: {conv_color}{conv_sig}{D}")
    print(f"{'-' * W}")
    for d in conv_analysis["details"]:
        print(f"  {d}")
    print(f"  -> {conv_analysis['bull']}/4 bullish | {conv_analysis['bear']}/4 bearish")
    if conv_analysis["total"] >= 3:
        print(f"  {conv_color}=> NAGYON EROS JELZES — {conv_analysis['total']}/4 rendszer egyezik!{D}")

    # ---- WHALE / SMART MONEY ----
    print(f"\n{M} WHALE / SMART MONEY{D}")
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
    if binance_extra and binance_extra.get("trades_24h"):
        print(f"  24h tranzakciok: {binance_extra['trades_24h']:,}")

    # ---- KOCKAZAT ----
    print(f"\n{M} KOCKAZAT ERTEKELES{D}")
    print(f"{'-' * W}")
    print(_build_detailed_risk(df, close, atr, vol_24h, rsi, mtf_result,
                                penalty, pump_flags))
    print(f"\n  Poziciomeret ($10,000 portfolio):")
    print(f"    Max kockazat:  {pos['max_risk_pct']:.1f}% (${pos['risk_usd']:.0f})")
    print(f"    Javasolt:      ${pos['position_usd']:,.0f} ({pos['position_pct']:.0f}%)")
    if atr > 0:
        days_to_target = abs(levels["reward1"]) / atr
        print(f"  Becsult ido celarig: ~{days_to_target:.0f} nap")

    # ---- PUMP SZURO ----
    if pump_flags:
        print(f"\n{R} PUMP & DUMP SZURO (penalty: -{penalty}){D}")
        print(f"{'-' * W}")
        for f in pump_flags:
            print(f"  {R}>>{D} {f}")
        if pump_warn:
            print(f"  {R}FIGYELEM: Magas pump kockazat!{D}")

    # ---- MTF ----
    if mtf_result:
        print_mtf_table(symbol, mtf_result)

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
        "score": score, "raw_score": raw_score, "penalty": penalty,
        "rec": rec, "alerts": len(alerts),
        "pump_warn": pump_warn,
    }


# ============================================================================
# 11. MULTI-COIN SCANNER
# ============================================================================
def run_scanner(symbols: list, days: int, source: str, provider: str,
                interval: str, quote: str,
                detail_threshold: float = 0, use_mtf: bool = False) -> None:
    """detail_threshold: csak ez feletti score-nal ad reszletes elemzest + chartot."""
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
            # MTF
            mtf = None
            if use_mtf and source == "binance":
                try:
                    mtf = calc_mtf(sym, quote)
                except Exception:
                    pass
            # Ellenorizzuk a score-t elore
            score, rec_label, raw_s, pen, pflags = calc_swing_score(df, sr)
            if mtf:
                score = max(0, min(100, score + mtf_score_modifier(mtf)))
            if detail_threshold > 0 and score < detail_threshold:
                last = df.iloc[-1]
                chg = ((last["close"] / df.iloc[-2]["close"]) - 1) * 100
                mtf_str = ""
                if mtf:
                    mtf_str = f"{mtf['bull_count']}/4"
                results.append({
                    "symbol": sym, "close": last["close"], "change_pct": chg,
                    "rsi": last.get("rsi", 50), "adx": last.get("adx", 0),
                    "macd": last.get("macd", 0), "score": score,
                    "raw_score": raw_s, "penalty": pen,
                    "rec": rec_label, "alerts": len(generate_alerts(df, sr)),
                    "pump_warn": pen >= 20, "mtf": mtf_str,
                })
                continue
            info = print_summary(df, sym, sr, binance_extra, mtf_result=mtf)
            if mtf:
                info["mtf"] = f"{mtf['bull_count']}/4"
            plot_chart(df, sym, sr)
            results.append(info)
        except Exception as e:
            print(f"  HIBA ({sym}): {e}")

    if len(results) > 1:
        results.sort(key=lambda x: x["score"], reverse=True)
        W2 = 105
        print("\n\n" + "=" * W2)
        print("  MULTI-COIN SCANNER OSSZEFOGLALO (rendezve swing score szerint)")
        print("=" * W2)
        hdr = f"  {'Coin':<14}{'Ar':>14}{'Valt%':>8}{'RSI':>7}{'ADX':>7}{'MACD':>12}{'Score':>10}{'Pen':>5}  {'Jelzes':<14}{'Flag':>6}"
        print(hdr)
        print("-" * W2)
        for r in results:
            pen = r.get("penalty", 0)
            pw = r.get("pump_warn", False)
            flag_str = "PUMP!" if pw else ""
            pen_str = f"-{pen:.0f}" if pen > 0 else ""
            print(
                f"  {r['symbol']:<14}"
                f"${r['close']:>12,.4g}"
                f"{r['change_pct']:>+7.2f}%"
                f"{r['rsi']:>7.1f}"
                f"{r['adx']:>7.1f}"
                f"{r['macd']:>+12.4g}"
                f"{r['score']:>7.1f}"
                f"{pen_str:>5}"
                f"  {r['rec']:<14}"
                f"{flag_str:>5}"
            )
        print("=" * W2)


def run_binance_scan(days: int, interval: str, quote: str,
                     min_volume: float, detail_threshold: float = 0,
                     use_mtf: bool = False, scan_all: bool = False) -> None:
    """Binance par scan es elemzes. scan_all=True: osszes par, nem csak top 50."""
    limit = 0 if scan_all else 50
    label = "OSSZES" if scan_all else "top 50"
    print(f"\n  Binance Scanner: {label} {quote} par lekerese (min vol: ${min_volume:,.0f})...")
    top_symbols = scan_binance_top_pairs(quote, min_volume, limit=limit)
    print(f"  Talalt parok: {len(top_symbols)}\n")
    if not top_symbols:
        print("  Nincs elegendo par a szuresnek megfelelo.")
        return
    run_scanner(top_symbols, days, "binance", "yfinance", interval, quote,
                detail_threshold=detail_threshold, use_mtf=use_mtf)


# ============================================================================
# 13. SHORT SCANNER
# ============================================================================
STABLECOINS = {"USDC", "USDT", "DAI", "TUSD", "BUSD", "FDUSD", "USDP",
               "PYUSD", "GUSD", "FRAX", "LUSD", "SUSD", "EUSD", "USDJ",
               "BFUSD", "XUSD", "USD1", "RLUSD", "U", "AEUR", "EURT",
               "USDD", "CUSD", "USDY", "USDX", "ZUSD"}
FIAT_BASES = {"EUR", "GBP", "JPY", "TRY", "AUD", "BRL", "ARS", "PLN",
              "RON", "UAH", "NGN", "PAXG", "XAUT"}
_BLACKLIST_SUBSTRINGS = ("USD", "EUR", "GBP", "JPY")  # ha base tartalmazza

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

    raw_score = min(score, 100)

    # Pump penalty a short score-ra is (extrem volatilitas nem megbizhato)
    penalty, pflags = calc_pump_penalty(df)
    # Short-nal felezzuk a penalty-t (a volatilitas reszben jo shorthoz)
    short_penalty = penalty * 0.5
    final = max(0, min(100, raw_score - short_penalty))

    return final, reasons, raw_score, short_penalty, pflags


LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR", "3L", "3S", "2L", "2S")


def _get_valid_usdt_symbols() -> set[str]:
    """ExchangeInfo-ból TRADING statuszu USDT parok, kiszurve a leveraged tokeneket."""
    try:
        resp = requests.get(f"{BINANCE_BASE_URL}/exchangeInfo", timeout=15)
        resp.raise_for_status()
        symbols_info = resp.json().get("symbols", [])
        valid = set()
        for s in symbols_info:
            if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING":
                sym = s["symbol"]
                base = sym[:-4]
                # Leveraged tokenek kiszurese
                if any(base.endswith(suf) for suf in LEVERAGED_SUFFIXES):
                    continue
                valid.add(sym)
        return valid
    except Exception:
        return set()


def _prefilter_short_candidates(tickers: list, min_volume: float,
                                exclude_stablecoins: bool,
                                valid_symbols: set[str] | None = None) -> list[dict]:
    """Eloszures: exchangeInfo, volume, stablecoin, leveraged token filter."""
    candidates = []
    for t in tickers:
        sym = t["symbol"]
        if not sym.endswith("USDT"):
            continue
        # ExchangeInfo filter (ha elerheto)
        if valid_symbols and sym not in valid_symbols:
            continue
        base = sym[:-4]
        if exclude_stablecoins and base in STABLECOINS:
            continue
        # Fiat/forex filter
        if base in FIAT_BASES:
            continue
        # Stablecoin substring filter (USD, EUR stb. a nev belsejeben)
        if exclude_stablecoins and any(sub in base for sub in _BLACKLIST_SUBSTRINGS):
            continue
        # Leveraged token filter (dupla biztonsag)
        if any(base.endswith(suf) for suf in LEVERAGED_SUFFIXES):
            continue
        qv = float(t.get("quoteVolume", 0))
        if qv < min_volume:
            continue
        candidates.append({
            "symbol": sym, "base": base,
            "price": float(t.get("lastPrice", 0)),
            "change_pct": float(t.get("priceChangePercent", 0)),
            "quote_volume": qv,
            "volume_24h": float(t.get("volume", 0)),
            "high_24h": float(t.get("highPrice", 0)),
            "low_24h": float(t.get("lowPrice", 0)),
        })
    return candidates


def run_short_scanner(days: int, interval: str, quote: str,
                      min_volume: float, min_score: float,
                      exclude_stablecoins: bool,
                      use_mtf: bool = False) -> None:
    """Short opportunity scanner - globalis Binance API, batch lekerdezesekkel."""
    global _short_scan_cache, _short_scan_cache_time
    import time

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

    # 1. ExchangeInfo - valid szimbolumok
    print(f"\n  Exchange info lekerese...", end="", flush=True)
    valid_symbols = _get_valid_usdt_symbols()
    print(f" {len(valid_symbols)} aktiv USDT par")

    # 2. Osszes ticker lekerese (cached)
    print(f"  24h tickers lekerese...", end="", flush=True)
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

    # 3. Pre-filter
    candidates = _prefilter_short_candidates(tickers, min_volume,
                                              exclude_stablecoins,
                                              valid_symbols)
    print(f"  Szurt jeloltek (vol > ${min_volume:,.0f}, USDT, no leverage): "
          f"{len(candidates)}")

    if not candidates:
        print(f"  {Y}Nincs jelolt a filternek megfelelo.{D}")
        return

    # 4. Mindegyikre technikai elemzes
    results = []
    total = len(candidates)
    errors = 0
    req_count = 0
    last_req_time = time.time()

    for idx, cand in enumerate(candidates):
        sym = cand["symbol"]

        # Progress (minden 10 parnal)
        if (idx + 1) % 10 == 0 or idx == total - 1:
            pct = (idx + 1) / total * 100
            print(f"\r  Szkenneles... {idx + 1}/{total} par atvizsgalva ({pct:.0f}%)   ",
                  end="", flush=True)

        try:
            # Rate limit: max 10 req/sec
            elapsed = time.time() - last_req_time
            if elapsed < 0.1:
                time.sleep(0.1 - elapsed)

            df = fetch_binance_data(sym, days=days, interval=interval,
                                   quote=quote, quiet=True)
            req_count += 1
            last_req_time = time.time()

            if len(df) < 20:
                continue
            df = add_all_indicators(df)
            sr = get_sr_levels(df)

            # Funding rate (proba, nem szamit bele a rate limitbe)
            fr = fetch_binance_funding_rate(sym, quote)

            short_score, reasons, raw_ss, ss_pen, ss_pflags = calc_short_score(df, sr, fr)

            # MTF modifier
            mtf = None
            mtf_str = ""
            if use_mtf:
                try:
                    mtf = calc_mtf(sym, quote)
                    mtf_mod = mtf_score_modifier(mtf)
                    # Short: csak MTF SHORT CONFIRMED ad bonust
                    if "SHORT" in mtf.get("signal", ""):
                        short_score = min(100, short_score + 15)
                    elif "LONG" in mtf.get("signal", ""):
                        short_score = max(0, short_score - 20)
                    mtf_str = f"{mtf['bear_count']}/4"
                except Exception:
                    pass

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
                    "mtf": mtf_str,
                    "mtf_result": mtf,
                })

        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                print(f"\n  {Y}Rate limit! Varakozas 30 mp...{D}", flush=True)
                time.sleep(30)
            else:
                errors += 1
            continue
        except Exception:
            errors += 1
            continue

    print(f"\r  Szkenneles... {total}/{total} par atvizsgalva (100%) - KESZ!     ")
    if errors:
        print(f"  ({errors} par atugorva hiba miatt)")
    print(f"  Talalatok: {len(results)} par score >= {min_score}")

    if not results:
        print(f"\n  {Y}Nincs short jelolt score >= {min_score} felett.{D}")
        return

    # 5. Rangsolas
    results.sort(key=lambda x: x["short_score"], reverse=True)
    top20 = results[:20]

    # 5. Osszefoglalo tabla
    print(f"\n{R}{'=' * 95}")
    print(f" TOP {len(top20)} SHORT LEHETOSEG (score >= {min_score})")
    print(f"{'=' * 95}{D}")
    mtf_hdr = "{'MTF':>6}" if use_mtf else ""
    hdr = f"  {'#':<4}{'Coin':<12}{'Ar':>14}{'24h%':>8}{'RSI':>7}{'Score':>8}"
    if use_mtf:
        hdr += f"{'MTF':>7}"
    hdr += f"  {'Fo indok'}"
    print(hdr)
    print(f"{'-' * 95}")
    for i, r in enumerate(top20):
        main_reason = r["reasons"][0] if r["reasons"] else "-"
        score_color = R if r["short_score"] >= 70 else (Y if r["short_score"] >= 50 else D)
        line = (
            f"  {i + 1:<4}"
            f"{r['symbol']:<12}"
            f"${r['price']:>12,.4g}"
            f"{r['change_pct']:>+7.2f}%"
            f"{r['rsi']:>7.1f}"
            f"  {score_color}{r['short_score']:>5.0f}{D}"
        )
        if use_mtf:
            line += f"  {r.get('mtf', ''):>4}"
        line += f"  {main_reason}"
        print(line)
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
# 14. MULTI-TIMEFRAME (MTF) MOTOR
# ============================================================================
_mtf_cache = {}
_mtf_cache_time = {}

MTF_INTERVALS = [
    ("1h", 168),   # 1 het
    ("4h", 180),   # 30 nap
    ("1d", 200),   # 200 nap
    ("1w", 104),   # 2 ev
]


def _fetch_mtf_data(symbol: str, quote: str = "USDT") -> dict:
    """Minden timeframe-re lekeri az OHLCV adatot (cached 15 perc)."""
    import time as _time
    bn_symbol = _symbol_to_binance(symbol, quote)
    now = datetime.now()
    cache_key = bn_symbol

    if (cache_key in _mtf_cache and cache_key in _mtf_cache_time and
            (now - _mtf_cache_time[cache_key]).total_seconds() < 900):
        return _mtf_cache[cache_key]

    result = {}
    for interval, limit in MTF_INTERVALS:
        try:
            params = {"symbol": bn_symbol, "interval": interval, "limit": limit}
            resp = requests.get(f"{BINANCE_BASE_URL}/klines", params=params, timeout=15)
            resp.raise_for_status()
            rows = resp.json()
            if not rows:
                continue
            df = pd.DataFrame(rows, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_vol",
                "taker_buy_quote_vol", "ignore",
            ])
            df["date"] = pd.to_datetime(df["open_time"], unit="ms")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            df = df.set_index("date")[["open", "high", "low", "close", "volume"]]
            result[interval] = df
            _time.sleep(0.1)
        except Exception:
            continue

    _mtf_cache[cache_key] = result
    _mtf_cache_time[cache_key] = now
    return result


def _analyze_single_tf(df: pd.DataFrame) -> dict:
    """Egy timeframe indikatorai es iranyjelzese."""
    if len(df) < 20:
        return {"direction": "NEUTRAL", "rsi": 50, "macd_bull": False,
                "trend_up": False, "adx": 0, "vol_trend": "?"}

    close = df["close"]
    sma20 = close.rolling(20).mean()
    trend_up = bool(close.iloc[-1] > sma20.iloc[-1])

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    ag = gain.ewm(alpha=1/14, min_periods=14).mean()
    al = loss.ewm(alpha=1/14, min_periods=14).mean()
    rsi_s = 100 - (100 / (1 + ag / al))
    rsi = float(rsi_s.iloc[-1])

    # MACD
    ef = close.ewm(span=12, adjust=False).mean()
    es = close.ewm(span=26, adjust=False).mean()
    macd_line = ef - es
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_bull = bool(macd_line.iloc[-1] > signal_line.iloc[-1])

    # ADX
    h, l, c = df["high"], df["low"], df["close"]
    pdm = h.diff(); mdm = -l.diff()
    pdm = pdm.where((pdm > mdm) & (pdm > 0), 0.0)
    mdm = mdm.where((mdm > pdm) & (mdm > 0), 0.0)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, min_periods=14).mean()
    pdi = 100 * (pdm.ewm(alpha=1/14, min_periods=14).mean() / atr)
    mdi = 100 * (mdm.ewm(alpha=1/14, min_periods=14).mean() / atr)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = float(dx.ewm(alpha=1/14, min_periods=14).mean().iloc[-1])

    # Volume trend
    vol = df["volume"].fillna(0)
    vol_trend = "?"
    if len(vol) >= 10:
        recent = vol.iloc[-5:].mean()
        older = vol.iloc[-10:-5].mean()
        if older > 0:
            ratio = recent / older
            vol_trend = "UP" if ratio > 1.2 else ("DOWN" if ratio < 0.8 else "FLAT")

    # Direction
    if trend_up and (rsi > 50 or macd_bull):
        direction = "BULLISH"
    elif not trend_up and (rsi < 50 or not macd_bull):
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"

    return {
        "direction": direction, "rsi": rsi, "macd_bull": macd_bull,
        "trend_up": trend_up, "adx": adx, "vol_trend": vol_trend,
    }


def calc_mtf(symbol: str, quote: str = "USDT") -> dict:
    """Multi-timeframe elemzes: iranyok + alignment score."""
    tf_data = _fetch_mtf_data(symbol, quote)
    analyses = {}
    for interval, _ in MTF_INTERVALS:
        if interval in tf_data:
            analyses[interval] = _analyze_single_tf(tf_data[interval])
        else:
            analyses[interval] = {"direction": "NEUTRAL", "rsi": 50,
                                  "macd_bull": False, "trend_up": False,
                                  "adx": 0, "vol_trend": "?"}

    directions = [a["direction"] for a in analyses.values()]
    bull_count = directions.count("BULLISH")
    bear_count = directions.count("BEARISH")

    if bull_count >= 3:
        signal = "MTF LONG CONFIRMED"
    elif bear_count >= 3:
        signal = "MTF SHORT CONFIRMED"
    elif bull_count == 2 or bear_count == 2:
        signal = "MTF WEAK"
    else:
        signal = "MTF CONFLICT"

    return {
        "analyses": analyses,
        "bull_count": bull_count,
        "bear_count": bear_count,
        "signal": signal,
    }


def mtf_score_modifier(mtf_result: dict) -> int:
    """Score modosito az MTF eredmeny alapjan."""
    sig = mtf_result["signal"]
    if "CONFIRMED" in sig:
        return 15
    if sig == "MTF CONFLICT":
        return -20
    return 0


def print_mtf_table(symbol: str, mtf_result: dict) -> None:
    """MTF osszefoglalo tablazat kiirasa."""
    B = Fore.CYAN + Style.BRIGHT
    G = Fore.GREEN + Style.BRIGHT
    R = Fore.RED + Style.BRIGHT
    Y = Fore.YELLOW + Style.BRIGHT
    D = Style.RESET_ALL

    sig = mtf_result["signal"]
    sig_color = G if "LONG" in sig else (R if "SHORT" in sig else (Y if "WEAK" in sig else R))

    print(f"\n{B} MULTI-TIMEFRAME ELEMZES: {symbol}{D}")
    print(f"  {'TF':<6}{'Trend':<10}{'RSI':<8}{'MACD':<10}{'ADX':<7}{'Vol':<7}{'Jelzes'}")
    print(f"  {'-' * 58}")

    for interval, _ in MTF_INTERVALS:
        a = mtf_result["analyses"].get(interval, {})
        trend_icon = G + "Bull" + D if a.get("trend_up") else R + "Bear" + D
        rsi_v = a.get("rsi", 0)
        macd_icon = G + "Bull" + D if a.get("macd_bull") else R + "Bear" + D
        adx_v = a.get("adx", 0)
        vol_t = a.get("vol_trend", "?")

        d = a.get("direction", "NEUTRAL")
        d_color = G if d == "BULLISH" else (R if d == "BEARISH" else Y)
        print(
            f"  {interval:<6}"
            f"{trend_icon:<19}"
            f"{rsi_v:<8.1f}"
            f"{macd_icon:<19}"
            f"{adx_v:<7.0f}"
            f"{vol_t:<7}"
            f"{d_color}{d}{D}"
        )

    bc = mtf_result["bull_count"]
    brc = mtf_result["bear_count"]
    print(f"  {'-' * 58}")
    print(f"  Alignment: {G}{bc}/4 BULL{D} | {R}{brc}/4 BEAR{D} | "
          f"Jelzes: {sig_color}{sig}{D}")


# ============================================================================
# 15. BACKTESTING MOTOR
# ============================================================================
def _rolling_score(df: pd.DataFrame, idx: int, window: int,
                   side: str) -> tuple:
    """Score szamitas egy adott pontra az adatban, a pump penaltyvel egyutt."""
    start = max(0, idx - window + 1)
    sub = df.iloc[start:idx + 1].copy()
    if len(sub) < 20:
        return 0, 0, 0
    sub["sma_20"] = sub["close"].rolling(20).mean()
    sub["sma_50"] = sub["close"].rolling(50).mean()
    sub["sma_200"] = sub["close"].rolling(200).mean()
    delta = sub["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    ag = gain.ewm(alpha=1/14, min_periods=14).mean()
    al = loss.ewm(alpha=1/14, min_periods=14).mean()
    sub["rsi"] = 100 - (100 / (1 + ag / al))
    ef = sub["close"].ewm(span=12, adjust=False).mean()
    es = sub["close"].ewm(span=26, adjust=False).mean()
    sub["macd"] = ef - es
    sub["macd_signal"] = sub["macd"].ewm(span=9, adjust=False).mean()
    sub["macd_hist"] = sub["macd"] - sub["macd_signal"]
    bb_mid = sub["close"].rolling(20).mean()
    bb_std = sub["close"].rolling(20).std()
    sub["bb_upper"] = bb_mid + 2 * bb_std
    sub["bb_lower"] = bb_mid - 2 * bb_std
    sub["bb_middle"] = bb_mid
    h, l, c = sub["high"], sub["low"], sub["close"]
    pdm = h.diff(); mdm = -l.diff()
    pdm = pdm.where((pdm > mdm) & (pdm > 0), 0.0)
    mdm = mdm.where((mdm > pdm) & (mdm > 0), 0.0)
    import pandas as _pd
    tr = _pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, min_periods=14).mean()
    sub["plus_di"] = 100*(pdm.ewm(alpha=1/14,min_periods=14).mean()/atr)
    sub["minus_di"] = 100*(mdm.ewm(alpha=1/14,min_periods=14).mean()/atr)
    dx = 100*(sub["plus_di"]-sub["minus_di"]).abs()/(sub["plus_di"]+sub["minus_di"]).replace(0,np.nan)
    sub["adx"] = dx.ewm(alpha=1/14, min_periods=14).mean()
    hi, lo = sub["high"], sub["low"]
    sub["senkou_a"] = ((hi.rolling(9).max()+lo.rolling(9).min())/2 + (hi.rolling(26).max()+lo.rolling(26).min())/2)/2
    sub["senkou_b"] = (hi.rolling(52).max()+lo.rolling(52).min())/2

    sr = get_sr_levels(sub)

    if side in ("long", "both"):
        score, _, raw, pen, _ = calc_swing_score(sub, sr)
        return score, raw, pen
    else:
        from scipy.signal import argrelextrema as _are2
        sc, _, raw, pen, _ = calc_short_score(sub, sr)
        return sc, raw, pen


def run_backtest(symbol: str, days: int, source: str, provider: str,
                 interval: str, quote: str, threshold: float,
                 sl_pct: float, tp_pct: float, capital: float,
                 risk_pct: float, side: str) -> dict:
    """Backtest motor egyetlen coinra."""
    B = Fore.CYAN + Style.BRIGHT
    R = Fore.RED + Style.BRIGHT
    G = Fore.GREEN + Style.BRIGHT
    Y = Fore.YELLOW + Style.BRIGHT
    D = Style.RESET_ALL

    # Extra napok az indikatorokhoz
    fetch_days = days + 60
    df = fetch_crypto_data(symbol, fetch_days, source, provider, interval, quote)
    if len(df) < days:
        print(f"  {Y}Csak {len(df)} nap adat erheto el.{D}")

    trades = []
    equity = [capital]
    current_capital = capital
    position = None  # {"entry_price", "entry_idx", "entry_date", "size_usd", "direction"}
    window = min(90, len(df) - 1)

    test_start = max(60, len(df) - days)

    print(f"  Backtest: {symbol} | {side} | {len(df)-test_start} nap | "
          f"threshold={threshold} SL={sl_pct}% TP={tp_pct}%")

    for i in range(test_start, len(df)):
        close = df["close"].iloc[i]
        high = df["high"].iloc[i]
        low = df["low"].iloc[i]
        date = df.index[i]

        # Ha van nyitott pozicio: check SL/TP/timeout/score-drop
        if position is not None:
            entry = position["entry_price"]
            days_held = i - position["entry_idx"]
            direction = position["direction"]

            if direction == "long":
                pl_pct = (close - entry) / entry * 100
                hit_sl = low <= entry * (1 - sl_pct / 100)
                hit_tp = high >= entry * (1 + tp_pct / 100)
            else:
                pl_pct = (entry - close) / entry * 100
                hit_sl = high >= entry * (1 + sl_pct / 100)
                hit_tp = low <= entry * (1 - tp_pct / 100)

            exit_reason = None
            exit_price = close

            if hit_sl:
                exit_reason = "Stop-loss"
                exit_price = entry * (1 - sl_pct/100) if direction == "long" else entry * (1 + sl_pct/100)
                pl_pct = -sl_pct
            elif hit_tp:
                exit_reason = "Take-profit"
                exit_price = entry * (1 + tp_pct/100) if direction == "long" else entry * (1 - tp_pct/100)
                pl_pct = tp_pct
            elif days_held >= 14:
                exit_reason = "Timeout (14 nap)"
            else:
                # Score check
                sc, _, _ = _rolling_score(df, i, window, direction)
                if sc < 40:
                    exit_reason = f"Score drop ({sc:.0f})"

            if exit_reason:
                trade_pl_usd = position["size_usd"] * pl_pct / 100
                current_capital += trade_pl_usd
                trades.append({
                    "entry_date": position["entry_date"].strftime("%Y-%m-%d"),
                    "exit_date": date.strftime("%Y-%m-%d"),
                    "symbol": symbol,
                    "direction": direction,
                    "entry_price": entry,
                    "exit_price": exit_price,
                    "pl_pct": pl_pct,
                    "pl_usd": trade_pl_usd,
                    "days_held": days_held,
                    "exit_reason": exit_reason,
                    "score": position["score"],
                })
                position = None

            equity.append(current_capital)
            continue

        # Nincs pozicio: check signal
        test_side = side if side != "both" else "long"
        sc, raw, pen = _rolling_score(df, i, window, test_side)

        # Both: ha long score alacsony, probaljuk short-ot
        if side == "both" and sc < threshold:
            sc2, raw2, pen2 = _rolling_score(df, i, window, "short")
            if sc2 >= threshold:
                sc, test_side = sc2, "short"

        if sc >= threshold:
            size_usd = current_capital * risk_pct / 100 / (sl_pct / 100)
            size_usd = min(size_usd, current_capital * 0.3)
            if size_usd > 10:  # min $10 pozicio
                position = {
                    "entry_price": close,
                    "entry_idx": i,
                    "entry_date": date,
                    "size_usd": size_usd,
                    "direction": test_side,
                    "score": sc,
                }

        equity.append(current_capital)

    # Nyitott pozicio zarasa az utolso napon
    if position is not None:
        close = df["close"].iloc[-1]
        entry = position["entry_price"]
        direction = position["direction"]
        pl_pct = ((close - entry)/entry*100) if direction == "long" else ((entry - close)/entry*100)
        trade_pl_usd = position["size_usd"] * pl_pct / 100
        current_capital += trade_pl_usd
        trades.append({
            "entry_date": position["entry_date"].strftime("%Y-%m-%d"),
            "exit_date": df.index[-1].strftime("%Y-%m-%d"),
            "symbol": symbol, "direction": direction,
            "entry_price": entry, "exit_price": close,
            "pl_pct": pl_pct, "pl_usd": trade_pl_usd,
            "days_held": len(df) - 1 - position["entry_idx"],
            "exit_reason": "Vegso zaras", "score": position["score"],
        })
        equity.append(current_capital)

    return {
        "symbol": symbol, "trades": trades, "equity": equity,
        "capital": capital, "final": current_capital,
        "side": side, "threshold": threshold,
        "sl_pct": sl_pct, "tp_pct": tp_pct,
        "df": df, "test_start": test_start,
    }


def _calc_backtest_metrics(result: dict) -> dict:
    trades = result["trades"]
    equity = result["equity"]
    capital = result["capital"]

    if not trades:
        return {"total": 0}

    wins = [t for t in trades if t["pl_pct"] > 0]
    losses = [t for t in trades if t["pl_pct"] <= 0]
    pls = [t["pl_pct"] for t in trades]
    usd_pls = [t["pl_usd"] for t in trades]

    gross_profit = sum(t["pl_usd"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pl_usd"] for t in losses)) if losses else 1

    # Max drawdown
    peak = capital
    max_dd = 0
    for eq in equity:
        if eq > peak:
            peak = eq
        dd = (eq - peak) / peak * 100
        if dd < max_dd:
            max_dd = dd

    # Sharpe (annualizalt, napi equity-bol)
    if len(equity) > 2:
        eq_arr = np.array(equity)
        returns = np.diff(eq_arr) / eq_arr[:-1]
        sharpe = (returns.mean() / returns.std() * np.sqrt(365)) if returns.std() > 0 else 0
    else:
        sharpe = 0

    return {
        "total": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(trades) * 100,
        "avg_profit": np.mean([t["pl_pct"] for t in wins]) if wins else 0,
        "avg_loss": np.mean([t["pl_pct"] for t in losses]) if losses else 0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "expectancy": np.mean(usd_pls),
        "best_trade": max(pls),
        "worst_trade": min(pls),
        "avg_days": np.mean([t["days_held"] for t in trades]),
        "final_equity": result["final"],
        "total_return": (result["final"] - capital) / capital * 100,
    }


def _print_backtest_results(result: dict, metrics: dict) -> None:
    B = Fore.CYAN + Style.BRIGHT
    R = Fore.RED + Style.BRIGHT
    G = Fore.GREEN + Style.BRIGHT
    Y = Fore.YELLOW + Style.BRIGHT
    D = Style.RESET_ALL

    sym = result["symbol"]
    trades = result["trades"]

    print(f"\n{B}{'=' * 75}")
    print(f" BACKTEST EREDMENYEK: {sym} ({result['side'].upper()})")
    print(f"{'=' * 75}{D}")

    if metrics["total"] == 0:
        print(f"  {Y}Nincs trade a megadott parameterekkel.{D}")
        return

    ret_color = G if metrics["total_return"] > 0 else R
    wr_color = G if metrics["win_rate"] > 50 else (Y if metrics["win_rate"] > 40 else R)
    pf_color = G if metrics["profit_factor"] > 1.5 else (Y if metrics["profit_factor"] > 1 else R)

    print(f"  Osszes trade:        {metrics['total']}")
    print(f"  Nyero / vesztes:     {G}{metrics['wins']}{D} / {R}{metrics['losses']}{D}")
    print(f"  Win rate:            {wr_color}{metrics['win_rate']:.1f}%{D}")
    print(f"  Atlag profit/trade:  {G}+{metrics['avg_profit']:.2f}%{D}")
    print(f"  Atlag veszteseg:     {R}{metrics['avg_loss']:.2f}%{D}")
    print(f"  Profit factor:       {pf_color}{metrics['profit_factor']:.2f}{D}")
    print(f"  Legjobb trade:       {G}+{metrics['best_trade']:.2f}%{D}")
    print(f"  Legrosszabb trade:   {R}{metrics['worst_trade']:.2f}%{D}")
    print(f"  Atlag tartas:        {metrics['avg_days']:.1f} nap")
    print(f"  Max drawdown:        {R}{metrics['max_drawdown']:.1f}%{D}")
    print(f"  Sharpe ratio:        {metrics['sharpe']:.2f}")
    print(f"  Expectancy:          ${metrics['expectancy']:.2f}/trade")
    print(f"{'-' * 75}")
    print(f"  Kezdo toke:          ${result['capital']:,.2f}")
    print(f"  Vegso egyenleg:      {ret_color}${metrics['final_equity']:,.2f}{D}")
    print(f"  Osszesitett hozam:   {ret_color}{metrics['total_return']:+.2f}%{D}")
    print(f"{B}{'=' * 75}{D}")

    # Trade log
    print(f"\n{B} TRADE LOG{D}")
    print(f"  {'Datum':<12}{'Exit':<12}{'Dir':<6}{'Entry':>10}{'Exit$':>10}"
          f"{'P/L%':>8}{'P/L$':>10}{'Nap':>5} {'Ok'}")
    print(f"  {'-' * 80}")
    for t in trades:
        c = G if t["pl_pct"] > 0 else R
        print(
            f"  {t['entry_date']:<12}{t['exit_date']:<12}"
            f"{t['direction']:<6}"
            f"${t['entry_price']:>9,.4g}"
            f"${t['exit_price']:>9,.4g}"
            f"  {c}{t['pl_pct']:>+6.2f}%{D}"
            f"  {c}${t['pl_usd']:>+8.2f}{D}"
            f"  {t['days_held']:>3}  {t['exit_reason']}"
        )


def _print_score_calibration(trades: list) -> None:
    B = Fore.CYAN + Style.BRIGHT
    G = Fore.GREEN + Style.BRIGHT
    R = Fore.RED + Style.BRIGHT
    Y = Fore.YELLOW + Style.BRIGHT
    D = Style.RESET_ALL

    if not trades:
        return

    print(f"\n{B} SCORE KALIBRALAS{D}")
    print(f"  {'Score tartomany':<20}{'Trades':>8}{'Win%':>8}{'Avg P/L':>10}")
    print(f"  {'-' * 50}")
    for lo, hi in [(60, 70), (70, 80), (80, 90), (90, 100)]:
        bucket = [t for t in trades if lo <= t["score"] < hi]
        if not bucket:
            print(f"  {lo}-{hi:<18}{'0':>8}{'–':>8}{'–':>10}")
            continue
        wins = sum(1 for t in bucket if t["pl_pct"] > 0)
        wr = wins / len(bucket) * 100
        avg = np.mean([t["pl_pct"] for t in bucket])
        c = G if wr > 50 else (Y if wr > 40 else R)
        print(f"  {lo}-{hi:<18}{len(bucket):>8}{c}{wr:>7.1f}%{D}{avg:>+9.2f}%")


def _plot_equity_curve(result: dict, metrics: dict) -> None:
    equity = result["equity"]
    df = result["df"]
    test_start = result["test_start"]
    symbol = result["symbol"]
    capital = result["capital"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    height_ratios=[3, 1], sharex=True)
    fig.suptitle(f"{symbol} Backtest — Equity Curve ({result['side'].upper()})",
                 fontsize=14, fontweight="bold")

    dates = df.index[test_start:test_start + len(equity)]
    if len(dates) < len(equity):
        dates = df.index[-len(equity):]

    # Equity curve
    ax1.plot(dates[:len(equity)], equity, lw=1.5, color="#1f77b4", label="Equity")
    ax1.axhline(capital, lw=0.7, ls="--", color="gray", alpha=0.5, label=f"Kezdo (${capital:,.0f})")

    # Buy & hold osszehasonlitas
    bh_start = df["close"].iloc[test_start]
    bh_eq = [capital * df["close"].iloc[test_start + j] / bh_start
             for j in range(min(len(equity), len(df) - test_start))]
    ax1.plot(dates[:len(bh_eq)], bh_eq, lw=1, ls="--", color="#ff7f0e",
             alpha=0.7, label="Buy & Hold")

    ax1.set_ylabel("Egyenleg (USD)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Drawdown
    eq_arr = np.array(equity)
    peak = np.maximum.accumulate(eq_arr)
    dd_pct = (eq_arr - peak) / peak * 100
    ax2.fill_between(dates[:len(dd_pct)], dd_pct, 0, color="red", alpha=0.3)
    ax2.plot(dates[:len(dd_pct)], dd_pct, lw=0.8, color="red")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_ylim(min(dd_pct) * 1.2, 5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"results/backtest_{symbol}_{datetime.now().strftime('%Y%m%d')}.png"
    import os
    os.makedirs("results", exist_ok=True)
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"\n  Equity curve mentve: {fname}")
    plt.close(fig)


def _walk_forward(result: dict, capital: float, sl_pct: float,
                  tp_pct: float, risk_pct: float, side: str) -> None:
    """Walk-forward teszt: 70% train / 30% test split."""
    B = Fore.CYAN + Style.BRIGHT
    G = Fore.GREEN + Style.BRIGHT
    R = Fore.RED + Style.BRIGHT
    Y = Fore.YELLOW + Style.BRIGHT
    D = Style.RESET_ALL

    trades = result["trades"]
    if len(trades) < 6:
        print(f"\n  {Y}Walk-forward: tul keves trade ({len(trades)}).{D}")
        return

    split = int(len(trades) * 0.7)
    train = trades[:split]
    test = trades[split:]

    train_wins = sum(1 for t in train if t["pl_pct"] > 0)
    test_wins = sum(1 for t in test if t["pl_pct"] > 0)
    train_wr = train_wins / len(train) * 100 if train else 0
    test_wr = test_wins / len(test) * 100 if test else 0

    train_pl = np.mean([t["pl_pct"] for t in train])
    test_pl = np.mean([t["pl_pct"] for t in test])

    overfit = abs(train_wr - test_wr) > 15

    print(f"\n{B} WALK-FORWARD TESZT (70/30 split){D}")
    print(f"  {'':20}{'Train':>10}{'Test':>10}")
    print(f"  {'-' * 42}")
    print(f"  {'Trades':<20}{len(train):>10}{len(test):>10}")
    tc = G if train_wr > 50 else R
    vc = G if test_wr > 50 else R
    print(f"  {'Win rate':<20}{tc}{train_wr:>9.1f}%{D}{vc}{test_wr:>9.1f}%{D}")
    print(f"  {'Avg P/L':<20}{train_pl:>+9.2f}%{test_pl:>+9.2f}%")
    if overfit:
        print(f"\n  {R}FIGYELEM: Train/test win rate elteres > 15% -> overfitting gyanu!{D}")
    else:
        print(f"\n  {G}Train/test konzisztens — nincs overfitting jel.{D}")


def run_backtest_suite(symbols: list, days: int, source: str, provider: str,
                       interval: str, quote: str, threshold: float,
                       sl_pct: float, tp_pct: float, capital: float,
                       risk_pct: float, side: str) -> None:
    """Backtest futtatasa egy vagy tobb coinra."""
    B = Fore.CYAN + Style.BRIGHT
    D = Style.RESET_ALL

    print(f"\n{B}{'=' * 75}")
    print(f" BACKTESTING ENGINE")
    print(f"{'=' * 75}{D}")
    print(f"  Coinok: {', '.join(symbols)}")
    print(f"  Idoszak: {days} nap | Side: {side} | Threshold: {threshold}")
    print(f"  SL: {sl_pct}% | TP: {tp_pct}% | Capital: ${capital:,.0f} | Risk: {risk_pct}%")
    print()

    all_results = []

    for sym in symbols:
        try:
            result = run_backtest(sym, days, source, provider, interval,
                                  quote, threshold, sl_pct, tp_pct,
                                  capital, risk_pct, side)
            metrics = _calc_backtest_metrics(result)
            _print_backtest_results(result, metrics)
            if metrics["total"] > 0:
                _print_score_calibration(result["trades"])
                _plot_equity_curve(result, metrics)
                _walk_forward(result, capital, sl_pct, tp_pct, risk_pct, side)
            all_results.append((sym, result, metrics))
        except Exception as e:
            print(f"  HIBA ({sym}): {e}")

    # Multi-coin osszefoglalo
    if len(all_results) > 1:
        print(f"\n{B}{'=' * 85}")
        print(f" MULTI-COIN BACKTEST OSSZEFOGLALO")
        print(f"{'=' * 85}{D}")
        print(f"  {'Coin':<14}{'Trades':>7}{'Win%':>7}{'P/L%':>8}{'PF':>6}"
              f"{'MaxDD':>8}{'Sharpe':>8}{'Vegso$':>12}")
        print(f"  {'-' * 72}")
        for sym, res, m in sorted(all_results, key=lambda x: x[2].get("total_return", 0), reverse=True):
            if m["total"] == 0:
                print(f"  {sym:<14}{'0':>7}{'–':>7}{'–':>8}{'–':>6}{'–':>8}{'–':>8}{'–':>12}")
                continue
            rc = Fore.GREEN if m["total_return"] > 0 else Fore.RED
            print(
                f"  {sym:<14}"
                f"{m['total']:>7}"
                f"{m['win_rate']:>6.1f}%"
                f"{rc}{m['total_return']:>+7.1f}%{D}"
                f"{m['profit_factor']:>6.2f}"
                f"{m['max_drawdown']:>7.1f}%"
                f"{m['sharpe']:>8.2f}"
                f"  {rc}${m['final_equity']:>9,.2f}{D}"
            )
        print(f"{'=' * 85}")


# ============================================================================
# 16. FULL SCAN (--scan-all)
# ============================================================================
def run_full_scan(days: int, interval: str, quote: str,
                  min_volume: float, min_short_score: float,
                  use_mtf: bool) -> None:
    """Teljes scan: long + short, osszes par, top 3+3 reszletes elemzes."""
    import os, time as _time
    B = Fore.CYAN + Style.BRIGHT
    G = Fore.GREEN + Style.BRIGHT
    R = Fore.RED + Style.BRIGHT
    Y = Fore.YELLOW + Style.BRIGHT
    D = Style.RESET_ALL

    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    outfile = f"results/full_scan_{ts}.txt"

    # Redirect print to both stdout and file
    import io, sys
    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()

    logf = open(outfile, "w")
    old_stdout = sys.stdout
    sys.stdout = Tee(old_stdout, logf)

    try:
        print(f"\n{B}{'=' * 75}")
        print(f" FULL CRYPTO SCAN — {ts}")
        print(f"{'=' * 75}{D}")
        print(f"  Min volume: ${min_volume:,.0f} | MTF: {'ON' if use_mtf else 'OFF'}")
        print(f"  Idoszak: {days} nap | Interval: {interval}\n")

        # ---- LONG SCAN ----
        print(f"{G}{'=' * 75}")
        print(f" [1/3] LONG SCANNER — OSSZES USDT par")
        print(f"{'=' * 75}{D}")

        all_symbols = scan_binance_top_pairs(quote, min_volume, limit=0)
        print(f"  Szurt parok: {len(all_symbols)}\n")

        long_results = []
        total = len(all_symbols)
        for idx, sym in enumerate(all_symbols):
            if (idx + 1) % 10 == 0 or idx == total - 1:
                pct = (idx + 1) / total * 100
                print(f"\r  Long scan... {idx+1}/{total} ({pct:.0f}%)   ", end="", flush=True)
            try:
                df = fetch_binance_data(sym, days=days, interval=interval,
                                       quote=quote, quiet=True)
                if len(df) < 20:
                    continue
                df = add_all_indicators(df)
                sr = get_sr_levels(df)
                score, rec, raw, pen, pflags = calc_swing_score(df, sr)
                mtf = None
                mtf_str = ""
                if use_mtf:
                    try:
                        mtf = calc_mtf(sym, quote)
                        score = max(0, min(100, score + mtf_score_modifier(mtf)))
                        mtf_str = f"{mtf['bull_count']}/4"
                    except Exception:
                        pass
                last = df.iloc[-1]
                chg = ((last["close"] / df.iloc[-2]["close"]) - 1) * 100
                long_results.append({
                    "symbol": sym, "close": last["close"], "change_pct": chg,
                    "rsi": last.get("rsi", 50), "adx": last.get("adx", 0),
                    "macd": last.get("macd", 0), "score": score,
                    "raw_score": raw, "penalty": pen,
                    "rec": rec, "pump_warn": pen >= 20,
                    "mtf": mtf_str, "mtf_result": mtf,
                    "df": df, "sr": sr,
                })
                _time.sleep(0.1)
            except Exception:
                continue

        print(f"\r  Long scan... {total}/{total} (100%) - KESZ!         ")

        long_results.sort(key=lambda x: x["score"], reverse=True)
        top_long = [r for r in long_results if r["score"] >= 60 and not r["pump_warn"]]

        # Long tabla
        W2 = 105
        print(f"\n{'=' * W2}")
        print(f"  LONG SCANNER OSSZEFOGLALO — TOP 20 (rendezve score szerint)")
        print(f"{'=' * W2}")
        hdr = f"  {'#':<4}{'Coin':<14}{'Ar':>14}{'Valt%':>8}{'RSI':>7}{'Score':>8}{'Pen':>5}"
        if use_mtf:
            hdr += f"{'MTF':>6}"
        hdr += f"  {'Jelzes':<14}{'Flag':>6}"
        print(hdr)
        print(f"{'-' * W2}")
        for i, r in enumerate(long_results[:20]):
            pen = r.get("penalty", 0)
            pen_str = f"-{pen:.0f}" if pen > 0 else ""
            flag = "PUMP!" if r.get("pump_warn") else ""
            line = (
                f"  {i+1:<4}{r['symbol']:<14}"
                f"${r['close']:>12,.4g}"
                f"{r['change_pct']:>+7.2f}%"
                f"{r['rsi']:>7.1f}"
                f"{r['score']:>7.1f}"
                f"{pen_str:>5}"
            )
            if use_mtf:
                line += f"  {r.get('mtf', ''):>4}"
            line += f"  {r['rec']:<14}{flag:>5}"
            print(line)
        print(f"{'=' * W2}")
        print(f"  Osszes par: {total} | Score >= 60 (nem pump): {len(top_long)}")

        # ---- SHORT SCAN ----
        print(f"\n{R}{'=' * 75}")
        print(f" [2/3] SHORT SCANNER — OSSZES USDT par")
        print(f"{'=' * 75}{D}")
        run_short_scanner(days, interval, quote, min_volume,
                          min_short_score, True, use_mtf=use_mtf)

        # ---- TOP 3+3 RESZLETES ----
        print(f"\n{B}{'=' * 75}")
        print(f" [3/3] RESZLETES ELEMZES — TOP 3 LONG + TOP 3 SHORT")
        print(f"{'=' * 75}{D}")

        # Top 3 long
        for i, r in enumerate(top_long[:3]):
            sym = r["symbol"]
            print(f"\n{G}--- LONG #{i+1}: {sym} (Score: {r['score']}) ---{D}")
            try:
                bx = None
                try:
                    bx = fetch_binance_ticker_24h(sym, quote)
                    bx["funding_rate"] = fetch_binance_funding_rate(sym, quote)
                except Exception:
                    pass
                print_summary(r["df"], sym, r["sr"], bx,
                              mtf_result=r.get("mtf_result"))
                plot_chart(r["df"], sym, r["sr"])
            except Exception as e:
                print(f"  HIBA: {e}")

        # Top 3 short — ujra lekerjuk a short scanner eredmenyeit
        # (a run_short_scanner nem adja vissza, szoval ujrafuttatjuk gyorsan a top 3-at)
        # Cached adatok vannak, szoval gyors lesz
        print(f"\n{R}--- TOP 3 SHORT reszletes elemzes ---{D}")
        # Keressuk meg a 3 legjobb short jeloltet a long_results-bol
        # ahol a short score magas
        short_candidates = []
        for r in long_results:
            if r["score"] < 50 and not r["pump_warn"]:
                # Alacsony long score = potencialis short
                short_candidates.append(r)
        # Vagy hasznaljuk az MTF bearish jelolteket
        mtf_short = [r for r in long_results
                     if r.get("mtf_result") and r["mtf_result"].get("bear_count", 0) >= 3
                     and not r["pump_warn"]]
        mtf_short.sort(key=lambda x: x.get("mtf_result", {}).get("bear_count", 0), reverse=True)

        short_top = mtf_short[:3] if mtf_short else short_candidates[:3]
        for i, r in enumerate(short_top):
            sym = r["symbol"]
            bear_c = r.get("mtf_result", {}).get("bear_count", 0) if r.get("mtf_result") else "?"
            print(f"\n{R}--- SHORT #{i+1}: {sym} (MTF bear: {bear_c}/4) ---{D}")
            try:
                bx = None
                try:
                    bx = fetch_binance_ticker_24h(sym, quote)
                except Exception:
                    pass
                print_summary(r["df"], sym, r["sr"], bx,
                              mtf_result=r.get("mtf_result"))
                plot_chart(r["df"], sym, r["sr"])
            except Exception as e:
                print(f"  HIBA: {e}")

        print(f"\n{B}{'=' * 75}")
        print(f" FULL SCAN VEGE — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'=' * 75}{D}")

    finally:
        sys.stdout = old_stdout
        logf.close()

    print(f"\n  Eredmeny mentve: {outfile}")
    print(f"  Meret: {os.path.getsize(outfile) / 1024:.0f} KB")


# ============================================================================
# 17. FOPROGRAM
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
    parser.add_argument("--scan-all", action="store_true",
                        help="TELJES scan: long + short, osszes par, top 3+3 reszletes")
    parser.add_argument("--min-volume", type=float, default=1_000_000,
                        help="Minimum 24h volume USD-ben")
    parser.add_argument("--min-short-score", type=float, default=60,
                        help="Minimum short score (0-100, alapert: 60)")
    parser.add_argument("--exclude-stablecoins", action="store_true", default=True,
                        help="Stablecoinok kiszurese (alapert: igen)")
    parser.add_argument("--detail-threshold", type=float, default=0,
                        help="Reszletes elemzes csak e score felett (0=mindig)")
    # Backtest
    parser.add_argument("--backtest", action="store_true",
                        help="Backtest mod: szimulacio historikus adaton")
    parser.add_argument("--backtest-days", type=int, default=365,
                        help="Backtest idoszak napokban (alapert: 365)")
    parser.add_argument("--backtest-threshold", type=float, default=65,
                        help="Minimum score a belepeshez (alapert: 65)")
    parser.add_argument("--backtest-sl", type=float, default=5,
                        help="Stop-loss %% (alapert: 5)")
    parser.add_argument("--backtest-tp", type=float, default=10,
                        help="Take-profit %% (alapert: 10)")
    parser.add_argument("--backtest-capital", type=float, default=1000,
                        help="Kezdo toke USD (alapert: 1000)")
    parser.add_argument("--backtest-risk", type=float, default=2,
                        help="Pozicionkenti kockazat %% (alapert: 2)")
    parser.add_argument("--backtest-side", default="long",
                        choices=["long", "short", "both"],
                        help="Backtest irany (alapert: long)")
    # MTF
    parser.add_argument("--mtf", action="store_true",
                        help="Multi-timeframe elemzes (1h, 4h, 1d, 1w)")
    parser.add_argument("--mtf-min", type=int, default=3,
                        help="Minimum egyezo timeframe szam (alapert: 3)")
    args = parser.parse_args()

    source = args.source
    if args.scan_binance or args.scan_shorts or args.scan_all:
        source = "binance"
    if args.backtest and source == "openbb":
        source = "binance"

    print(f"\nCrypto Swing Trading Analyzer")
    print(f"Idoszak: {args.days} nap | Forras: {source}"
          + (f" | Interval: {args.interval}" if source in ("binance", "alpha") else
             f" | Provider: {args.provider}"))

    if args.scan_all:
        run_full_scan(args.days, args.interval, args.quote,
                      args.min_volume, args.min_short_score, args.mtf)
        return

    if args.backtest:
        if args.symbols:
            coin_list = [s.strip() for s in args.symbols.split(",")]
        elif args.symbol:
            coin_list = [args.symbol]
        else:
            coin_list = ["BTCUSDT"]
        run_backtest_suite(
            coin_list, args.backtest_days, source, args.provider,
            args.interval, args.quote, args.backtest_threshold,
            args.backtest_sl, args.backtest_tp, args.backtest_capital,
            args.backtest_risk, args.backtest_side)
        return

    if args.scan_shorts:
        run_short_scanner(args.days, args.interval, args.quote,
                          args.min_volume, args.min_short_score,
                          args.exclude_stablecoins, use_mtf=args.mtf)
        return

    if args.scan_binance:
        run_binance_scan(args.days, args.interval, args.quote, args.min_volume,
                         detail_threshold=args.detail_threshold,
                         use_mtf=args.mtf)
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
                args.interval, args.quote,
                detail_threshold=args.detail_threshold,
                use_mtf=args.mtf)


if __name__ == "__main__":
    main()
