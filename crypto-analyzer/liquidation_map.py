"""
Liquidation Map modul
=====================
Leveraged poziciok becsult likvidacios szintjeit szamolja, es a futures
piaci metaadatokat (funding rate, open interest, long/short ratio) gyujti.

Adatforras prioritas:
  1. Binance Futures (fapi.binance.com)  — elsodleges
  2. OKX public API                       — fallback (PROXY mod)
  3. CoinGlass public                      — liquidation history / sweep log
A Binance fapi sok regioban geo-korlatozott (451); ilyenkor automatikusan
OKX-re vagy proxy adatra esik vissza, es a Data Regime PROXY-t jelez.

Bovitmenyek:
  - HTF (Higher Timeframe) levels toggle
  - Zone width (ATR-alapu klaszter sav)
  - Pivot count parameterrel beallithato S/R szintek
  - Kiemelt szintek (MAJOR RESISTANCE + LIQ IMBALANCE szint)
  - Sweep log eletciklus (PENDING/ACTIVE/COMPLETED/FAILED) + JSON perzisztencia
  - Next funding timer
  - Multi-timeframe liq map
  - Attractor distance (top-3 magnes suly+tavolsag)
  - Proxy mode figyelmeztetesi rendszer
"""

import json
import math
import os
from datetime import datetime, timezone

import pandas as pd
import requests

BINANCE_FAPI = "https://fapi.binance.com"
OKX_BASE = "https://www.okx.com/api/v5"
COINGLASS_BASE = "https://open-api.coinglass.com/public/v2"
_HEADERS = {"User-Agent": "Mozilla/5.0"}

LEVERAGE_TIERS = [10, 25, 50, 100]
LEV_WEIGHTS = {10: 0.40, 25: 0.30, 50: 0.20, 100: 0.10}

SWEEP_LOG_PATH = "results/sweep_log.json"

# Default zone width multiplier
DEFAULT_ZONE_WIDTH = 0.3  # ATR * 0.3
VALID_ZONE_WIDTHS = (0.1, 0.2, 0.3, 0.5)


# ---------------------------------------------------------------------------
# UTILITY
# ---------------------------------------------------------------------------
def _to_okx_swap(symbol: str) -> str:
    s = symbol.upper().replace("-", "")
    for q in ("USDT", "USDC", "USD"):
        if s.endswith(q):
            return f"{s[:-len(q)]}-{q}-SWAP"
    return f"{symbol}-USDT-SWAP"


def _okx_ccy(symbol: str) -> str:
    s = symbol.upper().replace("-", "")
    for q in ("USDT", "USDC", "USD"):
        if s.endswith(q):
            return s[:-len(q)]
    return symbol


def _get(url: str, params: dict) -> dict | None:
    for _ in range(3):
        try:
            r = requests.get(url, params=params, timeout=10, headers=_HEADERS)
            if r.status_code == 200:
                d = r.json()
                if d.get("code") == "0":
                    return d
            return None
        except Exception:
            import time as _t
            _t.sleep(1)
    return None


def _binance_symbol(symbol: str) -> str:
    return symbol.upper().replace("-", "")


def _binance_get(path: str, params: dict) -> object | None:
    try:
        r = requests.get(f"{BINANCE_FAPI}{path}", params=params,
                         timeout=8, headers=_HEADERS)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# DATA FETCHERS (OKX + Binance + CoinGlass)
# ---------------------------------------------------------------------------
def fetch_funding_rate(symbol: str) -> float | None:
    d = _get(f"{OKX_BASE}/public/funding-rate", {"instId": _to_okx_swap(symbol)})
    if d and d.get("data"):
        try:
            return float(d["data"][0]["fundingRate"]) * 100
        except (KeyError, ValueError, IndexError):
            return None
    return None


def fetch_open_interest(symbol: str) -> dict | None:
    d = _get(f"{OKX_BASE}/public/open-interest",
             {"instType": "SWAP", "instId": _to_okx_swap(symbol)})
    if d and d.get("data"):
        try:
            row = d["data"][0]
            return {"oi_coin": float(row.get("oi", 0)),
                    "oi_usd": float(row.get("oiUsd", 0))}
        except (ValueError, IndexError):
            return None
    return None


def fetch_long_short_ratio(symbol: str) -> float | None:
    d = _get(f"{OKX_BASE}/rubik/stat/contracts/long-short-account-ratio",
             {"ccy": _okx_ccy(symbol), "period": "1H"})
    if d and d.get("data"):
        try:
            return float(d["data"][0][1])
        except (ValueError, IndexError):
            return None
    return None


def fetch_binance_funding(symbol: str) -> float | None:
    d = _binance_get("/fapi/v1/premiumIndex", {"symbol": _binance_symbol(symbol)})
    if isinstance(d, dict):
        try:
            return float(d["lastFundingRate"]) * 100
        except (KeyError, ValueError, TypeError):
            return None
    return None


def fetch_binance_oi(symbol: str, price: float = 0) -> dict | None:
    d = _binance_get("/fapi/v1/openInterest", {"symbol": _binance_symbol(symbol)})
    if isinstance(d, dict):
        try:
            oi_coin = float(d["openInterest"])
            return {"oi_coin": oi_coin, "oi_usd": oi_coin * price if price else 0}
        except (KeyError, ValueError, TypeError):
            return None
    return None


def fetch_binance_ls_ratio(symbol: str) -> float | None:
    d = _binance_get("/futures/data/topLongShortAccountRatio",
                     {"symbol": _binance_symbol(symbol), "period": "1h", "limit": 1})
    if isinstance(d, list) and d:
        try:
            return float(d[-1]["longShortRatio"])
        except (KeyError, ValueError, TypeError, IndexError):
            return None
    return None


def fetch_binance_force_orders(symbol: str, limit: int = 50) -> list | None:
    d = _binance_get("/fapi/v1/allForceOrders",
                     {"symbol": _binance_symbol(symbol), "limit": limit})
    return d if isinstance(d, list) else None


def fetch_coinglass_liquidations(symbol: str) -> list | None:
    ccy = _okx_ccy(symbol)
    try:
        r = requests.get(f"{COINGLASS_BASE}/liquidation_history",
                         params={"symbol": ccy, "time_type": "h1"},
                         timeout=8, headers=_HEADERS)
        if r.status_code == 200:
            d = r.json()
            if isinstance(d, dict) and d.get("data"):
                return d["data"]
    except Exception:
        pass
    return None


def fetch_futures_meta(symbol: str, price: float = 0) -> dict:
    """Egyesitett futures metaadat: Binance elsodleges, OKX fallback."""
    funding = fetch_binance_funding(symbol)
    oi = fetch_binance_oi(symbol, price)
    ls_ratio = fetch_binance_ls_ratio(symbol)
    source = "BINANCE"

    if funding is None:
        funding = fetch_funding_rate(symbol)
        if funding is not None:
            source = "OKX"
    if oi is None:
        oi = fetch_open_interest(symbol)
        if oi is not None:
            source = "OKX"
    if ls_ratio is None:
        ls_ratio = fetch_long_short_ratio(symbol)
        if ls_ratio is not None:
            source = "OKX"

    return {"funding": funding, "oi": oi, "ls_ratio": ls_ratio, "source": source}


# ---------------------------------------------------------------------------
# 6. NEXT FUNDING TIMER
# ---------------------------------------------------------------------------
def next_funding_time() -> str:
    """Kovetkezo funding esemenyig hatralevo ido (HH:MM formatum).

    Funding idopontok: 0:00, 8:00, 16:00 UTC.
    """
    now = datetime.now(timezone.utc)
    h = now.hour
    if h < 8:
        nxt = now.replace(hour=8, minute=0, second=0, microsecond=0)
    elif h < 16:
        nxt = now.replace(hour=16, minute=0, second=0, microsecond=0)
    else:
        nxt = (now.replace(hour=0, minute=0, second=0, microsecond=0)
               + pd.Timedelta(days=1))
    delta = nxt - now
    mins = int(delta.total_seconds() // 60)
    secs = int(delta.total_seconds() % 60)
    return f"{mins}:{secs:02d}"


# ---------------------------------------------------------------------------
# REGIME + CORE CALCULATIONS
# ---------------------------------------------------------------------------
def detect_regime(df) -> str:
    if df is None or len(df) < 30:
        return "NORMAL"
    close = df["close"]
    tr = df["high"] - df["low"]
    atr = tr.rolling(14).mean()
    atr_now = float(atr.iloc[-1])
    atr_avg = float(atr.iloc[-30:].mean())
    chg30 = (close.iloc[-1] - close.iloc[-30]) / close.iloc[-30] * 100
    if atr_avg > 0 and atr_now > atr_avg * 1.3:
        return "HIGH_VOL"
    if chg30 < -10:
        return "BEAR"
    if chg30 > 10:
        return "BULL"
    return "NORMAL"


def calc_liquidation_levels(price: float, oi_usd: float = 0,
                            ls_ratio: float = 1.0) -> dict:
    longs, shorts = {}, {}
    long_notional, short_notional = {}, {}
    long_share = ls_ratio / (1 + ls_ratio) if ls_ratio > 0 else 0.5
    short_share = 1 - long_share
    total_long = oi_usd * long_share
    total_short = oi_usd * short_share

    for lev in LEVERAGE_TIERS:
        longs[lev] = price * (1 - 1 / lev * 0.9)
        shorts[lev] = price * (1 + 1 / lev * 0.9)
        long_notional[lev] = total_long * LEV_WEIGHTS[lev]
        short_notional[lev] = total_short * LEV_WEIGHTS[lev]

    return {"long_liqs": longs, "short_liqs": shorts,
            "long_notional": long_notional, "short_notional": short_notional,
            "total_long": total_long, "total_short": total_short}


# ---------------------------------------------------------------------------
# 3. PIVOT COUNT — konfiguralhato pivot szintek (S/R szintek)
# ---------------------------------------------------------------------------
def calc_pivot_levels(df, piv_count: int = 5) -> list:
    """A legutolso piv_count pivot magasabb- es merulesi szintet adja.

    Visszater a szintek listajat [(level, 'HIGH'|'LOW'), ...].
    """
    if df is None or len(df) < piv_count * 2 + 1:
        return []
    high = df["high"].values
    low = df["low"].values
    pivots = []
    for i in range(piv_count, len(high) - piv_count):
        if all(high[i] >= high[i - j] for j in range(1, piv_count + 1)) and \
           all(high[i] >= high[i + j] for j in range(1, min(piv_count + 1, len(high) - i))):
            pivots.append((float(high[i]), "HIGH"))
        if all(low[i] <= low[i - j] for j in range(1, piv_count + 1)) and \
           all(low[i] <= low[i + j] for j in range(1, min(piv_count + 1, len(low) - i))):
            pivots.append((float(low[i]), "LOW"))
    return pivots[-piv_count * 2:]


# ---------------------------------------------------------------------------
# 2. ZONE WIDTH — klaszter sav ATR×multiplier alapjan
# ---------------------------------------------------------------------------
def _in_zone(level: float, price: float, zone_half: float) -> bool:
    """level a price kornyezeti savjaba esik-e?"""
    return abs(level - price) <= zone_half


def cluster_density_with_zone(liqs: dict, price: float, zone_half: float) -> dict:
    """Klaszter suruseg a megadott zona-savon belul."""
    up_count = 0
    dn_count = 0
    up_w = 0.0
    dn_w = 0.0
    for lev in LEVERAGE_TIERS:
        sl = liqs["short_liqs"][lev]
        ll = liqs["long_liqs"][lev]
        if sl > price and abs(sl - price) <= zone_half:
            up_count += 1
            up_w += LEV_WEIGHTS[lev]
        if ll < price and abs(price - ll) <= zone_half:
            dn_count += 1
            dn_w += LEV_WEIGHTS[lev]
    return {"up_count": up_count, "dn_count": dn_count,
            "up_w": round(up_w, 2), "dn_w": round(dn_w, 2),
            "str": f"FEL: {up_count}L/{up_w:.1f}W | LE: {dn_count}L/{dn_w:.1f}W"}


# ---------------------------------------------------------------------------
# 4. KIEMELT SZINTEK (MAJOR RESISTANCE + LIQ IMBALANCE szint)
# ---------------------------------------------------------------------------
def find_highlighted_levels(liqs: dict, price: float, atr: float,
                            pivots: list | None = None) -> list:
    """Detektalja a kiemelt szinteket.

    a) MAJOR RESISTANCE: ahol 3+ klaszter halmozodik (leverage szintek + pivot).
    b) LIQ IMBALANCE: ahol a long/short aszimmetria a legnagyobb.
    """
    highlights = []

    # Osszegyujtjuk az osszes szintet zone-onkent
    zone = atr * 0.5 if atr > 0 else price * 0.02
    all_levels = []
    for lev in LEVERAGE_TIERS:
        all_levels.append(("SHORT_LIQ", liqs["short_liqs"][lev], lev))
        all_levels.append(("LONG_LIQ", liqs["long_liqs"][lev], lev))
    if pivots:
        for plvl, ptype in pivots:
            all_levels.append(("PIVOT", plvl, 0))

    # Klasztereles: szintek amik zone tavolsagon belul vannak egymashoz
    used = set()
    for i, (t1, l1, lev1) in enumerate(all_levels):
        if i in used:
            continue
        cluster = [(t1, l1, lev1)]
        for j in range(i + 1, len(all_levels)):
            if j in used:
                continue
            if abs(all_levels[j][1] - l1) <= zone:
                cluster.append(all_levels[j])
                used.add(j)
        if len(cluster) >= 3:
            avg_lvl = sum(x[1] for x in cluster) / len(cluster)
            highlights.append({
                "level": avg_lvl, "type": "MAJOR_RESISTANCE",
                "count": len(cluster), "dist_atr": abs(avg_lvl - price) / atr if atr else 0,
                "dir": "FEL" if avg_lvl > price else "LE",
            })

    # LIQ IMBALANCE szint: ahol a long/short notional arany a legnagyobb
    max_imb = 0
    imb_level = None
    for lev in LEVERAGE_TIERS:
        sl_n = liqs["short_notional"].get(lev, 0)
        ll_n = liqs["long_notional"].get(lev, 0)
        imb = abs(sl_n - ll_n)
        if imb > max_imb:
            max_imb = imb
            imb_level = liqs["short_liqs"][lev] if sl_n > ll_n else liqs["long_liqs"][lev]
            imb_dir = "FEL" if sl_n > ll_n else "LE"
    if imb_level and max_imb > 0:
        already = any(abs(h["level"] - imb_level) / price < 0.005 for h in highlights)
        if not already:
            highlights.append({
                "level": imb_level, "type": "LIQ_IMBALANCE",
                "count": 1, "dist_atr": abs(imb_level - price) / atr if atr else 0,
                "dir": imb_dir, "imb_usd": max_imb,
            })

    highlights.sort(key=lambda x: abs(x["level"] - price))
    return highlights


# ---------------------------------------------------------------------------
# 8. ATTRACTOR DISTANCE
# ---------------------------------------------------------------------------
def calc_attractors(liqs: dict, price: float, atr: float,
                    pivots: list | None = None) -> list:
    """Top-3 attractor score-al rendezve.

    Attractor erosseg = count_in_zone × leverage_weight × notional_$M.
    """
    if atr <= 0:
        return []
    attractors = []
    for lev in LEVERAGE_TIERS:
        w = LEV_WEIGHTS[lev]
        for side, lvl_dict, not_dict, label in [
            ("FEL", "short_liqs", "short_notional", "SHORT_LIQ"),
            ("LE", "long_liqs", "long_notional", "LONG_LIQ"),
        ]:
            lvl = liqs[lvl_dict][lev]
            notional = liqs[not_dict][lev]
            dist = abs(lvl - price) / atr
            strength = w * (notional / 1e6) * (1 / max(dist, 0.1))
            direction = "FEL" if lvl > price else "LE"
            size = "MAJOR" if strength > 50 else ("MEDIUM" if strength > 10 else "MINOR")
            attractors.append({
                "level": lvl, "lev": lev, "dist_atr": round(dist, 1),
                "dir": direction, "strength": round(strength, 1),
                "size": size, "notional_m": round(notional / 1e6, 1),
            })
    # Pivot szintek mint attractorok
    if pivots:
        for plvl, ptype in pivots:
            dist = abs(plvl - price) / atr
            direction = "FEL" if plvl > price else "LE"
            attractors.append({
                "level": plvl, "lev": 0, "dist_atr": round(dist, 1),
                "dir": direction, "strength": round(5 / max(dist, 0.1), 1),
                "size": "PIVOT", "notional_m": 0,
            })
    attractors.sort(key=lambda x: x["strength"], reverse=True)
    return attractors[:3]


# ---------------------------------------------------------------------------
# 5. SWEEP LOG ELETCIKLUS (PENDING/ACTIVE/COMPLETED/FAILED) + JSON
# ---------------------------------------------------------------------------
def _load_sweep_log() -> list:
    if os.path.exists(SWEEP_LOG_PATH):
        try:
            with open(SWEEP_LOG_PATH) as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_sweep_log(entries: list) -> None:
    os.makedirs(os.path.dirname(SWEEP_LOG_PATH) or ".", exist_ok=True)
    with open(SWEEP_LOG_PATH, "w") as f:
        json.dump(entries[-500:], f, indent=2)


def build_sweep_log(df, liqs: dict, price: float, symbol: str = "",
                    force_orders: list | None = None) -> dict:
    """Sweep log a teljes eletciklussal:
    PENDING -> ACTIVE -> COMPLETED / FAILED.

    Allapot logika:
      PENDING:   ar nem erte meg el a szintet (>1 ATR tavolsag)
      ACTIVE:    ar kozel a szinthez (<1% tavolsag) de meg nem torte at
      COMPLETED: ar atlette a szintet (sweep megtortent)
      FAILED:    ar kozel volt de visszafordult (az elozo 10 barban)
    """
    entries = []
    hi = lo = price
    touch_hi = touch_lo = price
    if df is not None and len(df) >= 2:
        window = df.iloc[-10:] if len(df) >= 10 else df
        hi = float(window["high"].max())
        lo = float(window["low"].min())
        if len(df) >= 3:
            recent3 = df.iloc[-3:]
            touch_hi = float(recent3["high"].max())
            touch_lo = float(recent3["low"].min())

    for lev in sorted(LEVERAGE_TIERS):
        lvl_s = liqs["short_liqs"][lev]
        if hi >= lvl_s:
            status = "COMPLETED"
        elif touch_hi >= lvl_s * 0.99:
            status = "ACTIVE" if price >= lvl_s * 0.985 else "FAILED"
        else:
            status = "PENDING"
        entries.append({"dir": "SHORT", "lev": lev, "level": lvl_s, "status": status})

    for lev in sorted(LEVERAGE_TIERS):
        lvl_l = liqs["long_liqs"][lev]
        if lo <= lvl_l:
            status = "COMPLETED"
        elif touch_lo <= lvl_l * 1.01:
            status = "ACTIVE" if price <= lvl_l * 1.015 else "FAILED"
        else:
            status = "PENDING"
        entries.append({"dir": "LONG", "lev": lev, "level": lvl_l, "status": status})

    pending_long = sum(1 for e in entries if e["dir"] == "LONG" and e["status"] == "PENDING")
    pending_short = sum(1 for e in entries if e["dir"] == "SHORT" and e["status"] == "PENDING")
    active = sum(1 for e in entries if e["status"] == "ACTIVE")
    completed = sum(1 for e in entries if e["status"] == "COMPLETED")
    failed = sum(1 for e in entries if e["status"] == "FAILED")

    # Valos force orderek
    real = []
    if force_orders:
        for fo in force_orders[-10:]:
            try:
                side = fo.get("side", "")
                ldir = "LONG" if side == "SELL" else "SHORT"
                real.append({"dir": ldir, "price": float(fo.get("price", 0)),
                             "qty": float(fo.get("origQty", 0))})
            except (ValueError, TypeError):
                continue

    # Perzisztencia (JSON)
    if symbol:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        log = _load_sweep_log()
        for e in entries:
            log.append({
                "timestamp": now, "symbol": symbol, "level": round(e["level"], 1),
                "direction": e["dir"], "lev": e["lev"],
                "count": 1, "status": e["status"],
            })
        # Csak az utolso 500 bejegyzes
        _save_sweep_log(log)

    return {"entries": entries, "pending_long": pending_long,
            "pending_short": pending_short, "active": active,
            "completed": completed, "failed": failed,
            "real_orders": real}


# ---------------------------------------------------------------------------
# 1. HTF LEVELS + 7. MULTI-TIMEFRAME LIQ MAP
# ---------------------------------------------------------------------------
def calc_htf_levels(price: float, oi_usd: float, ls_ratio: float,
                    htf_multiplier: float = 1.5) -> dict:
    """Magasabb timeframe (4H/1D) likvidacios szintek.

    HTF szintek szelessebbek (tobb pozicio gyul fel) -> a multiplier novelesevel
    a likvidacios savok kijjebb kerulnek, es az erosseg is no.
    """
    longs, shorts = {}, {}
    for lev in LEVERAGE_TIERS:
        margin = 1 / lev * 0.9 * htf_multiplier
        longs[lev] = price * (1 - margin)
        shorts[lev] = price * (1 + margin)

    long_share = ls_ratio / (1 + ls_ratio) if ls_ratio > 0 else 0.5
    short_share = 1 - long_share
    return {"long_liqs": longs, "short_liqs": shorts,
            "total_long": oi_usd * long_share * htf_multiplier,
            "total_short": oi_usd * short_share * htf_multiplier}


def analyze_multi_tf(symbol: str, price: float, atr: float,
                     oi_usd: float, ls_ratio: float) -> dict:
    """Multi-timeframe likvidacios terkep (1H/2H/4H/1D)."""
    tf_map = {}
    for tf_label, mult in [("1H", 0.5), ("2H", 0.75), ("4H", 1.0), ("1D", 1.5)]:
        longs, shorts = {}, {}
        for lev in LEVERAGE_TIERS:
            margin = 1 / lev * 0.9 * mult
            longs[lev] = price * (1 - margin)
            shorts[lev] = price * (1 + margin)
        tf_map[tf_label] = {"long_liqs": longs, "short_liqs": shorts, "mult": mult}
    return tf_map


def print_multi_tf(symbol: str, tf_map: dict, price: float) -> None:
    """Multi-timeframe liq map kiiras."""
    print(f"\n  MULTI-TIMEFRAME LIQ MAP — {symbol}")
    print(f"  {'TF':<5} {'25x LONG liq':>14} {'25x SHORT liq':>14} {'Sav':>10}")
    print(f"  {'-'*48}")
    for tf in ("1H", "2H", "4H", "1D"):
        m = tf_map[tf]
        ll = m["long_liqs"][25]
        sl = m["short_liqs"][25]
        spread = (sl - ll) / price * 100
        print(f"  {tf:<5} ${ll:>12,.4g} ${sl:>12,.4g} {spread:>8.1f}%")


# ---------------------------------------------------------------------------
# BOUNCE RATE
# ---------------------------------------------------------------------------
def calc_bounce_rate(df, levels: list, tolerance: float = 0.015) -> dict:
    if df is None or len(df) < 30 or not levels:
        return {"rate": None, "events": 0}
    close = df["close"].values
    bounces = 0
    touches = 0
    for i in range(1, len(close) - 1):
        for lvl in levels:
            if abs(close[i] - lvl) / lvl < tolerance:
                touches += 1
                if abs(close[i + 1] - lvl) > abs(close[i] - lvl):
                    bounces += 1
                break
    rate = bounces / touches * 100 if touches > 0 else None
    return {"rate": rate, "events": touches}


# ---------------------------------------------------------------------------
# MAIN ANALYSIS
# ---------------------------------------------------------------------------
def analyze_liquidation_map(symbol: str, price: float, atr: float = 0,
                            df=None, htf_levels: bool = False,
                            zone_width: float = DEFAULT_ZONE_WIDTH,
                            piv_count: int = 5, tf: str | None = None) -> dict:
    """Teljes likvidacios elemzes az osszes bovitmenynyel.

    Parameterek:
      htf_levels:  True -> magasabb TF szintek is (1D-es multiplierrel)
      zone_width:  ATR-szorzo a klaszter savhoz (0.1/0.2/0.3/0.5)
      piv_count:   hany pivot szintet detektaljunk (5/10/15)
      tf:          multi-TF mod ('1H'/'2H'/'4H'/'1D'); None=default
    """
    meta = fetch_futures_meta(symbol, price)
    funding = meta["funding"]
    oi = meta["oi"]
    ls_ratio = meta["ls_ratio"]
    source = meta["source"]
    oi_usd = oi["oi_usd"] if oi else 0
    liqs = calc_liquidation_levels(price, oi_usd, ls_ratio or 1.0)

    signals = []
    score_long = 0
    score_short = 0
    liq_hunt = None
    regime = detect_regime(df)
    proxy_mode = (df is None) or (source != "BINANCE")

    # Funding + next funding timer
    funding_timer = next_funding_time()
    if funding is not None:
        bias = "long bias" if funding > 0 else "short bias"
        signals.append(f"Funding rate: {funding:+.4f}% ({bias}) | Next: {funding_timer}")
        if funding > 0.02:
            score_short += 5
        elif funding < -0.02:
            score_long += 5
    else:
        signals.append(f"Next funding: {funding_timer}")

    # L/S ratio
    if ls_ratio is not None:
        signals.append(f"Long/Short ratio: {ls_ratio:.2f}")
        if ls_ratio > 1.5:
            score_short += 5
            signals.append("  -> Tul sok long (short edge)")
        elif ls_ratio < 0.7:
            score_long += 5
            signals.append("  -> Tul sok short, squeeze setup (long edge)")

    if oi and oi.get("oi_usd"):
        signals.append(f"Open Interest ({source}): ${oi['oi_usd'] / 1e6:.1f}M")

    # --- ASZIMMETRIA ---
    total_long = liqs["total_long"]
    total_short = liqs["total_short"]
    asymmetry = None
    asym_dir = None
    if total_long > 0 and total_short > 0:
        asymmetry = total_long / total_short
        if asymmetry < 0.3:
            asym_dir = "SHORT"
            score_short += 15
            signals.append(f"ASZIMMETRIA {asymmetry:.2f}: eros SHORT likviditas felfele (+15)")
        elif asymmetry > 3.0:
            asym_dir = "LONG"
            score_long += 15
            signals.append(f"ASZIMMETRIA {asymmetry:.2f}: eros LONG likviditas lefele (+15)")
        else:
            if total_short > total_long:
                asym_dir = "SHORT_lean"
                signals.append(f"Likviditas: ${total_short / 1e6:.0f}M felfele vs ${total_long / 1e6:.0f}M lefele")
            else:
                asym_dir = "LONG_lean"
                signals.append(f"Likviditas: ${total_long / 1e6:.0f}M lefele vs ${total_short / 1e6:.0f}M felfele")

    # --- ZONE-BASED CLUSTER DENSITY ---
    zone_half = atr * zone_width if atr > 0 else price * 0.02
    zone_density = cluster_density_with_zone(liqs, price, zone_half)
    density_str = zone_density["str"]

    # --- HTF LEVELS ---
    htf = None
    if htf_levels:
        htf = calc_htf_levels(price, oi_usd, ls_ratio or 1.0, htf_multiplier=1.5)

    # --- PIVOT LEVELS ---
    pivots = calc_pivot_levels(df, piv_count) if df is not None else []

    # --- KIEMELT SZINTEK ---
    highlights = find_highlighted_levels(liqs, price, atr, pivots) if atr > 0 else []

    # --- ATTRACTOR DISTANCE (top-3) ---
    attractors = calc_attractors(liqs, price, atr, pivots)

    # --- NEAREST MAGNET ---
    nearest_magnet = None
    magnet_dist_atr = None
    magnet_dir = None
    if atr > 0:
        candidates = []
        for lev in (25, 50):
            candidates.append(("SHORT", liqs["short_liqs"][lev]))
            candidates.append(("LONG", liqs["long_liqs"][lev]))
        nearest = min(candidates, key=lambda x: abs(x[1] - price))
        magnet_dir, nearest_magnet = nearest
        magnet_dist_atr = abs(nearest_magnet - price) / atr
        if magnet_dist_atr < 3:
            signals.append(f"Magnes KOZEL ({magnet_dist_atr:.1f} ATR) {magnet_dir} iranyba (+5)")
            if magnet_dir == "SHORT":
                score_short += 5
            else:
                score_long += 5

    # --- BOUNCE RATE ---
    all_liq_levels = list(liqs["short_liqs"].values()) + list(liqs["long_liqs"].values())
    bounce = calc_bounce_rate(df, all_liq_levels)

    # --- CVD BIAS ---
    cvd_bias = None
    if df is not None and len(df) >= 5:
        recent = df.iloc[-5:]
        buy_vol = recent[recent["close"] >= recent["open"]]["volume"].sum()
        sell_vol = recent[recent["close"] < recent["open"]]["volume"].sum()
        if buy_vol + sell_vol > 0:
            cvd = (buy_vol - sell_vol) / (buy_vol + sell_vol)
            if cvd > 0.2:
                cvd_bias = "LONG"
                signals.append(f"CVD bias: LONG (vetel dominal, {cvd:+.2f})")
            elif cvd < -0.2:
                cvd_bias = "SHORT"
                signals.append(f"CVD bias: SHORT (eladas dominal, {cvd:+.2f})")
            else:
                cvd_bias = "NEUTRAL"

    # Liq hunt
    if funding is not None and ls_ratio is not None:
        if funding > 0.02 and ls_ratio > 1.5:
            liq_hunt = "ARCSOKKENES gyanu (longok zsufoltak)"
            score_short += 5
        elif funding < -0.02 and ls_ratio < 0.7:
            liq_hunt = "SHORT SQUEEZE setup (shortok zsufoltak)"
            score_long += 5

    # --- SWEEP LOG (eletciklussal) ---
    force_orders = fetch_binance_force_orders(symbol)
    sweep = build_sweep_log(df, liqs, price, symbol, force_orders)

    # --- LIQ IMBALANCE ---
    if total_long > 0 and total_short > 0:
        imb_ratio = total_short / total_long
    else:
        imb_ratio = 1.0
    if imb_ratio > 1.5:
        liq_imbalance = "▲▲ BULLISH" if imb_ratio > 2.5 else "▲ BULLISH"
    elif imb_ratio < 0.67:
        liq_imbalance = "▼▼ BEARISH" if imb_ratio < 0.4 else "▼ BEARISH"
    else:
        liq_imbalance = "= NEUTRAL"

    # --- ACTIVE klaszterek ---
    band15 = 0.15
    active_short = [lev for lev in LEVERAGE_TIERS
                    if abs(liqs["short_liqs"][lev] - price) / price <= band15]
    active_long = [lev for lev in LEVERAGE_TIERS
                   if abs(liqs["long_liqs"][lev] - price) / price <= band15]
    active_count = len(active_short) + len(active_long)

    # --- MULTI-TF ---
    tf_map = None
    if tf or htf_levels:
        tf_map = analyze_multi_tf(symbol, price, atr, oi_usd, ls_ratio or 1.0)

    # Edge
    if score_long > score_short:
        edge, edge_score = "LONG", min(score_long, 25)
    elif score_short > score_long:
        edge, edge_score = "SHORT", min(score_short, 25)
    else:
        edge, edge_score = "NEUTRAL", 0

    return {
        "funding": funding, "oi": oi, "ls_ratio": ls_ratio, "liqs": liqs,
        "signals": signals, "liq_hunt": liq_hunt, "edge": edge,
        "edge_score": edge_score, "regime": regime, "proxy_mode": proxy_mode,
        "source": source,
        "asymmetry": asymmetry, "asym_dir": asym_dir, "density": density_str,
        "nearest_magnet": nearest_magnet, "magnet_dist_atr": magnet_dist_atr,
        "magnet_dir": magnet_dir, "bounce": bounce, "cvd_bias": cvd_bias,
        "total_long": total_long, "total_short": total_short,
        "sweep": sweep, "liq_imbalance": liq_imbalance,
        "active_count": active_count, "active_long": len(active_long),
        "active_short": len(active_short),
        "available": funding is not None or ls_ratio is not None,
        "htf": htf, "htf_enabled": htf_levels,
        "highlights": highlights, "attractors": attractors,
        "pivots": pivots, "piv_count": piv_count,
        "zone_width": zone_width, "zone_half": zone_half,
        "funding_timer": funding_timer, "tf_map": tf_map, "tf": tf,
    }


# ---------------------------------------------------------------------------
# PRINT OUTPUT
# ---------------------------------------------------------------------------
def print_liquidation_map(symbol: str, lm: dict) -> None:
    src = lm.get("source", "OKX")
    proxy = lm["proxy_mode"]
    regime_str = f"{'PROXY | ' if proxy else ''}{lm['regime']}"

    print(f"\n LIKVIDACIOS TERKEP — {symbol} [{regime_str}]")
    print(f"{'=' * 72}")

    # 9. PROXY MODE figyelmeztetés
    if proxy:
        print(f"  !! PROXY MODE — kulso adatforras, kevesbe megbizhato (-10 score)")
        print(f"     Ne lepj be ha PROXY + alacsony confidence!")
        print(f"{'-' * 72}")

    if not lm["available"]:
        print("  Nincs futures adat (Binance/OKX API nem elerheto).")
        return

    for s in lm["signals"]:
        print(f"  {s}")

    print(f"  Active: {lm.get('active_count', 0)} | "
          f"Long/Short: {lm.get('active_long', 0)}/{lm.get('active_short', 0)} "
          f"(aszimmetria!)" if lm.get("active_long", 0) != lm.get("active_short", 0) else
          f"  Active: {lm.get('active_count', 0)} | "
          f"Long/Short: {lm.get('active_long', 0)}/{lm.get('active_short', 0)}")
    print(f"  Data Regime: {regime_str} | Forras: {src}")
    print(f"  Liq Imbalance: {lm.get('liq_imbalance', '= NEUTRAL')}")
    print(f"  Zone Width: ATR×{lm.get('zone_width', 0.3)} | "
          f"Pivots: {lm.get('piv_count', 5)}")

    liqs = lm["liqs"]

    # --- KIEMELT SZINTEK ---
    highlights = lm.get("highlights", [])
    if highlights:
        print(f"\n  KIEMELT SZINTEK:")
        for h in highlights:
            tag = "MAJOR" if h["type"] == "MAJOR_RESISTANCE" else "LIQ IMB"
            imb_str = f" (${h.get('imb_usd', 0) / 1e6:.0f}M)" if "imb_usd" in h else ""
            print(f"  >> ${h['level']:,.4g} [{tag}] {h['dir']} "
                  f"({h.get('dist_atr', 0):.1f} ATR, {h.get('count', 0)} klaszter){imb_str}")

    # --- KLASZTEREK ---
    if liqs and lm["total_short"] > 0:
        print(f"\n  LIKVIDACIOS KLASZTEREK (felfele - short liq, ossz ${lm['total_short'] / 1e6:.0f}M):")
        for lev in sorted(LEVERAGE_TIERS):
            p = liqs["short_liqs"][lev]
            n = liqs["short_notional"][lev]
            is_hl = any(abs(h["level"] - p) / p < 0.005 for h in highlights)
            marker = " <<" if is_hl else ""
            print(f"    ${p:,.4g} — ${n / 1e6:.1f}M ({lev}x shortok){marker}")
        print(f"\n  LIKVIDACIOS KLASZTEREK (lefele - long liq, ossz ${lm['total_long'] / 1e6:.0f}M):")
        for lev in sorted(LEVERAGE_TIERS):
            p = liqs["long_liqs"][lev]
            n = liqs["long_notional"][lev]
            is_hl = any(abs(h["level"] - p) / p < 0.005 for h in highlights)
            marker = " <<" if is_hl else ""
            print(f"    ${p:,.4g} — ${n / 1e6:.1f}M ({lev}x longok){marker}")

    # --- HTF LEVELS ---
    if lm.get("htf_enabled") and lm.get("htf"):
        htf = lm["htf"]
        print(f"\n  HTF LEVELS (1D, erossebb magneses):")
        for lev in (25, 50):
            print(f"    SHORT liq ${htf['short_liqs'][lev]:,.4g} | "
                  f"LONG liq ${htf['long_liqs'][lev]:,.4g} ({lev}x)")

    print(f"\n  Cluster density: {lm['density']}")
    if lm["asymmetry"] is not None:
        print(f"  Aszimmetria (long/short liq): {lm['asymmetry']:.2f} -> {lm['asym_dir']}")
    if lm["bounce"]["rate"] is not None:
        print(f"  Bounce rate: {lm['bounce']['rate']:.1f}% ({lm['bounce']['events']} esemeny)")
    if lm["cvd_bias"]:
        print(f"  CVD bias: {lm['cvd_bias']}")
    if lm["nearest_magnet"]:
        atr_str = f" ({lm['magnet_dist_atr']:.1f} ATR)" if lm["magnet_dist_atr"] else ""
        print(f"  Legkozelebbi magnes: ${lm['nearest_magnet']:,.4g} [{lm['magnet_dir']}]{atr_str}")
    if lm["liq_hunt"]:
        print(f"  LIQ HUNT: {lm['liq_hunt']}")

    # --- 8. ATTRACTOR DISTANCE ---
    attractors = lm.get("attractors", [])
    if attractors:
        print(f"\n  ATTRACTOR DISTANCE:")
        for i, a in enumerate(attractors):
            ord_label = ["1st", "2nd", "3rd"][i] if i < 3 else f"{i + 1}th"
            print(f"    {ord_label}: ${a['level']:,.4g} ({a['dist_atr']} ATR {a['dir']}, {a['size']})")

    # --- 5. SWEEP LOG (eletciklus) ---
    sweep = lm.get("sweep")
    if sweep:
        print(f"\n  SWEEP LOG (klaszter status):")
        if sweep["pending_long"]:
            print(f"    ▼ LONG likvidacio: {sweep['pending_long']}x PENDING")
        if sweep["pending_short"]:
            print(f"    ▲ SHORT likvidacio: {sweep['pending_short']}x PENDING")
        if sweep.get("active"):
            print(f"    >> {sweep['active']}x ACTIVE (ar kozel a szinthez)")
        if sweep["completed"]:
            print(f"    ✓ {sweep['completed']}x COMPLETED (mar sweepelt)")
        if sweep.get("failed"):
            print(f"    ✗ {sweep['failed']}x FAILED (visszafordult)")
        if sweep["real_orders"]:
            print(f"    Valos force orderek ({len(sweep['real_orders'])}):")
            for ro in sweep["real_orders"][-3:]:
                print(f"      {ro['dir']} liq @ ${ro['price']:,.4g}")

    # --- 7. MULTI-TF ---
    tf_map = lm.get("tf_map")
    if tf_map:
        # price a liqs-bol becslunk (25x short + long atlag)
        approx_price = (lm["liqs"]["short_liqs"][25] + lm["liqs"]["long_liqs"][25]) / 2
        print_multi_tf(symbol, tf_map, approx_price)

    # --- 9. PROXY EDGE MODIFIER ---
    proxy_note = " [PROXY -10]" if proxy else ""
    print(f"\n  EDGE: {lm['edge']} (+{lm['edge_score']} pont){proxy_note}")
    print(f"{'=' * 72}")
