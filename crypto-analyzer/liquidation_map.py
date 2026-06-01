"""
Liquidation Map modul
=====================
Leveraged pozíciók becsült likvidációs szintjeit szamolja, es a futures
piaci metaadatokat (funding rate, open interest, long/short ratio) gyujti.

Adatforras prioritas:
  1. Binance Futures (fapi.binance.com)  — elsodleges
  2. OKX public API                       — fallback (PROXY mod)
  3. CoinGlass public                      — liquidation history / sweep log
A Binance fapi sok regioban geo-korlatozott (451); ilyenkor automatikusan
OKX-re vagy proxy adatra esik vissza, es a Data Regime PROXY-t jelez.

A liquidation klaszterek a jelenlegi arbol + tokeattetel szintekbol szamoltak.
"""

import requests

BINANCE_FAPI = "https://fapi.binance.com"
OKX_BASE = "https://www.okx.com/api/v5"
COINGLASS_BASE = "https://open-api.coinglass.com/public/v2"
_HEADERS = {"User-Agent": "Mozilla/5.0"}

# Tokeattetel szintek amiket figyelunk
LEVERAGE_TIERS = [10, 25, 50, 100]


def _to_okx_swap(symbol: str) -> str:
    """BTCUSDT -> BTC-USDT-SWAP."""
    s = symbol.upper().replace("-", "")
    for q in ("USDT", "USDC", "USD"):
        if s.endswith(q):
            base = s[: -len(q)]
            return f"{base}-{q}-SWAP"
    return f"{symbol}-USDT-SWAP"


def _okx_ccy(symbol: str) -> str:
    """BTCUSDT -> BTC."""
    s = symbol.upper().replace("-", "")
    for q in ("USDT", "USDC", "USD"):
        if s.endswith(q):
            return s[: -len(q)]
    return symbol


def _get(url: str, params: dict) -> dict | None:
    for _retry in range(3):
        try:
            r = requests.get(url, params=params, timeout=10, headers=_HEADERS)
            if r.status_code == 200:
                d = r.json()
                if d.get("code") == "0":
                    return d
            return None
        except Exception:
            import time
            time.sleep(1)
    return None


def fetch_funding_rate(symbol: str) -> float | None:
    """Aktualis funding rate (%)."""
    d = _get(f"{OKX_BASE}/public/funding-rate", {"instId": _to_okx_swap(symbol)})
    if d and d.get("data"):
        try:
            return float(d["data"][0]["fundingRate"]) * 100
        except (KeyError, ValueError, IndexError):
            return None
    return None


def fetch_open_interest(symbol: str) -> dict | None:
    """Open interest USD-ben es coin-ban."""
    d = _get(f"{OKX_BASE}/public/open-interest",
             {"instType": "SWAP", "instId": _to_okx_swap(symbol)})
    if d and d.get("data"):
        try:
            row = d["data"][0]
            return {
                "oi_coin": float(row.get("oi", 0)),
                "oi_usd": float(row.get("oiUsd", 0)),
            }
        except (ValueError, IndexError):
            return None
    return None


def fetch_long_short_ratio(symbol: str) -> float | None:
    """Top trader long/short account ratio."""
    d = _get(f"{OKX_BASE}/rubik/stat/contracts/long-short-account-ratio",
             {"ccy": _okx_ccy(symbol), "period": "1H"})
    if d and d.get("data"):
        try:
            # [timestamp, ratio] - legfrissebb az elso
            return float(d["data"][0][1])
        except (ValueError, IndexError):
            return None
    return None


# ---------------------------------------------------------------------------
# BINANCE FUTURES (elsodleges; sok regioban geo-korlatozott -> graceful fallback)
# ---------------------------------------------------------------------------
def _binance_symbol(symbol: str) -> str:
    """BTC-USDT / BTCUSDT -> BTCUSDT."""
    return symbol.upper().replace("-", "")


def _binance_get(path: str, params: dict) -> object | None:
    """Binance fapi GET. None ha geo-blokk (451), timeout vagy hiba."""
    try:
        r = requests.get(f"{BINANCE_FAPI}{path}", params=params,
                         timeout=8, headers=_HEADERS)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def fetch_binance_funding(symbol: str) -> float | None:
    """Binance premiumIndex -> lastFundingRate (%)."""
    d = _binance_get("/fapi/v1/premiumIndex", {"symbol": _binance_symbol(symbol)})
    if isinstance(d, dict):
        try:
            return float(d["lastFundingRate"]) * 100
        except (KeyError, ValueError, TypeError):
            return None
    return None


def fetch_binance_oi(symbol: str, price: float = 0) -> dict | None:
    """Binance openInterest (coin) -> USD becsles ar alapjan."""
    d = _binance_get("/fapi/v1/openInterest", {"symbol": _binance_symbol(symbol)})
    if isinstance(d, dict):
        try:
            oi_coin = float(d["openInterest"])
            return {"oi_coin": oi_coin, "oi_usd": oi_coin * price if price else 0}
        except (KeyError, ValueError, TypeError):
            return None
    return None


def fetch_binance_ls_ratio(symbol: str) -> float | None:
    """Binance topLongShortAccountRatio (legfrissebb)."""
    d = _binance_get("/futures/data/topLongShortAccountRatio",
                     {"symbol": _binance_symbol(symbol), "period": "1h", "limit": 1})
    if isinstance(d, list) and d:
        try:
            return float(d[-1]["longShortRatio"])
        except (KeyError, ValueError, TypeError, IndexError):
            return None
    return None


def fetch_binance_force_orders(symbol: str, limit: int = 50) -> list | None:
    """Binance allForceOrders (valos likvidaciok). None ha nem elerheto."""
    d = _binance_get("/fapi/v1/allForceOrders",
                     {"symbol": _binance_symbol(symbol), "limit": limit})
    if isinstance(d, list):
        return d
    return None


def fetch_coinglass_liquidations(symbol: str) -> list | None:
    """CoinGlass public liquidation history (proxy adat). None ha nem elerheto."""
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
    """Egyesitett futures metaadat: Binance elsodleges, OKX fallback.

    Visszaad: funding, oi, ls_ratio, source ('BINANCE' / 'OKX' / 'PROXY').
    """
    # 1) Binance fapi probalkozas
    funding = fetch_binance_funding(symbol)
    oi = fetch_binance_oi(symbol, price)
    ls_ratio = fetch_binance_ls_ratio(symbol)
    source = "BINANCE"

    # 2) Ami hianyzik, OKX-rol potoljuk (PROXY)
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


def build_sweep_log(df, liqs: dict, price: float,
                    force_orders: list | None = None) -> dict:
    """Sweep log: melyik likvidacios szintek PENDING vs COMPLETED.

    Egy klaszter COMPLETED, ha az utolso ~10 nap ara mar elerte/atlepte;
    egyebkent PENDING (meg nem sweepelt magnes). Ha vannak valos force
    orderek (Binance), azokat is beemeli.
    """
    entries = []
    lookback = None
    if df is not None and len(df) >= 2:
        lookback = df.iloc[-10:] if len(df) >= 10 else df
        hi = float(lookback["high"].max())
        lo = float(lookback["low"].min())
    else:
        hi = lo = price

    # SHORT likvidaciok felfele (ar emelkedeskor sweepelodnek)
    for lev in sorted(LEVERAGE_TIERS):
        lvl = liqs["short_liqs"][lev]
        status = "COMPLETED" if hi >= lvl else "PENDING"
        entries.append({"dir": "SHORT", "lev": lev, "level": lvl, "status": status})
    # LONG likvidaciok lefele (ar eseskor sweepelodnek)
    for lev in sorted(LEVERAGE_TIERS):
        lvl = liqs["long_liqs"][lev]
        status = "COMPLETED" if lo <= lvl else "PENDING"
        entries.append({"dir": "LONG", "lev": lev, "level": lvl, "status": status})

    pending_long = sum(1 for e in entries if e["dir"] == "LONG" and e["status"] == "PENDING")
    pending_short = sum(1 for e in entries if e["dir"] == "SHORT" and e["status"] == "PENDING")
    completed = sum(1 for e in entries if e["status"] == "COMPLETED")

    # Valos force orderek (ha elerheto - altalaban geo-blokk miatt None)
    real = []
    if force_orders:
        for fo in force_orders[-10:]:
            try:
                side = fo.get("side", "")
                # SELL force order = LONG pozicio likvidalva; BUY = SHORT likvidalva
                ldir = "LONG" if side == "SELL" else "SHORT"
                real.append({"dir": ldir, "price": float(fo.get("price", 0)),
                             "qty": float(fo.get("origQty", 0))})
            except (ValueError, TypeError):
                continue

    return {"entries": entries, "pending_long": pending_long,
            "pending_short": pending_short, "completed": completed,
            "real_orders": real}




# Tokeattetel sulyozas (10x es 25x a leggyakoribb retail, 100x ritka)
LEV_WEIGHTS = {10: 0.40, 25: 0.30, 50: 0.20, 100: 0.10}


def detect_regime(df) -> str:
    """Piaci rezsim: NORMAL / HIGH_VOL / BEAR / BULL."""
    if df is None or len(df) < 30:
        return "NORMAL"
    close = df["close"]
    tr = (df["high"] - df["low"])
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
    """Likvidacios szintek + becsult notional minden szinten."""
    longs, shorts = {}, {}
    long_notional, short_notional = {}, {}

    # OI felosztas long/short oldalra a L/S ratio alapjan
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


def calc_bounce_rate(df, levels: list, tolerance: float = 0.015) -> dict:
    """Historikus bounce rate a likvidacios szintek kozeleben.
    Hanyszor pattant vissza az ar a szintekrol?"""
    if df is None or len(df) < 30 or not levels:
        return {"rate": None, "events": 0}
    close = df["close"].values
    bounces = 0
    touches = 0
    for i in range(1, len(close) - 1):
        for lvl in levels:
            if abs(close[i] - lvl) / lvl < tolerance:
                touches += 1
                # Bounce = az ar elmozdult a szinttol a kovetkezo napon
                if abs(close[i+1] - lvl) > abs(close[i] - lvl):
                    bounces += 1
                break
    rate = bounces / touches * 100 if touches > 0 else None
    return {"rate": rate, "events": touches}


def analyze_liquidation_map(symbol: str, price: float, atr: float = 0,
                            df=None) -> dict:
    """Teljes likvidacios elemzes: density, aszimmetria, regime, bounce, CVD."""
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
    # PROXY mod: ha nincs OHLCV VAGY a futures adat nem Binance-rol jott
    proxy_mode = (df is None) or (source != "BINANCE")

    # Funding
    if funding is not None:
        bias = "long bias" if funding > 0 else "short bias"
        signals.append(f"Funding rate: {funding:+.4f}% ({bias})")
        if funding > 0.02:
            score_short += 5
        elif funding < -0.02:
            score_long += 5

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
        signals.append(f"Open Interest ({source}): ${oi['oi_usd']/1e6:.1f}M")

    # --- ASZIMMETRIA (dollar notional alapjan) ---
    total_long = liqs["total_long"]
    total_short = liqs["total_short"]
    asymmetry = None
    asym_dir = None
    if total_long > 0 and total_short > 0:
        asymmetry = total_long / total_short  # long_liq / short_liq
        if asymmetry < 0.3:
            asym_dir = "SHORT"  # sok short likviditas felfele
            score_short += 15
            signals.append(f"ASZIMMETRIA {asymmetry:.2f}: eros SHORT likviditas felfele (+15)")
        elif asymmetry > 3.0:
            asym_dir = "LONG"
            score_long += 15
            signals.append(f"ASZIMMETRIA {asymmetry:.2f}: eros LONG likviditas lefele (+15)")
        else:
            # Melyik oldalon van tobb $
            if total_short > total_long:
                asym_dir = "SHORT_lean"
                signals.append(f"Likviditas: ${total_short/1e6:.0f}M felfele vs ${total_long/1e6:.0f}M lefele")
            else:
                asym_dir = "LONG_lean"
                signals.append(f"Likviditas: ${total_long/1e6:.0f}M lefele vs ${total_short/1e6:.0f}M felfele")

    # --- CLUSTER DENSITY (X% savban hany szint) ---
    band = 0.05  # 5% sav
    up_clusters = [l for l in liqs["short_liqs"].values() if 0 < (l - price) / price <= band]
    dn_clusters = [l for l in liqs["long_liqs"].values() if 0 < (price - l) / price <= band]
    up_density = sum(LEV_WEIGHTS[lev] for lev in LEVERAGE_TIERS if 0 < (liqs["short_liqs"][lev] - price)/price <= band)
    dn_density = sum(LEV_WEIGHTS[lev] for lev in LEVERAGE_TIERS if 0 < (price - liqs["long_liqs"][lev])/price <= band)
    density_str = f"FEL: {len(up_clusters)}L/{up_density:.1f}W | LE: {len(dn_clusters)}L/{dn_density:.1f}W"

    # --- NEAREST MAGNET (ATR-ben) ---
    nearest_magnet = None
    magnet_dist_atr = None
    magnet_dir = None
    if atr > 0:
        candidates = []
        for lev in (25, 50):  # a legnagyobb klaszterek
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

    # --- CVD BIAS (approx: utolso napok zaro vs nyito) ---
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

    # --- SWEEP LOG ---
    force_orders = fetch_binance_force_orders(symbol)
    sweep = build_sweep_log(df, liqs, price, force_orders)

    # --- LIQ IMBALANCE (vizualis irany) ---
    if total_long > 0 and total_short > 0:
        imb_ratio = total_short / total_long  # felfele / lefele likviditas
    else:
        imb_ratio = 1.0
    if imb_ratio > 1.5:
        liq_imbalance = "▲▲ BULLISH" if imb_ratio > 2.5 else "▲ BULLISH"
    elif imb_ratio < 0.67:
        liq_imbalance = "▼▼ BEARISH" if imb_ratio < 0.4 else "▼ BEARISH"
    else:
        liq_imbalance = "= NEUTRAL"

    # --- ACTIVE klaszterek (15% savon belul) + long/short bontas ---
    band15 = 0.15
    active_short = [lev for lev in LEVERAGE_TIERS
                    if abs(liqs["short_liqs"][lev] - price) / price <= band15]
    active_long = [lev for lev in LEVERAGE_TIERS
                   if abs(liqs["long_liqs"][lev] - price) / price <= band15]
    active_count = len(active_short) + len(active_long)

    # Edge
    if score_long > score_short:
        edge, edge_score = "LONG", min(score_long, 25)
    elif score_short > score_long:
        edge, edge_score = "SHORT", min(score_short, 25)
    else:
        edge, edge_score = "NEUTRAL", 0

    # Megj.: a PROXY mod -10 buntetese a kombinalt dontesi motorban
    # ervenyesul (combined_decision), itt nem vonjuk le, hogy ne legyen dupla.

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
    }


def print_liquidation_map(symbol: str, lm: dict) -> None:
    """Likvidacios terkep kiiras."""
    src = lm.get("source", "OKX")
    regime_str = f"{'PROXY | ' if lm['proxy_mode'] else ''}{lm['regime']}"
    print(f"\n LIKVIDACIOS TERKEP — {symbol} [{regime_str}]")
    print(f"{'-' * 60}")
    if not lm["available"]:
        print("  Nincs futures adat (Binance/OKX API nem elerheto).")
        return

    for s in lm["signals"]:
        print(f"  {s}")

    # Active + aszimmetria sor
    print(f"  Active: {lm.get('active_count', 0)} | "
          f"Long/Short: {lm.get('active_long', 0)}/{lm.get('active_short', 0)}")
    print(f"  Data Regime: {regime_str} | Forras: {src}")
    print(f"  Liq Imbalance: {lm.get('liq_imbalance', '= NEUTRAL')}")

    liqs = lm["liqs"]
    if liqs and lm["total_short"] > 0:
        print(f"\n  LIKVIDACIOS KLASZTEREK (felfele - short liq, ossz ${lm['total_short']/1e6:.0f}M):")
        for lev in sorted(LEVERAGE_TIERS):
            p = liqs["short_liqs"][lev]
            n = liqs["short_notional"][lev]
            print(f"    ${p:,.4g} — ${n/1e6:.1f}M ({lev}x shortok)")
        print(f"\n  LIKVIDACIOS KLASZTEREK (lefele - long liq, ossz ${lm['total_long']/1e6:.0f}M):")
        for lev in sorted(LEVERAGE_TIERS):
            p = liqs["long_liqs"][lev]
            n = liqs["long_notional"][lev]
            print(f"    ${p:,.4g} — ${n/1e6:.1f}M ({lev}x longok)")

    print(f"\n  Cluster density: {lm['density']}")
    if lm["asymmetry"] is not None:
        print(f"  Aszimmetria (long/short liq): {lm['asymmetry']:.2f} -> {lm['asym_dir']}")
    if lm["bounce"]["rate"] is not None:
        print(f"  Bounce rate: {lm['bounce']['rate']:.1f}% ({lm['bounce']['events']} esemeny)")
    if lm["cvd_bias"]:
        print(f"  CVD bias: {lm['cvd_bias']}")
    if lm["nearest_magnet"]:
        atr_str = f" ({lm['magnet_dist_atr']:.1f} ATR)" if lm['magnet_dist_atr'] else ""
        print(f"  Legkozelebbi magnes: ${lm['nearest_magnet']:,.4g} [{lm['magnet_dir']}]{atr_str}")
    if lm["liq_hunt"]:
        print(f"  LIQ HUNT: {lm['liq_hunt']}")

    # --- SWEEP LOG ---
    sweep = lm.get("sweep")
    if sweep:
        print(f"\n  SWEEP LOG (klaszter status):")
        if sweep["pending_long"]:
            print(f"    ▼ LONG likvidacio: {sweep['pending_long']}x PENDING")
        if sweep["pending_short"]:
            print(f"    ▲ SHORT likvidacio: {sweep['pending_short']}x PENDING")
        if sweep["completed"]:
            print(f"    ✓ {sweep['completed']}x COMPLETED (mar sweepelt)")
        if sweep["real_orders"]:
            print(f"    Valos force orderek ({len(sweep['real_orders'])}):")
            for ro in sweep["real_orders"][-3:]:
                print(f"      {ro['dir']} liq @ ${ro['price']:,.4g}")

    print(f"\n  EDGE: {lm['edge']} (+{lm['edge_score']} pont)")


def _safe(liqs, p):
    return 1
