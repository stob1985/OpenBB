"""
Liquidation Map modul
=====================
Leveraged pozíciók becsült likvidációs szintjeit szamolja, es a futures
piaci metaadatokat (funding rate, open interest, long/short ratio) gyujti.

Adatforras: OKX public API (a Binance Futures geo-korlatozott).
A liquidation klaszterek a jelenlegi arbol + tokeattetel szintekbol szamoltak.
"""

import requests

OKX_BASE = "https://www.okx.com/api/v5"
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


def calc_liquidation_levels(price: float) -> dict:
    """Likvidacios szintek tokeattetel szintenkent, mindket iranyba."""
    longs = {}   # ar csokkenes -> long likvidacio
    shorts = {}  # ar emelkedes -> short likvidacio
    for lev in LEVERAGE_TIERS:
        # 0.9 buffer a maintenance margin miatt
        longs[lev] = price * (1 - 1 / lev * 0.9)
        shorts[lev] = price * (1 + 1 / lev * 0.9)
    return {"long_liqs": longs, "short_liqs": shorts}


def analyze_liquidation_map(symbol: str, price: float, atr: float = 0) -> dict:
    """Teljes likvidacios elemzes + scoring edge."""
    funding = fetch_funding_rate(symbol)
    oi = fetch_open_interest(symbol)
    ls_ratio = fetch_long_short_ratio(symbol)
    liqs = calc_liquidation_levels(price)

    signals = []
    score_long = 0
    score_short = 0
    liq_hunt = None

    # Funding rate jelzes
    if funding is not None:
        bias = "long bias" if funding > 0 else "short bias"
        signals.append(f"Funding rate: {funding:+.4f}% ({bias})")
        if funding > 0.02:
            score_short += 5
            signals.append("  -> Funding tul pozitiv, longok zsufoltak (short edge)")
        elif funding < -0.02:
            score_long += 5
            signals.append("  -> Funding tul negativ, shortok zsufoltak (long edge)")

    # Long/Short ratio
    if ls_ratio is not None:
        signals.append(f"Long/Short ratio: {ls_ratio:.2f}")
        if ls_ratio > 1.5:
            score_short += 5
            signals.append("  -> Tul sok long, likvidalodhat lefele (short edge)")
        elif ls_ratio < 0.7:
            score_long += 5
            signals.append("  -> Tul sok short, short squeeze johet (long edge)")

    # Open Interest
    if oi:
        signals.append(f"Open Interest: ${oi['oi_usd']/1e9:.2f}B")

    # Liq hunt gyanu
    if funding is not None and ls_ratio is not None:
        if funding > 0.02 and ls_ratio > 1.5:
            liq_hunt = "ARCSOKKENES gyanu (piac tul bullish, lefele likvidalhat)"
            score_short += 5
        elif funding < -0.02 and ls_ratio < 0.7:
            liq_hunt = "ARNOVEKEDES gyanu (short squeeze setup)"
            score_long += 5

    # Legkozelebbi magnes (nagy klaszter)
    nearest_magnet = None
    magnet_dist_atr = None
    if atr > 0:
        # A 25x es 50x klaszterek a legnagyobbak tipikusan
        candidates = [
            ("short", liqs["short_liqs"][25], "25x short cluster"),
            ("short", liqs["short_liqs"][50], "50x short cluster"),
            ("long", liqs["long_liqs"][25], "25x long cluster"),
            ("long", liqs["long_liqs"][50], "50x long cluster"),
        ]
        nearest = min(candidates, key=lambda x: abs(x[1] - price))
        nearest_magnet = nearest[1]
        magnet_dist_atr = abs(nearest_magnet - price) / atr

    # Score edge: max +10
    if score_long > score_short:
        edge = "LONG"
        edge_score = min(score_long, 10)
    elif score_short > score_long:
        edge = "SHORT"
        edge_score = min(score_short, 10)
    else:
        edge = "NEUTRAL"
        edge_score = 0

    return {
        "funding": funding, "oi": oi, "ls_ratio": ls_ratio,
        "liqs": liqs, "signals": signals, "liq_hunt": liq_hunt,
        "edge": edge, "edge_score": edge_score,
        "nearest_magnet": nearest_magnet, "magnet_dist_atr": magnet_dist_atr,
        "available": funding is not None or ls_ratio is not None,
    }


def print_liquidation_map(symbol: str, lm: dict) -> None:
    """Likvidacios terkep kiiras."""
    print(f"\n LIKVIDACIOS TERKEP — {symbol}")
    print(f"{'-' * 60}")
    if not lm["available"]:
        print("  Nincs futures adat (OKX API nem elerheto erre a parra).")
        return

    for s in lm["signals"]:
        print(f"  {s}")

    liqs = lm["liqs"]
    if liqs:
        print(f"\n  LIKVIDACIOS KLASZTEREK (felfele - short likvidaciok):")
        for lev in sorted(LEVERAGE_TIERS):
            p = liqs["short_liqs"][lev]
            size = "HATALMAS" if lev == 100 else ("NAGY" if lev == 50 else ("KOZEPES" if lev == 25 else "kicsi"))
            print(f"    ${p:,.4g} — {lev}x shortok ({size})")
        print(f"\n  LIKVIDACIOS KLASZTEREK (lefele - long likvidaciok):")
        for lev in sorted(LEVERAGE_TIERS):
            p = liqs["long_liqs"][lev]
            size = "HATALMAS" if lev == 100 else ("NAGY" if lev == 50 else ("KOZEPES" if lev == 25 else "kicsi"))
            print(f"    ${p:,.4g} — {lev}x longok ({size})")

    if lm["liq_hunt"]:
        print(f"\n  LIQ HUNT GYANU: {lm['liq_hunt']}")

    if lm["nearest_magnet"]:
        atr_str = f" ({lm['magnet_dist_atr']:.1f} ATR)" if lm['magnet_dist_atr'] else ""
        print(f"  Legkozelebbi magnes: ${lm['nearest_magnet']:,.4g}{atr_str}")

    print(f"\n  EDGE: {lm['edge']} (+{lm['edge_score']} pont)")
