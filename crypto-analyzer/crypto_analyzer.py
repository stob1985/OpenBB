"""
Crypto Technical Analyzer
=========================
OpenBB-alapu kriptovaluta technikai elemzo script.
Lekeri a historikus arfolyamot, kiszamolja a fo indikatorokat
(RSI, MACD, Bollinger-szalagok, SMA 20/50/200), es megjeleniti charton.

Hasznalat:
    python crypto_analyzer.py                        # alapertelmezett: BTC-USD, 1 ev
    python crypto_analyzer.py --symbol ETH-USD
    python crypto_analyzer.py --symbol BTC-USD --days 365 --provider yfinance
"""

import argparse
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from openbb import obb


# ---------------------------------------------------------------------------
# 1. Adatlekerdezes
# ---------------------------------------------------------------------------
def fetch_crypto_data(
    symbol: str = "BTC-USD",
    days: int = 365,
    provider: str = "yfinance",
) -> pd.DataFrame:
    """Historikus arfolyamadatok lekerese az OpenBB-vel."""
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    print(f"Adatok lekerese: {symbol} ({start_date} -> {end_date}, provider={provider})")

    result = obb.crypto.price.historical(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        interval="1d",
        provider=provider,
    )
    df = result.to_df()
    if df.empty:
        raise ValueError(f"Nem erkezett adat a(z) {symbol} parhoz.")
    return df


# ---------------------------------------------------------------------------
# 2. Indikatorok szamitasa
# ---------------------------------------------------------------------------
def calc_sma(df: pd.DataFrame, window: int) -> pd.Series:
    """Egyszeru mozgoatlag (SMA)."""
    return df["close"].rolling(window=window).mean()


def calc_rsi(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Relative Strength Index (RSI)."""
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_macd(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD vonal, szignalvonal es hisztogram."""
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": histogram}
    )


def calc_bollinger(
    df: pd.DataFrame, length: int = 20, std_dev: float = 2.0
) -> pd.DataFrame:
    """Bollinger-szalagok (also, kozepso, felso)."""
    middle = df["close"].rolling(window=length).mean()
    std = df["close"].rolling(window=length).std()
    return pd.DataFrame(
        {
            "bb_upper": middle + std_dev * std,
            "bb_middle": middle,
            "bb_lower": middle - std_dev * std,
        }
    )


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Osszes indikator hozzaadasa a DataFrame-hez."""
    df["sma_20"] = calc_sma(df, 20)
    df["sma_50"] = calc_sma(df, 50)
    df["sma_200"] = calc_sma(df, 200)
    df["rsi"] = calc_rsi(df)

    macd = calc_macd(df)
    df["macd"] = macd["macd"]
    df["macd_signal"] = macd["signal"]
    df["macd_hist"] = macd["histogram"]

    bb = calc_bollinger(df)
    df["bb_upper"] = bb["bb_upper"]
    df["bb_middle"] = bb["bb_middle"]
    df["bb_lower"] = bb["bb_lower"]

    return df


# ---------------------------------------------------------------------------
# 3. Megjelenites
# ---------------------------------------------------------------------------
def plot_chart(df: pd.DataFrame, symbol: str) -> None:
    """4-paneles chart: arfolyam+BB+SMA, volume, RSI, MACD."""

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"{symbol} — Technikai elemzes", fontsize=16, fontweight="bold")

    gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.25)

    dates = df.index

    # --- Panel 1: Arfolyam + Bollinger + SMA --------------------------
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(dates, df["close"], linewidth=1.2, color="#1f77b4", label="Zaroar")
    ax1.plot(dates, df["sma_20"], linewidth=0.9, linestyle="--", label="SMA 20")
    ax1.plot(dates, df["sma_50"], linewidth=0.9, linestyle="--", label="SMA 50")
    ax1.plot(dates, df["sma_200"], linewidth=0.9, linestyle="--", label="SMA 200")
    ax1.fill_between(
        dates, df["bb_upper"], df["bb_lower"], alpha=0.12, color="gray", label="Bollinger"
    )
    ax1.plot(dates, df["bb_upper"], linewidth=0.5, color="gray")
    ax1.plot(dates, df["bb_lower"], linewidth=0.5, color="gray")
    ax1.set_ylabel("Arfolyam (USD)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Arfolyam + Bollinger-szalagok + SMA")

    # --- Panel 2: Volume -----------------------------------------------
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    if "volume" in df.columns and df["volume"].notna().any():
        colors = [
            "#26a69a" if c >= o else "#ef5350"
            for c, o in zip(df["close"], df["open"])
        ]
        ax2.bar(dates, df["volume"], color=colors, alpha=0.7, width=0.8)
    ax2.set_ylabel("Forgalom")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Forgalom")

    # --- Panel 3: RSI ---------------------------------------------------
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(dates, df["rsi"], linewidth=1, color="#ab47bc")
    ax3.axhline(70, linewidth=0.7, linestyle="--", color="red", alpha=0.6)
    ax3.axhline(30, linewidth=0.7, linestyle="--", color="green", alpha=0.6)
    ax3.fill_between(dates, 70, 100, alpha=0.06, color="red")
    ax3.fill_between(dates, 0, 30, alpha=0.06, color="green")
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("RSI (14)")
    ax3.grid(True, alpha=0.3)
    ax3.set_title("RSI (14)")

    # --- Panel 4: MACD --------------------------------------------------
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(dates, df["macd"], linewidth=1, color="#1f77b4", label="MACD")
    ax4.plot(dates, df["macd_signal"], linewidth=1, color="#ff7f0e", label="Szignal")
    hist_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df["macd_hist"]]
    ax4.bar(dates, df["macd_hist"], color=hist_colors, alpha=0.5, width=0.8)
    ax4.axhline(0, linewidth=0.5, color="black", alpha=0.3)
    ax4.set_ylabel("MACD")
    ax4.legend(loc="upper left", fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_title("MACD (12, 26, 9)")

    # Datum formatum
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax4.get_xticklabels(), rotation=45, ha="right")

    gs.tight_layout(fig, rect=[0, 0, 1, 0.96])
    plt.savefig(f"{symbol.replace('/', '-')}_technical_analysis.png", dpi=150, bbox_inches="tight")
    print(f"Chart elmentve: {symbol.replace('/', '-')}_technical_analysis.png")
    plt.show()


# ---------------------------------------------------------------------------
# 4. Osszefoglalo kiiras
# ---------------------------------------------------------------------------
def print_summary(df: pd.DataFrame, symbol: str) -> None:
    """Aktualis indikator-ertekek osszefoglaloja."""
    last = df.iloc[-1]
    prev = df.iloc[-2]

    print("\n" + "=" * 60)
    print(f"  {symbol} — Technikai osszefoglalo")
    print("=" * 60)
    print(f"  Datum:             {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Zaroar:            ${last['close']:,.2f}")
    print(f"  Valtozas (1 nap):  {((last['close'] / prev['close']) - 1) * 100:+.2f}%")
    print("-" * 60)
    print(f"  SMA  20:           ${last['sma_20']:,.2f}")
    print(f"  SMA  50:           ${last['sma_50']:,.2f}")
    print(f"  SMA 200:           ${last['sma_200']:,.2f}")
    print("-" * 60)
    print(f"  RSI (14):          {last['rsi']:.1f}", end="")
    if last["rsi"] > 70:
        print("  << TULVETT")
    elif last["rsi"] < 30:
        print("  << TULELADOTT")
    else:
        print("  (semleges)")
    print("-" * 60)
    print(f"  MACD:              {last['macd']:.2f}")
    print(f"  MACD szignal:      {last['macd_signal']:.2f}")
    macd_cross = "BULLISH" if last["macd"] > last["macd_signal"] else "BEARISH"
    print(f"  MACD helyzet:      {macd_cross}")
    print("-" * 60)
    print(f"  Bollinger felso:   ${last['bb_upper']:,.2f}")
    print(f"  Bollinger kozep:   ${last['bb_middle']:,.2f}")
    print(f"  Bollinger also:    ${last['bb_lower']:,.2f}")

    bb_pos = (last["close"] - last["bb_lower"]) / (last["bb_upper"] - last["bb_lower"])
    print(f"  BB %B:             {bb_pos:.2%}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# 5. Foprogram
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Crypto Technical Analyzer (OpenBB)")
    parser.add_argument("--symbol", default="BTC-USD", help="Kriptopar (pl. BTC-USD, ETH-USD)")
    parser.add_argument("--days", type=int, default=365, help="Visszatekintesi idoszak napokban")
    parser.add_argument("--provider", default="yfinance", help="Adatforras (yfinance, fmp, tiingo)")
    args = parser.parse_args()

    df = fetch_crypto_data(symbol=args.symbol, days=args.days, provider=args.provider)
    df = add_indicators(df)
    print_summary(df, args.symbol)
    plot_chart(df, args.symbol)


if __name__ == "__main__":
    main()
