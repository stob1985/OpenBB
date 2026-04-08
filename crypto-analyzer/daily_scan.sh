#!/usr/bin/env bash
# ===========================================================================
# Crypto Daily Scanner
# ===========================================================================
# Futtatja a short es long scannereket, eredmenyt fajlba menti.
# Reszletes elemzes csak 70+ score-os jeloltekre.
#
# Hasznalat:
#   ./daily_scan.sh                  # alapertelmezett
#   ./daily_scan.sh --quick          # gyorsabb, kevesebb par
#   ./daily_scan.sh --deep           # alacsonyabb kuszobok
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M)
RESULTS_DIR="$SCRIPT_DIR/results"
OUTFILE="$RESULTS_DIR/${DATE}_scan.txt"

mkdir -p "$RESULTS_DIR"

# Parameterek
MIN_SHORT_SCORE=60
MIN_LONG_VOLUME=1000000
DETAIL_THRESHOLD=70
DAYS=180

case "${1:-}" in
    --quick)
        MIN_LONG_VOLUME=5000000
        DAYS=90
        ;;
    --deep)
        MIN_SHORT_SCORE=40
        MIN_LONG_VOLUME=500000
        DETAIL_THRESHOLD=60
        ;;
esac

# Szinek a terminalra (nem a fajlba)
GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${CYAN}=========================================${NC}"
echo -e "${CYAN} CRYPTO DAILY SCANNER - ${DATE} ${TIME}${NC}"
echo -e "${CYAN}=========================================${NC}"
echo ""

# Header a fajlban
{
    echo "============================================="
    echo " CRYPTO DAILY SCAN - ${DATE} ${TIME}"
    echo "============================================="
    echo ""
} > "$OUTFILE"

# ------------------------------------------------------------------
# 1. SHORT SCANNER
# ------------------------------------------------------------------
echo -e "${RED}[1/2] SHORT scanner indulas...${NC}"
echo "  Min score: $MIN_SHORT_SCORE | Min volume: \$$(printf "%'d" $MIN_LONG_VOLUME)"
echo ""

{
    echo "---------------------------------------------"
    echo " SHORT SCANNER"
    echo "---------------------------------------------"
    echo ""
} >> "$OUTFILE"

python3 crypto_analyzer.py \
    --scan-shorts \
    --min-short-score "$MIN_SHORT_SCORE" \
    --min-volume "$MIN_LONG_VOLUME" \
    --days "$DAYS" \
    2>&1 | tee -a "$OUTFILE"

echo "" >> "$OUTFILE"

# ------------------------------------------------------------------
# 2. LONG SCANNER
# ------------------------------------------------------------------
echo ""
echo -e "${GREEN}[2/2] LONG scanner indulas...${NC}"
echo "  Min volume: \$$(printf "%'d" $MIN_LONG_VOLUME) | Detail threshold: $DETAIL_THRESHOLD"
echo ""

{
    echo "---------------------------------------------"
    echo " LONG SCANNER"
    echo "---------------------------------------------"
    echo ""
} >> "$OUTFILE"

python3 crypto_analyzer.py \
    --scan-binance \
    --min-volume "$MIN_LONG_VOLUME" \
    --detail-threshold "$DETAIL_THRESHOLD" \
    --days "$DAYS" \
    2>&1 | tee -a "$OUTFILE"

echo "" >> "$OUTFILE"

# ------------------------------------------------------------------
# Osszefoglalo
# ------------------------------------------------------------------
SHORT_COUNT=$(grep -c "SHORT TERV:" "$OUTFILE" 2>/dev/null || echo 0)
LONG_DETAIL=$(grep -c "AKCIO TERV:" "$OUTFILE" 2>/dev/null || echo 0)

{
    echo ""
    echo "============================================="
    echo " SCAN VEGE: ${DATE} $(date +%H:%M)"
    echo "============================================="
    echo " Short jeloltek (reszletes): $SHORT_COUNT"
    echo " Long jeloltek (reszletes):  $LONG_DETAIL"
    echo "============================================="
} | tee -a "$OUTFILE"

echo ""
echo -e "${CYAN}Eredmeny mentve: ${YELLOW}${OUTFILE}${NC}"
echo -e "${CYAN}Meret: $(du -h "$OUTFILE" | cut -f1)${NC}"

# Ha nincs egyetlen 70+ jelolt sem
if [ "$SHORT_COUNT" -eq 0 ] && [ "$LONG_DETAIL" -eq 0 ]; then
    echo ""
    echo -e "${YELLOW}Nincs 70+ score-os jelolt ma. Nincs tennivalo.${NC}"
fi
