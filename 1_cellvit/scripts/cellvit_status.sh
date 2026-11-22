#!/bin/bash
# Queue, progress and failure summary for the CellViT jobs.
STAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$STAGE_DIR/../data}"
OUT="$STAGE_DIR/output/cellvit_out"
WSI_DIR="$DATA_ROOT/breast/wsis"
LOG_DIR="$STAGE_DIR/logs"

echo "===== QUEUE ====="
squeue -u "$USER" -o "%.12i %.10P %.8T %.10M %.20R"
echo "running: $(squeue -u $USER -h -t R | wc -l)   pending: $(squeue -u $USER -h -t PD | wc -l)"

echo ""; echo "===== PROGRESS ====="
echo "WSIs available: $(ls $WSI_DIR/*.tif 2>/dev/null | wc -l)"
echo "slides done   : $(ls $OUT/*/cells.geojson 2>/dev/null | wc -l)"

echo ""; echo "===== FAILURES ====="
grep -l -iE "error|traceback|failed|assert" "$LOG_DIR"/cellvit_*.log 2>/dev/null || echo "no errors found in logs"

echo ""; echo "===== LATEST LOG ====="
recent=$(ls -t "$LOG_DIR"/cellvit_*.log 2>/dev/null | head -1)
[ -n "$recent" ] && { echo "$recent"; tail -12 "$recent"; } || echo "no logs yet"

echo ""; echo "===== OUTPUT CHECK ====="
sample=$(ls $OUT/*/cells.geojson 2>/dev/null | head -1)
if [ -n "$sample" ]; then
  python - "$sample" <<'PY'
import json,sys
gj=json.load(open(sys.argv[1])); f=gj["features"] if isinstance(gj,dict) and "features" in gj else gj
n=sum(len(x["geometry"]["coordinates"]) if x.get("geometry",{}).get("type")=="MultiPolygon" else 1 for x in f)
print(f"{sys.argv[1].split('/')[-2]}: {n} cells")
PY
else echo "no output yet"; fi
