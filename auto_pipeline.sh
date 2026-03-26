#!/usr/bin/env bash
# auto_pipeline.sh — runs after clf_edge training completes
# Monitors clf_edge.keras mtime, then chains:
#   Step 1: make evaluate      (re-evaluate clf_top with fresh outputs)
#   Step 2: python src/evaluate_pipeline.py  (full Q1→Q2→Q3 chain report)
#
# Run in background: nohup bash auto_pipeline.sh &

set -euo pipefail
cd "$(dirname "$0")"

EDGE_MODEL="src/models/clf_edge.keras"
LOG="outputs/auto_pipeline.log"
PYTHON=".venv/bin/python"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

mkdir -p outputs

# ── Step 0: wait for clf_edge training to finish ──────────────────────────────
log "Watching $EDGE_MODEL for training completion..."
LAST_MTIME=$(stat -c %Y "$EDGE_MODEL" 2>/dev/null || echo 0)

while true; do
    sleep 30
    # Training is done when no python train process is running for this project
    if ! pgrep -f "train_hierarchical" > /dev/null 2>&1; then
        NEW_MTIME=$(stat -c %Y "$EDGE_MODEL" 2>/dev/null || echo 0)
        if [ "$NEW_MTIME" -gt "$LAST_MTIME" ]; then
            log "clf_edge.keras updated (mtime changed) and train process gone — proceeding"
        else
            log "train_hierarchical not running — proceeding (model may not have updated if early-stopped to same weights)"
        fi
        break
    fi
done

# ── Step 1: evaluate clf_top (fresh report + Grad-CAM) ────────────────────────
log "=== Step 1: make evaluate ==="
make evaluate >> "$LOG" 2>&1 && log "Step 1 complete" || { log "Step 1 FAILED"; exit 1; }

# ── Step 2: full pipeline evaluator ───────────────────────────────────────────
log "=== Step 2: full Q1->Q2->Q3 pipeline report ==="
$PYTHON src/evaluate_pipeline.py >> "$LOG" 2>&1 && log "Step 2 complete" || { log "Step 2 FAILED"; exit 1; }

log "=== auto_pipeline complete ==="
