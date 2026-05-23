#!/usr/bin/env bash
# mount-data-to-modal-volume.sh
#
# Uploads m3_20m parquet files from a local folder to the cs224r Modal volume.
#
# Usage:
#   ./mount-data-to-modal-volume.sh /path/to/downloaded/outputs
#   ./mount-data-to-modal-volume.sh /path/to/downloaded/outputs --all
#   ./mount-data-to-modal-volume.sh /path/to/downloaded/outputs --volume my-volume-name
#
# Workflow for contributors:
#   1. Download data.zip from Google Shared Drive: CS 224R > inputs > data
#   2. Unzip it:  unzip data.zip -d ~/cs224r-data
#   3. Install modal:  pip install modal
#   4. Authenticate:   modal token new
#   5. Run this script: ./mount-data-to-modal-volume.sh ~/cs224r-data/outputs
#
# The data is then accessible inside Modal functions at /mnt/data/m3_20m/outputs/

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODAL=/home/ayamin/miniconda3/envs/openmm_env/bin/modal   # owner's modal path; overridden if modal is on PATH
VOLUME=cs224r-data
VOLUME_PREFIX=m3_20m/outputs   # path inside the volume → /mnt/data/m3_20m/outputs/ in functions

# Files needed for training only
TRAINING_FILES=(fragments.parquet parents.parquet)
# All files in the dataset
ALL_FILES=(fragments.parquet fragments_raw.parquet parents.parquet
           decompositions.parquet trajectories.parquet attach_demos.parquet)

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

UPLOAD_ALL=false
CUSTOM_VOLUME=""
SRC_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)      UPLOAD_ALL=true;      shift   ;;
        --volume)   CUSTOM_VOLUME="$2";   shift 2 ;;
        --modal)    MODAL="$2";           shift 2 ;;
        -h|--help)
            sed -n '2,/^set /p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        -*)
            echo "Unknown option: $1  (use -h for help)" >&2; exit 1 ;;
        *)
            if [[ -z "$SRC_DIR" ]]; then
                SRC_DIR="$1"; shift
            else
                echo "Unexpected argument: $1" >&2; exit 1
            fi ;;
    esac
done

if [[ -z "$SRC_DIR" ]]; then
    echo "Usage: $0 /path/to/outputs [--all] [--volume VOLUME_NAME]" >&2
    echo "  /path/to/outputs   folder containing the .parquet files" >&2
    exit 1
fi

[[ -n "$CUSTOM_VOLUME" ]] && VOLUME="$CUSTOM_VOLUME"

if $UPLOAD_ALL; then
    FILES=("${ALL_FILES[@]}")
else
    FILES=("${TRAINING_FILES[@]}")
fi

# ---------------------------------------------------------------------------
# Resolve modal binary (fall back to PATH)
# ---------------------------------------------------------------------------

if [[ ! -x "$MODAL" ]]; then
    if command -v modal &>/dev/null; then
        MODAL=modal
    else
        echo "ERROR: modal not found." >&2
        echo "Install:      pip install modal" >&2
        echo "Authenticate: modal token new" >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Check authentication
# ---------------------------------------------------------------------------

echo "Checking modal authentication ..."
if ! "$MODAL" profile current &>/dev/null 2>&1; then
    echo "Modal not authenticated.  Run:  $MODAL token new" >&2
    exit 1
fi
PROFILE=$("$MODAL" profile current 2>/dev/null || true)
echo "  authenticated as: ${PROFILE:-<unknown>}"
echo

# ---------------------------------------------------------------------------
# Upload each file
# ---------------------------------------------------------------------------

echo "=== Uploading to Modal volume '$VOLUME' from $SRC_DIR ==="
echo

UPLOADED=0
SKIPPED=0

for f in "${FILES[@]}"; do
    src="$SRC_DIR/$f"
    dst="$VOLUME_PREFIX/$f"

    if [[ ! -f "$src" ]]; then
        echo "  SKIP  $f  (not found at $src)"
        (( SKIPPED++ )) || true
        continue
    fi

    size=$(du -sh "$src" | cut -f1)
    echo "  [$size]  $f  →  $VOLUME:/$dst"
    "$MODAL" volume put --force "$VOLUME" "$src" "$dst"
    echo "    done."
    (( UPLOADED++ )) || true
done

echo
echo "Uploaded $UPLOADED file(s), skipped $SKIPPED."
echo
echo "Verify with:"
echo "  $MODAL volume ls $VOLUME $VOLUME_PREFIX"
