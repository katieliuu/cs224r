#!/usr/bin/env bash
# mount-data-to-modal-volume.sh
#
# Uploads parquet files from data.zip directly to a Modal volume.
# Extracts one file at a time so you never need the full unzipped size on disk.
#
# Usage:
#   ./mount-data-to-modal-volume.sh /path/to/data.zip
#   ./mount-data-to-modal-volume.sh /path/to/data.zip --conda-env myenv
#   ./mount-data-to-modal-volume.sh /path/to/data.zip --all
#   ./mount-data-to-modal-volume.sh /path/to/data.zip --volume my-volume-name
#
# Contributor workflow:
#   1. Download data.zip from Google Shared Drive: CS 224R > inputs > data
#   2. pip install modal  (or conda install -c conda-forge modal-client)
#   3. modal token new
#   4. ./mount-data-to-modal-volume.sh ~/Downloads/data.zip --conda-env myenv

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (owner defaults — contributors override via flags)
# ---------------------------------------------------------------------------

VOLUME=cs224r-data
VOLUME_PREFIX=m3_20m/outputs       # files land at /mnt/data/m3_20m/outputs/ in Modal functions
TMP_DIR=$(mktemp -d /tmp/cs224r-upload-XXXXXX)

# Only the two files the training scripts actually need:
TRAINING_FILES=(fragments.parquet parents.parquet)
# Full dataset:
ALL_FILES=(fragments.parquet fragments_raw.parquet parents.parquet
           decompositions.parquet trajectories.parquet attach_demos.parquet)

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

ZIP_PATH=""
CONDA_ENV=""
MODAL_BIN=""
UPLOAD_ALL=false
CUSTOM_VOLUME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --conda-env)   CONDA_ENV="$2";    shift 2 ;;
        --modal)       MODAL_BIN="$2";    shift 2 ;;
        --volume)      CUSTOM_VOLUME="$2"; shift 2 ;;
        --all)         UPLOAD_ALL=true;   shift   ;;
        -h|--help)
            sed -n '2,/^set /p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        -*)
            echo "Unknown option: $1  (use -h for help)" >&2; exit 1 ;;
        *)
            [[ -z "$ZIP_PATH" ]] && { ZIP_PATH="$1"; shift; } || { echo "Unexpected: $1" >&2; exit 1; } ;;
    esac
done

[[ -n "$CUSTOM_VOLUME" ]] && VOLUME="$CUSTOM_VOLUME"

if [[ -z "$ZIP_PATH" ]]; then
    echo "Usage: $0 /path/to/data.zip [--conda-env ENV] [--all]" >&2
    exit 1
fi

[[ -f "$ZIP_PATH" ]] || { echo "ERROR: zip not found: $ZIP_PATH" >&2; exit 1; }

if $UPLOAD_ALL; then
    FILES=("${ALL_FILES[@]}")
else
    FILES=("${TRAINING_FILES[@]}")
fi

# ---------------------------------------------------------------------------
# Resolve modal binary
# ---------------------------------------------------------------------------

find_modal() {
    # 1. Explicit --modal path
    if [[ -n "$MODAL_BIN" ]]; then
        [[ -x "$MODAL_BIN" ]] || { echo "ERROR: modal not executable: $MODAL_BIN" >&2; exit 1; }
        echo "$MODAL_BIN"; return
    fi

    # 2. From named conda env
    if [[ -n "$CONDA_ENV" ]]; then
        # Try conda info --base to find envs root
        local conda_base
        conda_base=$(conda info --base 2>/dev/null) || conda_base="$HOME/miniconda3"
        local bin="$conda_base/envs/$CONDA_ENV/bin/modal"
        if [[ -x "$bin" ]]; then
            echo "$bin"; return
        fi
        # Might be in a non-standard envs dir; try `conda run` as fallback
        if conda run -n "$CONDA_ENV" modal --version &>/dev/null 2>&1; then
            # Return a wrapper that uses conda run
            echo "conda_run:$CONDA_ENV"; return
        fi
        echo "ERROR: modal not found in conda env '$CONDA_ENV'." >&2
        echo "  Install with:  conda activate $CONDA_ENV && pip install modal" >&2
        exit 1
    fi

    # 3. modal on PATH
    if command -v modal &>/dev/null; then
        echo "$(command -v modal)"; return
    fi

    # 4. Owner's machine default
    local owner_bin="/home/ayamin/miniconda3/envs/openmm_env/bin/modal"
    if [[ -x "$owner_bin" ]]; then
        echo "$owner_bin"; return
    fi

    echo "ERROR: modal not found." >&2
    echo "  Install:      pip install modal" >&2
    echo "  Authenticate: modal token new" >&2
    exit 1
}

MODAL_RAW=$(find_modal)

# Wrap invocations so conda_run: prefix is handled transparently
modal_cmd() {
    if [[ "$MODAL_RAW" == conda_run:* ]]; then
        local env="${MODAL_RAW#conda_run:}"
        conda run -n "$env" modal "$@"
    else
        "$MODAL_RAW" "$@"
    fi
}

# ---------------------------------------------------------------------------
# Check authentication
# ---------------------------------------------------------------------------

echo "Checking modal authentication ..."
if ! modal_cmd profile current &>/dev/null 2>&1; then
    echo "Not authenticated.  Run:  modal token new" >&2
    exit 1
fi
PROFILE=$(modal_cmd profile current 2>/dev/null || true)
echo "  authenticated as: ${PROFILE:-<unknown>}"
echo

# ---------------------------------------------------------------------------
# Inspect zip: find the path prefix for the parquet files
# ---------------------------------------------------------------------------

echo "Inspecting $ZIP_PATH ..."
# Find one known file to determine its path inside the zip
SAMPLE=$(unzip -l "$ZIP_PATH" | awk '{print $NF}' | grep 'fragments\.parquet$' | head -1)
if [[ -z "$SAMPLE" ]]; then
    echo "ERROR: fragments.parquet not found in zip. Check the zip contents with: unzip -l $ZIP_PATH" >&2
    exit 1
fi
# Strip the filename to get the directory prefix inside the zip (may be empty)
ZIP_PREFIX="${SAMPLE%fragments.parquet}"
echo "  zip internal prefix: '${ZIP_PREFIX:-(none)}'"
echo

# ---------------------------------------------------------------------------
# Upload: extract → upload → delete, one file at a time
# ---------------------------------------------------------------------------

cleanup() { rm -rf "$TMP_DIR"; }
trap cleanup EXIT

echo "=== Uploading to Modal volume '$VOLUME' (one file at a time) ==="
echo "    Temp dir: $TMP_DIR"
echo

UPLOADED=0
SKIPPED=0

for f in "${FILES[@]}"; do
    zip_entry="${ZIP_PREFIX}${f}"
    tmp_file="$TMP_DIR/$f"
    dst="$VOLUME_PREFIX/$f"

    # Check the file exists in the zip
    if ! unzip -l "$ZIP_PATH" "$zip_entry" &>/dev/null 2>&1; then
        echo "  SKIP  $f  (not found in zip as '$zip_entry')"
        (( SKIPPED++ )) || true
        continue
    fi

    echo "  Extracting $f ..."
    unzip -p "$ZIP_PATH" "$zip_entry" > "$tmp_file"
    size=$(du -sh "$tmp_file" | cut -f1)

    echo "  [$size]  Uploading  $f  →  $VOLUME:/$dst"
    modal_cmd volume put --force "$VOLUME" "$tmp_file" "$dst"

    rm -f "$tmp_file"
    echo "    done."
    (( UPLOADED++ )) || true
done

echo
echo "Uploaded $UPLOADED file(s), skipped $SKIPPED."
echo
echo "Verify with:  modal volume ls $VOLUME $VOLUME_PREFIX"
