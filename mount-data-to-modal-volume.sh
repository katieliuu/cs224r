#!/usr/bin/env bash
# mount-data-to-modal-volume.sh
#
# Uploads m3_20m parquet files to a Modal volume so all contributors
# can access the data in training runs.
#
# Two modes:
#   --local  : upload directly from a local path (default if LOCAL_SRC exists)
#   --gdrive : download from Google Drive via rclone, then upload to Modal
#
# Usage:
#   ./mount-data-to-modal-volume.sh                          # auto-detect mode
#   ./mount-data-to-modal-volume.sh --local                  # force local upload
#   ./mount-data-to-modal-volume.sh --gdrive                 # force rclone download
#   ./mount-data-to-modal-volume.sh --local  --src /path/to/outputs
#   ./mount-data-to-modal-volume.sh --gdrive --gdrive-path "gdrive:cs224r-data/m3_20m/outputs"
#   ./mount-data-to-modal-volume.sh --all                    # upload all parquets, not just training ones
#
# First-time setup for contributors (Google Drive mode):
#   1.  Install rclone:      sudo apt install rclone  OR  brew install rclone
#   2.  Configure gdrive:    rclone config   (add a remote named "gdrive")
#   3.  Install modal:       pip install modal
#   4.  Authenticate modal:  modal token new
#   5.  Run this script with --gdrive

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults — edit these to match your setup
# ---------------------------------------------------------------------------

MODAL=/home/ayamin/miniconda3/envs/openmm_env/bin/modal   # path to modal binary
VOLUME=cs224r-data                                          # Modal volume name
VOLUME_PREFIX=m3_20m/outputs                                # path inside the volume
                                                            # → mounts as /mnt/data/m3_20m/outputs/

LOCAL_SRC=/mnt/data/m3_20m/outputs                         # local data path (owner's machine)
GDRIVE_PATH="gdrive:cs224r-data/m3_20m/outputs"            # rclone remote path
TMP_DIR=/tmp/cs224r-parquets                                # temp dir for rclone downloads

# Files needed for training (fragments + parents).  Use --all to include the rest.
TRAINING_FILES=(fragments.parquet parents.parquet)
ALL_FILES=(fragments.parquet fragments_raw.parquet parents.parquet
           decompositions.parquet trajectories.parquet attach_demos.parquet)

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------

MODE=auto    # auto | local | gdrive
UPLOAD_ALL=false
CUSTOM_SRC=""
CUSTOM_GDRIVE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --local)         MODE=local;  shift ;;
        --gdrive)        MODE=gdrive; shift ;;
        --all)           UPLOAD_ALL=true; shift ;;
        --src)           CUSTOM_SRC="$2"; shift 2 ;;
        --gdrive-path)   CUSTOM_GDRIVE="$2"; shift 2 ;;
        --volume)        VOLUME="$2"; shift 2 ;;
        --modal)         MODAL="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^set /p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown option: $1  (use -h for help)" >&2; exit 1 ;;
    esac
done

[[ -n "$CUSTOM_SRC"    ]] && LOCAL_SRC="$CUSTOM_SRC"
[[ -n "$CUSTOM_GDRIVE" ]] && GDRIVE_PATH="$CUSTOM_GDRIVE"

if $UPLOAD_ALL; then
    FILES=("${ALL_FILES[@]}")
else
    FILES=("${TRAINING_FILES[@]}")
fi

# ---------------------------------------------------------------------------
# Resolve auto mode
# ---------------------------------------------------------------------------

if [[ "$MODE" == "auto" ]]; then
    missing=false
    for f in "${FILES[@]}"; do
        [[ -f "$LOCAL_SRC/$f" ]] || { missing=true; break; }
    done
    if $missing; then
        echo "Local source $LOCAL_SRC is missing some files → using gdrive mode."
        MODE=gdrive
    else
        echo "Local source $LOCAL_SRC found → using local mode."
        MODE=local
    fi
fi

# ---------------------------------------------------------------------------
# Verify modal is available
# ---------------------------------------------------------------------------

if [[ ! -x "$MODAL" ]]; then
    # fall back to modal on PATH
    if command -v modal &>/dev/null; then
        MODAL=modal
    else
        echo "ERROR: modal not found at $MODAL and not on PATH." >&2
        echo "Install with:  pip install modal   then run:  modal token new" >&2
        exit 1
    fi
fi

# Check modal can reach the server (catches missing/expired tokens)
echo "Checking modal authentication ..."
if ! "$MODAL" profile current &>/dev/null 2>&1; then
    echo "Modal not authenticated.  Run:  $MODAL token new" >&2
    exit 1
fi
PROFILE=$("$MODAL" profile current 2>/dev/null || true)
echo "  authenticated as: ${PROFILE:-<unknown>}"

# ---------------------------------------------------------------------------
# Local mode: upload directly from LOCAL_SRC
# ---------------------------------------------------------------------------

if [[ "$MODE" == "local" ]]; then
    echo
    echo "=== Uploading from $LOCAL_SRC → Modal volume '$VOLUME' ==="
    echo

    for f in "${FILES[@]}"; do
        src="$LOCAL_SRC/$f"
        dst="$VOLUME_PREFIX/$f"

        if [[ ! -f "$src" ]]; then
            echo "  SKIP  $f  (not found at $src)"
            continue
        fi

        size=$(du -sh "$src" | cut -f1)
        echo "  [$size]  $f  →  $VOLUME/$dst"
        "$MODAL" volume put "$VOLUME" "$src" "$dst"
        echo "    done."
    done

    echo
    echo "All done. Verify with:"
    echo "  $MODAL volume ls $VOLUME $VOLUME_PREFIX"
    exit 0
fi

# ---------------------------------------------------------------------------
# Google Drive mode: rclone download → upload to Modal
# ---------------------------------------------------------------------------

if ! command -v rclone &>/dev/null; then
    echo "ERROR: rclone not found.  Install with:  sudo apt install rclone" >&2
    exit 1
fi

# Extract remote name and check it's configured
RCLONE_REMOTE="${GDRIVE_PATH%%:*}"
if ! rclone listremotes | grep -q "^${RCLONE_REMOTE}:"; then
    echo "ERROR: rclone remote '${RCLONE_REMOTE}' not configured." >&2
    echo "Run:  rclone config   and add a remote named '${RCLONE_REMOTE}'" >&2
    exit 1
fi

echo
echo "=== Downloading from $GDRIVE_PATH → $TMP_DIR ==="
echo

mkdir -p "$TMP_DIR"

for f in "${FILES[@]}"; do
    remote_file="$GDRIVE_PATH/$f"
    local_file="$TMP_DIR/$f"

    echo "  Downloading $f ..."
    rclone copy --progress "$remote_file" "$TMP_DIR/"
    echo "    downloaded."
done

echo
echo "=== Uploading $TMP_DIR → Modal volume '$VOLUME' ==="
echo

for f in "${FILES[@]}"; do
    src="$TMP_DIR/$f"
    dst="$VOLUME_PREFIX/$f"

    if [[ ! -f "$src" ]]; then
        echo "  SKIP  $f  (download may have failed)"
        continue
    fi

    size=$(du -sh "$src" | cut -f1)
    echo "  [$size]  $f  →  $VOLUME/$dst"
    "$MODAL" volume put "$VOLUME" "$src" "$dst"
    echo "    done."
done

echo
echo "Cleaning up $TMP_DIR ..."
rm -rf "$TMP_DIR"

echo
echo "All done. Verify with:"
echo "  $MODAL volume ls $VOLUME $VOLUME_PREFIX"
