#!/usr/bin/env bash
# commit-and-push.sh
#
# Stage, commit, and push to all remotes configured in commit-push.config.json.
#
# First-time setup:
#   cp commit-push.config.json.example commit-push.config.json
#   # edit commit-push.config.json with your username, token file path, and remotes
#
# Usage:
#   ./commit-and-push.sh                          # prompts for commit message
#   ./commit-and-push.sh -m "my commit message"   # inline message
#   ./commit-and-push.sh -f message.txt           # message from file
#   ./commit-and-push.sh --all                    # also stage untracked files
#   ./commit-and-push.sh --force                  # force push
#   ./commit-and-push.sh -m "msg" --all --force   # combine flags

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/commit-push.config.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

die() { echo "ERROR: $*" >&2; exit 1; }

read_json() {
    # read_json <file> <key>  — returns the string value for a top-level key
    python3 -c "
import json, sys
data = json.load(open('$1'))
val = data.get('$2')
if val is None:
    sys.exit(1)
if isinstance(val, list):
    print('\n'.join(val))
else:
    print(val)
" 2>/dev/null
}

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------

COMMIT_MSG=""
MSG_FILE=""
STAGE_ALL=false
FORCE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--message)   COMMIT_MSG="$2"; shift 2 ;;
        -f|--file)      MSG_FILE="$2";   shift 2 ;;
        --all)          STAGE_ALL=true;  shift   ;;
        --force)        FORCE=true;      shift   ;;
        -h|--help)
            sed -n '/^# Usage:/,/^[^#]/p' "$0" | head -n -1 | sed 's/^# //'
            exit 0 ;;
        *) die "Unknown option: $1. Use -h for help." ;;
    esac
done

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------

if [[ ! -f "$CONFIG" ]]; then
    echo "No config found at: $CONFIG"
    echo "Run:  cp commit-push.config.json.example commit-push.config.json"
    echo "Then edit it with your username, token file, and remotes."
    exit 1
fi

USERNAME="$(read_json "$CONFIG" username)"   || die "Missing 'username' in config."
TOKEN_FILE="$(read_json "$CONFIG" token_file)" || die "Missing 'token_file' in config."
REMOTES="$(read_json "$CONFIG" remotes)"     || die "Missing 'remotes' in config."

[[ -z "$USERNAME"  ]] && die "'username' is empty in config."
[[ -z "$TOKEN_FILE" ]] && die "'token_file' is empty in config."
[[ -z "$REMOTES"   ]] && die "'remotes' list is empty in config."

if [[ ! -f "$TOKEN_FILE" ]]; then
    echo "Token file not found: $TOKEN_FILE"
    printf "Enter your GitHub personal access token: "
    read -rs TOKEN
    echo
else
    TOKEN="$(cat "$TOKEN_FILE")"
fi

[[ -z "$TOKEN" ]] && die "Token is empty."

# ---------------------------------------------------------------------------
# Commit message
# ---------------------------------------------------------------------------

if [[ -n "$MSG_FILE" ]]; then
    [[ -f "$MSG_FILE" ]] || die "Message file not found: $MSG_FILE"
    COMMIT_MSG="$(cat "$MSG_FILE")"
fi

if [[ -z "$COMMIT_MSG" ]]; then
    echo "Enter commit message (blank line + Ctrl-D to finish):"
    COMMIT_MSG="$(cat)"
fi

[[ -z "$COMMIT_MSG" ]] && die "Commit message cannot be empty."

# ---------------------------------------------------------------------------
# Stage changes
# ---------------------------------------------------------------------------

cd "$SCRIPT_DIR"

echo
git status --short
echo

if $STAGE_ALL; then
    echo "Staging all changes (including untracked files) …"
    git add -A
else
    echo "Staging tracked modifications …"
    git add -u
fi

if git diff --cached --quiet; then
    echo "Nothing to commit."
    exit 0
fi

# ---------------------------------------------------------------------------
# Commit
# ---------------------------------------------------------------------------

git commit -m "$COMMIT_MSG"
echo

# ---------------------------------------------------------------------------
# Push to each remote
# ---------------------------------------------------------------------------

PUSH_FLAGS=""
$FORCE && PUSH_FLAGS="--force"

mapfile -t REMOTE_LIST <<< "$REMOTES"

for REMOTE_URL in "${REMOTE_LIST[@]}"; do
    # Inject credentials: https://user:token@github.com/...
    AUTH_URL="${REMOTE_URL/https:\/\//https://$USERNAME:$TOKEN@}"
    DISPLAY_URL="${REMOTE_URL}"   # never print the token

    echo "Pushing to $DISPLAY_URL …"
    git push $PUSH_FLAGS "$AUTH_URL" HEAD:main
    echo "  done."
done

echo
echo "All done."
