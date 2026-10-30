#!/usr/bin/env bash
# Download CelebA-Spoof from the official Google Drive folder with rclone and extract it.
# Needs an rclone remote named "gdrive" (see README). The download is ~80 GB and needs
# about twice that in free disk space while extracting.
#
# Usage: bash scripts/fetch_data.sh [destination_dir]

set -euo pipefail

FOLDER_ID="1OW_1bawO79pRqdVEVmBzp8HSxdSwln_Z"
DEST="${1:-data/raw/celeba_spoof}"
mkdir -p "$DEST"

if ! command -v rclone >/dev/null 2>&1; then
  echo "rclone not found, see README." >&2
  exit 1
fi

rclone size "gdrive:" --drive-root-folder-id "$FOLDER_ID"
read -r -p "Continue with the download? [y/N] " reply
[[ "$reply" =~ ^[Yy]$ ]] || exit 0

# re-run the script to resume an interrupted download
rclone copy "gdrive:" "$DEST" --drive-root-folder-id "$FOLDER_ID" --transfers 4 --retries 10 --progress

cd "$DEST"
shopt -s nullglob
parts=( *.zip.[0-9][0-9][0-9] )
if (( ${#parts[@]} > 0 )); then
  archive="${parts[0]%.*}"
  [[ -f "$archive" ]] || cat "$archive".[0-9][0-9][0-9] > "$archive"
  unzip -q -n "$archive"
else
  for archive in *.zip; do
    unzip -q -n "$archive"
  done
fi

sha256sum ./*.zip* > SHA256SUMS.txt
echo "Done: $DEST"
