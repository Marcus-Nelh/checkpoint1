#!/usr/bin/env bash
# Download CelebA-Spoof from the official Google Drive folder linked in
# https://github.com/ZhangYuanhan-AI/CelebA-Spoof, reassemble the split
# archive, extract it, and write a checksum manifest for reproducibility.
#
# One-time setup:
#   1. Install rclone:  curl https://rclone.org/install.sh | sudo bash
#      (or: brew install rclone / conda install -c conda-forge rclone)
#   2. Run `rclone config`, create a remote named "gdrive", type "drive",
#      scope "drive.readonly", and sign in with your Google account.
#
# Usage:  bash download_celeba_spoof.sh [destination_dir]
#         ASSUME_YES=1 bash download_celeba_spoof.sh   # skip the size prompt
#
# WARNING: this is a very large download (tens of GB of archives; the
# script prints the exact size before starting) and needs roughly twice
# that in free disk space while extracting.
#
# Note: the dataset is for non-commercial research only and may not be
# redistributed. Keep the data directory out of git (add it to .gitignore).

set -euo pipefail

FOLDER_ID="1OW_1bawO79pRqdVEVmBzp8HSxdSwln_Z"   # official Google Drive folder
REMOTE="gdrive"
DEST="${1:-data/raw/celeba_spoof}"

mkdir -p "$DEST"

if ! command -v rclone >/dev/null 2>&1; then
  echo "rclone not found. See setup instructions at the top of this script." >&2
  exit 1
fi

echo "Files in the shared folder:"
rclone lsl "${REMOTE}:" --drive-root-folder-id "$FOLDER_ID"

# Size warning: report the total, estimate time at 100 Mbps, and ask to confirm.
total_bytes=$(rclone size "${REMOTE}:" --drive-root-folder-id "$FOLDER_ID" --json \
  | sed -n 's/.*"bytes":\([0-9]*\).*/\1/p')
total_gb=$(awk -v b="$total_bytes" 'BEGIN { printf "%.1f", b / 1e9 }')
est_hours=$(awk -v b="$total_bytes" 'BEGIN { printf "%.1f", b * 8 / 100e6 / 3600 }')
avail_gb=$(df -Pk "$DEST" | awk 'NR==2 { printf "%.1f", $4 * 1024 / 1e9 }')

cat <<EOF

=====================================================================
  WARNING: large download
  Download size:        ${total_gb} GB
  Est. time @ 100 Mbps: ~${est_hours} h at full speed (often longer)
  Free space at ${DEST}: ${avail_gb} GB
  Extraction needs roughly as much space again (~2x the download
  at peak) until the .zip parts are deleted.
=====================================================================

EOF

need_gb=$(awk -v t="$total_gb" 'BEGIN { printf "%.1f", t * 2 }')
if awk -v a="$avail_gb" -v n="$need_gb" 'BEGIN { exit !(a < n) }'; then
  echo "WARNING: free space (${avail_gb} GB) is below the ~${need_gb} GB needed." >&2
fi

if [[ "${ASSUME_YES:-0}" != "1" ]]; then
  read -r -p "Continue with the download? [y/N] " reply
  [[ "$reply" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
fi

echo "Downloading to $DEST (resumable: re-run this script if it stops) ..."
rclone copy "${REMOTE}:" "$DEST" \
  --drive-root-folder-id "$FOLDER_ID" \
  --transfers 4 --checkers 8 --retries 10 --low-level-retries 20 \
  --progress

cd "$DEST"

# Reassemble split archives (e.g. NAME.zip.001, NAME.zip.002, ...) and extract.
shopt -s nullglob
parts=( *.zip.[0-9][0-9][0-9] )
if (( ${#parts[@]} > 0 )); then
  base="${parts[0]%.*}"                       # NAME.zip
  if [[ ! -f "$base" ]]; then
    echo "Joining ${#parts[@]} parts into $base ..."
    cat "${base}".[0-9][0-9][0-9] > "$base"
  fi
  echo "Extracting $base ..."
  unzip -q -n "$base"
else
  for z in *.zip; do
    echo "Extracting $z ..."
    unzip -q -n "$z"
  done
fi

# Checksum manifest of the downloaded archives, to cite in the report.
echo "Writing checksums ..."
sha256sum ./*.zip* > SHA256SUMS.txt 2>/dev/null || shasum -a 256 ./*.zip* > SHA256SUMS.txt

echo "Done. Contents of $DEST:"
ls -lh
