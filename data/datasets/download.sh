#!/usr/bin/env bash
set -e  # exit on error

# === Config ===
URL="https://ped.fz-juelich.de/experiments/2006.11.27_Duesseldorf_Casern/data/eg/2006bottleneck2_trajectories_txt.zip"
OUTDIR="data/datasets/julich_bottleneck_caserne/"
ZIPFILE="${OUTDIR}/$(basename $URL)"

# === Create output directory ===
mkdir -p "$OUTDIR"

# === Download file ===
echo "Downloading: $URL"
wget -q --show-progress -O "$ZIPFILE" "$URL"

# === Unzip file ===
echo "Unzipping to: $OUTDIR"
unzip -o "$ZIPFILE" -d "$OUTDIR"

# === Optional: remove ZIP after extraction ===
rm "$ZIPFILE"

echo "✅ Download and extraction complete: $OUTDIR"
