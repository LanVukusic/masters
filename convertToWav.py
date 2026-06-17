#!/usr/bin/env python3
"""
Recursively convert all .mp3 files in a folder to .wav (in‑place).
Requires ffmpeg in PATH.

Usage:
    python convert_mp3_to_wav.py /path/to/top/folder [--remove-original] [--dry-run]

Options:
    --remove-original   Delete the source .mp3 after successful conversion.
    --dry-run           Only print what would be done, no actual conversion.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm


def convert_mp3_to_wav(mp3_path: Path, remove_original: bool = False, dry_run: bool = False) -> bool:
    """Convert a single .mp3 file to .wav in the same directory."""
    wav_path = mp3_path.with_suffix(".wav")
    
    # Skip if WAV already exists (unless you want to force overwrite)
    if wav_path.exists():
        print(f"Skipping (already exists): {wav_path}")
        return True

    if dry_run:
        print(f"[DRY RUN] Would convert: {mp3_path} -> {wav_path}")
        if remove_original:
            print(f"[DRY RUN] Would delete: {mp3_path}")
        return True

    print(f"Converting: {mp3_path}")
    try:
        # ffmpeg command: -y overwrites output without asking, -hide_banner reduces output
        cmd = [
            "ffmpeg", "-y", "-hide_banner",
            "-i", str(mp3_path),
            "-q:a", "0",          # best WAV quality (PCM 16-bit)
            str(wav_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            print(f"  ERROR: ffmpeg failed.\n{result.stderr}")
            return False

        # Optionally delete original mp3
        if remove_original:
            print(f"  Deleting original: {mp3_path}")
            mp3_path.unlink()

        return True

    except Exception as e:
        print(f"  EXCEPTION: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert all .mp3 files to .wav recursively.")
    parser.add_argument("folder", help="Top folder to search")
    parser.add_argument("--remove-original", action="store_true",
                        help="Delete the source .mp3 after successful conversion")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without actually converting")
    args = parser.parse_args()

    top = Path(args.folder).resolve()
    if not top.is_dir():
        print(f"Error: '{top}' is not a valid directory.")
        sys.exit(1)

    mp3_files = list(top.rglob("*.mp3"))
    if not mp3_files:
        print("No .mp3 files found.")
        return

    print(f"Found {len(mp3_files)} .mp3 file(s).\n")

    success = 0
    fail = 0
    for mp3 in tqdm(mp3_files):
        if convert_mp3_to_wav(mp3, args.remove_original, args.dry_run):
            success += 1
        else:
            fail += 1

    print(f"\nDone. Success: {success}, Failed: {fail}")

if __name__ == "__main__":
    main()
