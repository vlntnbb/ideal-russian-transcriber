from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from transcribe_core.env import truthy_env
from transcribe_core.pipeline import run_transcription_job


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Ideal Russian Transcriber CLI (file -> transcripts + markdown + JSON)")
    parser.add_argument("--in", dest="input_path", required=True, help="Path to input audio/video file")
    parser.add_argument("--out", dest="out_dir", required=True, help="Directory for outputs")
    parser.add_argument("--norm", dest="normalize_audio", action="store_true", help="Enable audio normalization")
    parser.add_argument(
        "--no-copy-input",
        dest="copy_input",
        action="store_false",
        help="Do not copy input into out_dir (use original path as-is)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Skip ASR/LLM; write placeholder outputs (for testing)")
    args = parser.parse_args(argv)

    dry_run = bool(args.dry_run or truthy_env("TRANSCRIBE_DRY_RUN", False))
    out_dir = str(Path(args.out_dir).expanduser())
    os.makedirs(out_dir, exist_ok=True)

    result = run_transcription_job(
        input_path=str(Path(args.input_path).expanduser()),
        out_dir=out_dir,
        normalize_audio=bool(args.normalize_audio),
        copy_input=bool(args.copy_input),
        dry_run=dry_run,
        progress_file="progress.json",
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
