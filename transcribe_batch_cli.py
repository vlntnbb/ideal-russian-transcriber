from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import tempfile
import traceback
import wave
from pathlib import Path
from typing import Optional

from snapscript.core.audio_processor import AudioProcessor
from transcribe_core.env import truthy_env
from transcribe_core.pipeline import run_transcription_job


def _sanitize_segment(name: str, *, fallback: str) -> str:
    raw = (name or "").strip()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")
    return safe or fallback


def _concat_wav_files(wav_paths: list[str], out_path: str) -> None:
    if len(wav_paths) < 2:
        raise ValueError("Need at least two WAV files to concatenate.")

    first_format = None
    with wave.open(out_path, "wb") as w_out:
        for p in wav_paths:
            with wave.open(p, "rb") as w_in:
                fmt = (
                    int(w_in.getnchannels()),
                    int(w_in.getsampwidth()),
                    int(w_in.getframerate()),
                    str(w_in.getcomptype()),
                    str(w_in.getcompname()),
                )
                if first_format is None:
                    first_format = fmt
                    w_out.setnchannels(fmt[0])
                    w_out.setsampwidth(fmt[1])
                    w_out.setframerate(fmt[2])
                elif fmt != first_format:
                    raise RuntimeError("WAV formats mismatch; cannot concatenate.")
                w_out.writeframes(w_in.readframes(w_in.getnframes()))


def _print_event(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _run_single_job(
    *,
    input_path: str,
    out_dir: str,
    normalize_audio: bool,
    copy_input: bool,
    dry_run: bool,
) -> dict:
    result = run_transcription_job(
        input_path=input_path,
        out_dir=out_dir,
        normalize_audio=normalize_audio,
        copy_input=copy_input,
        dry_run=dry_run,
        progress_file="progress.json",
    )
    return {
        "status": str(result.get("status") or "error"),
        "input_path": input_path,
        "output_dir": out_dir,
        "transcript_md": str(Path(out_dir) / "transcript.md"),
        "result_json": str(Path(out_dir) / "result.json"),
        "error": None,
    }


def _merge_to_wav(input_paths: list[Path], *, tmp_dir: str) -> str:
    ap = AudioProcessor()
    wav_parts: list[str] = []
    for idx, src in enumerate(input_paths, start=1):
        wav_path = ap.extract_audio(str(src), output_dir=tmp_dir)
        wav_parts.append(wav_path)
        _print_event({"event": "merge_part_ready", "index": idx, "total": len(input_paths), "source": str(src)})

    merged_wav = str(Path(tmp_dir) / "merged_input.wav")
    _concat_wav_files(wav_parts, merged_wav)
    return merged_wav


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch transcription CLI: process multiple files separately or merge into one before transcription."
    )
    parser.add_argument("--mode", choices=["separate", "merge"], required=True, help="Batch processing mode")
    parser.add_argument("--out", dest="out_dir", required=True, help="Root output directory for batch artifacts")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input audio/video file paths")
    parser.add_argument("--norm", dest="normalize_audio", action="store_true", help="Enable audio normalization")
    parser.add_argument(
        "--no-copy-input",
        dest="copy_input",
        action="store_false",
        help="Do not copy input into each output folder (use original path as-is)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Skip ASR/LLM and write placeholder outputs")
    parser.add_argument("--summary-file", default="summary.json", help="Summary file name inside --out (default: summary.json)")
    args = parser.parse_args(argv)

    dry_run = bool(args.dry_run or truthy_env("TRANSCRIBE_DRY_RUN", False))
    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    inputs: list[Path] = [Path(p).expanduser().resolve() for p in (args.inputs or [])]
    missing = [str(p) for p in inputs if not p.exists()]
    if missing:
        summary = {
            "status": "error",
            "mode": args.mode,
            "output_root": str(out_root),
            "results": [],
            "errors": [f"Input not found: {p}" for p in missing],
        }
        (out_root / args.summary_file).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False))
        return 2

    now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()
    summary: dict = {
        "status": "ok",
        "mode": args.mode,
        "started_at": now,
        "finished_at": None,
        "output_root": str(out_root),
        "normalize_audio": bool(args.normalize_audio),
        "dry_run": bool(dry_run),
        "inputs": [str(p) for p in inputs],
        "results": [],
        "errors": [],
    }

    try:
        if args.mode == "separate":
            for idx, src in enumerate(inputs, start=1):
                stem = _sanitize_segment(src.stem, fallback=f"file{idx:03d}")
                out_dir = out_root / f"{idx:03d}_{stem}"
                out_dir.mkdir(parents=True, exist_ok=True)
                _print_event({"event": "job_started", "index": idx, "total": len(inputs), "input": str(src)})
                try:
                    row = _run_single_job(
                        input_path=str(src),
                        out_dir=str(out_dir),
                        normalize_audio=bool(args.normalize_audio),
                        copy_input=bool(args.copy_input),
                        dry_run=dry_run,
                    )
                except Exception as exc:
                    row = {
                        "status": "error",
                        "input_path": str(src),
                        "output_dir": str(out_dir),
                        "transcript_md": None,
                        "result_json": None,
                        "error": str(exc).strip() or exc.__class__.__name__,
                    }
                summary["results"].append(row)
                _print_event({"event": "job_finished", "index": idx, "total": len(inputs), "status": row.get("status")})
        else:
            # merge
            merged_out = out_root / "merged"
            merged_out.mkdir(parents=True, exist_ok=True)
            _print_event({"event": "merge_started", "total": len(inputs)})
            with tempfile.TemporaryDirectory(prefix="batch_merge_") as td:
                merged_wav = _merge_to_wav(inputs, tmp_dir=td)
                _print_event({"event": "job_started", "index": 1, "total": 1, "input": "[merged batch]"})
                try:
                    row = _run_single_job(
                        input_path=str(merged_wav),
                        out_dir=str(merged_out),
                        normalize_audio=bool(args.normalize_audio),
                        copy_input=True,
                        dry_run=dry_run,
                    )
                    row["input_path"] = "[merged from inputs]"
                    row["source_inputs"] = [str(p) for p in inputs]
                except Exception as exc:
                    row = {
                        "status": "error",
                        "input_path": "[merged from inputs]",
                        "source_inputs": [str(p) for p in inputs],
                        "output_dir": str(merged_out),
                        "transcript_md": None,
                        "result_json": None,
                        "error": str(exc).strip() or exc.__class__.__name__,
                    }
                summary["results"].append(row)
                _print_event({"event": "job_finished", "index": 1, "total": 1, "status": row.get("status")})
    except Exception as exc:
        summary["errors"].append(str(exc).strip() or exc.__class__.__name__)
        summary["errors"].append(traceback.format_exc())

    has_errors = bool(summary["errors"]) or any(str((r or {}).get("status") or "") != "ok" for r in summary["results"])
    summary["status"] = "error" if has_errors else "ok"
    summary["finished_at"] = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()

    summary_path = out_root / args.summary_file
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
