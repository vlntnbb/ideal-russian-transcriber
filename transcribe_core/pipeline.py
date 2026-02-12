from __future__ import annotations

import json
import os
import shutil
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from snapscript.core.audio_processor import AudioProcessor, GigaAMTranscriptionService, TranscriptionService

from .env import opt_str_env, str_env, truthy_env
from .llm import LLMResult, llm_finalize
from .markdown import render_markdown


DEFAULT_AUDIO_NORMALIZE_FILTER = "adeclick,dynaudnorm=f=500:g=11:p=0.95:m=20"


def audio_normalize_filter() -> str:
    return (os.environ.get("AUDIO_NORMALIZE_FILTER") or DEFAULT_AUDIO_NORMALIZE_FILTER).strip()


def _segments_text(segments) -> str:
    parts = []
    for s in segments or []:
        t = getattr(s, "text", "")
        t = (t or "").strip()
        if t:
            parts.append(t)
    return " ".join(parts).strip()


@dataclass
class TranscribeArtifacts:
    original_path: str
    wav_path: str
    wav_norm_path: Optional[str]
    whisper_text_path: str
    whisper_json_path: str
    gigaam_text_path: str
    gigaam_json_path: str
    transcript_md_path: str
    result_json_path: str


def _wav_duration_sec(path: str) -> float:
    try:
        with wave.open(path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate() or 0
            if rate <= 0:
                return 0.0
            return float(frames) / float(rate)
    except Exception:
        return 0.0


@dataclass
class _ProgressState:
    audio_sec: float
    est_total_sec: float
    processed_sec: float = 0.0

    def update(self, *, processed_sec: float, elapsed_sec: float) -> None:
        self.processed_sec = max(self.processed_sec, float(processed_sec or 0.0))
        # Estimate total time from progress ratio (smooth-ish).
        if self.audio_sec > 0 and self.processed_sec > 0:
            progress = min(self.processed_sec / self.audio_sec, 0.995)
            raw = max(elapsed_sec / max(1e-3, progress), elapsed_sec)
            self.est_total_sec = max(elapsed_sec, self.est_total_sec * 0.7 + raw * 0.3)


def _write_progress(progress_path: Optional[Path], payload: dict) -> None:
    if not progress_path:
        return
    try:
        tmp = progress_path.with_suffix(progress_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, progress_path)
    except Exception:
        pass


def run_transcription_job(
    *,
    input_path: str,
    out_dir: str,
    normalize_audio: bool,
    copy_input: bool = True,
    dry_run: bool = False,
    progress_file: str = "progress.json",
) -> dict:
    """
    Runs end-to-end transcription and writes all artifacts into out_dir.
    Returns a JSON-serializable dict with stats and file paths.
    """
    t0 = time.monotonic()
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    progress_path = (outp / progress_file) if progress_file else None

    def set_progress(*, stage: str, label: str, percent: float, processed_sec: float = 0.0, audio_sec: float = 0.0, est_total_sec: float = 0.0) -> None:
        elapsed = time.monotonic() - t0
        payload = {
            "stage": stage,
            "label": label,
            "percent": max(0.0, min(1.0, float(percent or 0.0))),
            "elapsed_sec": round(float(elapsed), 3),
            "processed_sec": round(float(processed_sec or 0.0), 3),
            "audio_sec": round(float(audio_sec or 0.0), 3),
            "est_total_sec": round(float(est_total_sec or 0.0), 3),
            "updated_at": time.time(),
        }
        _write_progress(progress_path, payload)

    inp = Path(input_path)
    if not inp.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    set_progress(stage="init", label="Старт…", percent=0.0)

    original_name = f"original{inp.suffix or ''}"
    original_path = str((outp / original_name).resolve())
    if copy_input:
        shutil.copy2(str(inp), original_path)
    else:
        original_path = str(inp.resolve())

    timings: dict[str, float] = {}
    audio: dict[str, object] = {"original_path": os.path.basename(original_path)}
    models: dict[str, object] = {}
    errors: dict[str, Optional[str]] = {"whisper": None, "gigaam": None, "llm": None}

    whisper_model = str_env("WHISPER_MODEL", "medium")
    gigaam_model = str_env("GIGAAM_MODEL", "v3_e2e_rnnt")
    device = str_env("DEVICE", "cpu")
    language = str_env("LANGUAGE", "ru")
    hf_token = opt_str_env("HF_TOKEN")

    models.update({"whisper": whisper_model, "gigaam": gigaam_model, "device": device, "language": language, "hf_token": bool(hf_token)})

    ap = AudioProcessor()

    set_progress(stage="extract_wav", label="🎛 Готовлю WAV (16kHz mono)…", percent=0.05)
    ex0 = time.monotonic()
    wav_path = ap.extract_audio(original_path, output_dir=str(outp))
    timings["extract_wav_sec"] = round(time.monotonic() - ex0, 3)
    audio["wav_path"] = os.path.basename(wav_path)

    wav_norm_path: Optional[str] = None
    if normalize_audio:
        set_progress(stage="normalize", label="🎚 Нормализую аудио…", percent=0.12)
        nx0 = time.monotonic()
        wav_norm_path = ap.extract_audio(
            original_path,
            output_dir=str(outp),
            audio_filter=audio_normalize_filter(),
        )
        timings["normalize_sec"] = round(time.monotonic() - nx0, 3)
        audio["wav_norm_path"] = os.path.basename(wav_norm_path)

    wav_for_asr = wav_norm_path or wav_path

    whisper_text = ""
    gigaam_text = ""
    whisper_info = {}
    gigaam_info = {}
    audio_sec = _wav_duration_sec(wav_for_asr)

    if dry_run:
        set_progress(stage="gigaam", label="🧠 GigaAM — (dry-run)…", percent=0.25, audio_sec=audio_sec, est_total_sec=10.0)
        whisper_text = "[dry-run] whisper transcript"
        gigaam_text = "[dry-run] gigaam transcript"
        whisper_info = {"dry_run": True}
        gigaam_info = {"dry_run": True}
    else:
        est_total_sec = max(5.0, audio_sec * 2.5) if audio_sec > 0 else 60.0
        state = _ProgressState(audio_sec=audio_sec, est_total_sec=est_total_sec)
        last_write = 0.0

        set_progress(
            stage="gigaam",
            label="🧠 GigaAM — распознаю…",
            percent=0.15,
            audio_sec=audio_sec,
            est_total_sec=max(state.est_total_sec, time.monotonic() - t0 + audio_sec * 1.2),
        )
        g0 = time.monotonic()
        try:
            gs = GigaAMTranscriptionService(model_name=gigaam_model, device=device, hf_token=hf_token)
            g_segments, g_info = gs.transcribe(wav_for_asr)
            gigaam_info = g_info or {}
            gigaam_text = _segments_text(g_segments)
        except Exception as exc:
            errors["gigaam"] = str(exc)
        timings["gigaam_sec"] = round(time.monotonic() - g0, 3)

        def on_whisper_progress(processed: float) -> None:
            nonlocal last_write
            elapsed = time.monotonic() - t0
            state.update(processed_sec=float(processed or 0.0), elapsed_sec=elapsed)
            now = time.monotonic()
            if (now - last_write) < 1.5:
                return
            last_write = now
            pct = (state.processed_sec / state.audio_sec) if state.audio_sec > 0 else 0.0
            # Keep Whisper within 60%..90% segment of the full pipeline.
            overall = 0.60 + min(0.30, max(0.0, pct) * 0.30)
            set_progress(
                stage="whisper",
                label="🧠 Whisper — распознаю…",
                percent=overall,
                processed_sec=state.processed_sec,
                audio_sec=state.audio_sec,
                est_total_sec=state.est_total_sec,
            )

        set_progress(stage="whisper", label="🧠 Whisper — распознаю…", percent=0.60, audio_sec=audio_sec, est_total_sec=est_total_sec)
        w0 = time.monotonic()
        try:
            ws = TranscriptionService(model_size=whisper_model, device=device, language=language)
            w_segments, w_info = ws.transcribe(wav_for_asr, progress_cb=on_whisper_progress)
            whisper_info = w_info or {}
            whisper_text = _segments_text(w_segments)
        except Exception as exc:
            errors["whisper"] = str(exc)
        timings["whisper_sec"] = round(time.monotonic() - w0, 3)

    set_progress(stage="write_asr", label="💾 Сохраняю результаты ASR…", percent=0.75, audio_sec=audio_sec)
    whisper_text_path = str((outp / "whisper.txt").resolve())
    (outp / "whisper.txt").write_text(whisper_text, encoding="utf-8")
    whisper_json_path = str((outp / "whisper.json").resolve())
    (outp / "whisper.json").write_text(json.dumps(whisper_info, ensure_ascii=False, indent=2), encoding="utf-8")

    gigaam_text_path = str((outp / "gigaam.txt").resolve())
    (outp / "gigaam.txt").write_text(gigaam_text, encoding="utf-8")
    gigaam_json_path = str((outp / "gigaam.json").resolve())
    (outp / "gigaam.json").write_text(json.dumps(gigaam_info, ensure_ascii=False, indent=2), encoding="utf-8")

    user_prompt = f"GigaAM:\n{(gigaam_text or '').strip()}\n\nWhisper:\n{(whisper_text or '').strip()}\n"

    llm_result: Optional[LLMResult] = None
    final_text = ""
    final_error = ""
    final_label = "LLM"
    if dry_run:
        set_progress(stage="llm", label="🧠 LLM — (dry-run)…", percent=0.85, audio_sec=audio_sec)
        final_text = "### Итоговый текст:\n\n[dry-run] итог\n"
        llm_result = LLMResult(text=final_text, provider="dry-run", model="dry-run")
        final_label = "dry-run"
    else:
        if truthy_env("LLM_ENABLED", True):
            set_progress(stage="llm", label="🧠 LLM — формирую итог…", percent=0.82, audio_sec=audio_sec)
            llm0 = time.monotonic()
            try:
                llm_result = llm_finalize(prompt=user_prompt)
                final_text = llm_result.text
                final_label = f"{llm_result.provider} ({llm_result.model})"
            except Exception as exc:
                final_error = str(exc)
                errors["llm"] = final_error
            timings["llm_sec"] = round(time.monotonic() - llm0, 3)
        else:
            final_error = "LLM disabled (LLM_ENABLED=0)"

    set_progress(stage="write_markdown", label="📄 Формирую итоговый файл…", percent=0.92, audio_sec=audio_sec)
    transcript_md = render_markdown(
        final_text=final_text,
        final_error=final_error,
        final_label=final_label,
        whisper_text=whisper_text,
        gigaam_text=gigaam_text,
    )
    transcript_md_path = str((outp / "transcript.md").resolve())
    (outp / "transcript.md").write_text(transcript_md, encoding="utf-8")

    status = "ok" if not any([errors["whisper"], errors["gigaam"], errors["llm"]]) else "error"
    timings["total_sec"] = round(time.monotonic() - t0, 3)

    result = {
        "status": status,
        "input": {"path": str(inp.resolve())},
        "output_dir": str(outp.resolve()),
        "audio": audio,
        "normalize_audio": bool(normalize_audio),
        "models": models,
        "timings": timings,
        "errors": errors,
        "paths": {
            "original": os.path.basename(original_path),
            "wav": os.path.basename(wav_path),
            "wav_norm": os.path.basename(wav_norm_path) if wav_norm_path else None,
            "whisper_txt": "whisper.txt",
            "whisper_json": "whisper.json",
            "gigaam_txt": "gigaam.txt",
            "gigaam_json": "gigaam.json",
            "transcript_md": "transcript.md",
        },
        "llm": {
            "used": bool(llm_result and llm_result.text),
            "provider": llm_result.provider if llm_result else None,
            "model": llm_result.model if llm_result else None,
        },
        "results": {
            "whisper_len": len(whisper_text or ""),
            "gigaam_len": len(gigaam_text or ""),
            "llm_len": len(final_text or ""),
        },
    }

    result_json_path = str((outp / "result.json").resolve())
    (outp / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    set_progress(stage="done", label="✅ Готово", percent=1.0, audio_sec=audio_sec, est_total_sec=time.monotonic() - t0)

    _ = TranscribeArtifacts(
        original_path=original_path,
        wav_path=wav_path,
        wav_norm_path=wav_norm_path,
        whisper_text_path=whisper_text_path,
        whisper_json_path=whisper_json_path,
        gigaam_text_path=gigaam_text_path,
        gigaam_json_path=gigaam_json_path,
        transcript_md_path=transcript_md_path,
        result_json_path=result_json_path,
    )

    return result
