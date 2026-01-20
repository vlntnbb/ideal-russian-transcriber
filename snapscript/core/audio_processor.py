from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ASRSegment:
    start: float
    end: float
    text: str


class TranscriptionCancelled(Exception):
    pass


def _wav_duration_sec(wav_path: str) -> float:
    try:
        with wave.open(wav_path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate() or 0
            if rate <= 0:
                return 0.0
            return float(frames) / float(rate)
    except Exception:
        return 0.0


def _ffmpeg_wav_chunk(
    *,
    src_wav: str,
    dst_wav: str,
    start_sec: float,
    dur_sec: float,
    ffmpeg_bin: str = "ffmpeg",
) -> None:
    cmd = [
        ffmpeg_bin,
        "-y",
        "-ss",
        f"{max(0.0, float(start_sec)):.3f}",
        "-t",
        f"{max(0.0, float(dur_sec)):.3f}",
        "-i",
        src_wav,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        dst_wav,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg chunk failed: {proc.stderr.strip()}")


class AudioProcessor:
    def __init__(self, *, ffmpeg_bin: str = "ffmpeg") -> None:
        self.ffmpeg_bin = ffmpeg_bin
        self.logger = logging.getLogger("snapscript.AudioProcessor")

    def extract_audio(
        self,
        media_path: str,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        output_dir: Optional[str] = None,
    ) -> str:
        if not media_path or not os.path.exists(media_path):
            raise FileNotFoundError(f"Input media not found: {media_path}")

        output_dir = output_dir or tempfile.gettempdir()
        os.makedirs(output_dir, exist_ok=True)

        fd, out_path = tempfile.mkstemp(prefix="asr_", suffix=".wav", dir=output_dir)
        os.close(fd)

        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-i",
            media_path,
            "-vn",
            "-ac",
            str(int(channels)),
            "-ar",
            str(int(sample_rate)),
            "-f",
            "wav",
            out_path,
        ]
        self.logger.debug("Running ffmpeg: %s", " ".join(cmd))
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg не найден в PATH. Установите ffmpeg.") from exc

        if proc.returncode != 0:
            try:
                os.remove(out_path)
            except Exception:
                pass
            raise RuntimeError(
                "ffmpeg завершился с ошибкой.\n"
                f"cmd: {' '.join(cmd)}\n"
                f"stderr: {proc.stderr.strip()}"
            )

        return out_path


class TranscriptionService:
    def __init__(
        self,
        *,
        model_size: str = "medium",
        device: str = "cpu",
        language: str = "ru",
        compute_type: Optional[str] = None,
        cpu_threads: int = 0,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.language = language
        self.compute_type = compute_type or ("int8" if device == "cpu" else "float16")
        self.cpu_threads = int(cpu_threads or 0)
        self.logger = logging.getLogger("snapscript.Whisper")

    @staticmethod
    @lru_cache(maxsize=8)
    def _load_model(model_size: str, device: str, compute_type: str, cpu_threads: int):
        from faster_whisper import WhisperModel

        return WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=int(cpu_threads or 0),
        )

    def transcribe(
        self,
        wav_path: str,
        *,
        progress_cb=None,
        cancel_cb=None,
    ) -> Tuple[List[ASRSegment], Dict[str, object]]:
        self._progress_cb = progress_cb
        self._cancel_cb = cancel_cb
        try:
            return self._transcribe_impl(wav_path)
        finally:
            self._progress_cb = None
            self._cancel_cb = None

    def _transcribe_impl(self, wav_path: str) -> Tuple[List[ASRSegment], Dict[str, object]]:
        t0 = time.monotonic()
        model = self._load_model(self.model_size, self.device, self.compute_type, self.cpu_threads)
        vad_filter_env = (os.environ.get("WHISPER_VAD_FILTER") or "").strip().lower()
        vad_filter = vad_filter_env in {"1", "true", "yes", "y", "on"}
        # Speed/quality knobs (on CPU medium defaults are slow: beam_size=5).
        try:
            beam_size = int((os.environ.get("WHISPER_BEAM_SIZE") or "").strip() or "1")
        except Exception:
            beam_size = 1
        try:
            best_of = int((os.environ.get("WHISPER_BEST_OF") or "").strip() or "1")
        except Exception:
            best_of = 1

        self.logger.info(
            "Whisper transcribe start (model=%s device=%s compute=%s threads=%s beam=%s best_of=%s vad=%s)",
            self.model_size,
            self.device,
            self.compute_type,
            self.cpu_threads,
            beam_size,
            best_of,
            vad_filter,
        )
        segments_iter, info = model.transcribe(
            wav_path,
            language=self.language or None,
            # `vad_filter=True` pulls in `onnxruntime`. Keep it off by default for easier local setup.
            vad_filter=vad_filter,
            beam_size=max(1, beam_size),
            best_of=max(1, best_of),
        )
        segments: List[ASRSegment] = []
        for s in segments_iter:
            try:
                cancel = getattr(self, "_cancel_cb", None)
                if cancel and cancel():
                    self.logger.info("Whisper transcribe canceled by user")
                    raise TranscriptionCancelled()
            except TranscriptionCancelled:
                raise
            except Exception:
                pass
            seg = ASRSegment(start=float(s.start), end=float(s.end), text=str(s.text))
            segments.append(seg)
            # Progress callback (called from worker thread)
            try:
                cb = getattr(self, "_progress_cb", None)
                if cb:
                    cb(float(seg.end))
            except Exception:
                pass
        duration = 0.0
        try:
            duration = float(getattr(info, "duration", 0.0) or 0.0)
        except Exception:
            duration = 0.0

        if duration <= 0.0:
            duration = _wav_duration_sec(wav_path)

        self.logger.info(
            "Whisper transcribe done (segments=%d audio=%.2fs wall=%.2fs)",
            len(segments),
            duration,
            time.monotonic() - t0,
        )
        return segments, {"duration": duration, "language": getattr(info, "language", None)}


class GigaAMTranscriptionService:
    """
    Thin wrapper around https://pypi.org/project/gigaam/
    """

    AVAILABLE_MODELS = (
        # Upstream GigaAM repo models (support punctuation for `v3_e2e_*`):
        "v3_e2e_rnnt",
        "v3_e2e_ctc",
        "v3_rnnt",
        "v3_ctc",
        # PyPI `gigaam` (0.1.0+) models:
        "rnnt",
        "ctc",
        "v2_rnnt",
        "v2_ctc",
        "v1_rnnt",
        "v1_ctc",
    )

    _ALIASES = {
        "rnnt": "v2_rnnt",
        "ctc": "v2_ctc",
    }

    _FALLBACK_TO_V2 = {
        "v3_e2e_rnnt": "v2_rnnt",
        "v3_rnnt": "v2_rnnt",
        "v3_e2e_ctc": "v2_ctc",
        "v3_ctc": "v2_ctc",
    }

    def __init__(
        self,
        *,
        model_name: str = "v2_rnnt",
        device: str = "cpu",
        hf_token: Optional[str] = None,
        fp16_encoder: bool = True,
        use_flash: Optional[bool] = False,
        download_root: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.hf_token = hf_token
        self.fp16_encoder = fp16_encoder
        self.use_flash = use_flash
        self.download_root = download_root
        self.logger = logging.getLogger("snapscript.GigaAM")

        if hf_token:
            os.environ.setdefault("HF_TOKEN", hf_token)

    @classmethod
    def _normalize_model_name(cls, name: str) -> str:
        name = (name or "").strip()
        if not name:
            return "v2_rnnt"
        return cls._ALIASES.get(name, name)

    @staticmethod
    @lru_cache(maxsize=4)
    def _load_model(model_name: str, device: str, fp16_encoder: bool, use_flash: Optional[bool], download_root: Optional[str]):
        from gigaam import load_model

        return load_model(
            model_name=model_name,
            device=device,
            fp16_encoder=fp16_encoder,
            use_flash=use_flash,
            download_root=download_root,
        )

    def transcribe(self, wav_path: str) -> Tuple[List[ASRSegment], Dict[str, object]]:
        t0 = time.monotonic()
        duration = _wav_duration_sec(wav_path)
        requested_model = self._normalize_model_name(self.model_name)
        model_name = requested_model
        try:
            model = self._load_model(
                model_name,
                self.device,
                self.fp16_encoder,
                self.use_flash,
                self.download_root,
            )
        except Exception as exc:
            # If the environment has only the PyPI `gigaam` (v1/v2 models), but user requested v3,
            # fall back to v2 automatically (keeps bot working), and log how to enable punctuation.
            fallback_name = self._FALLBACK_TO_V2.get(requested_model)
            if fallback_name:
                self.logger.warning(
                    "GigaAM model '%s' is unavailable in current gigaam install (%s). Falling back to '%s' (no punctuation). "
                    "To enable punctuation, install upstream GigaAM (supports v3_e2e_*).",
                    requested_model,
                    exc,
                    fallback_name,
                )
                model_name = fallback_name
                model = self._load_model(
                    model_name,
                    self.device,
                    self.fp16_encoder,
                    self.use_flash,
                    self.download_root,
                )
            else:
                raise

        has_hf_token = bool((self.hf_token or os.environ.get("HF_TOKEN") or "").strip())
        self.logger.info(
            "GigaAM transcribe start (model=%s device=%s audio=%.2fs hf_token=%s)",
            model_name,
            self.device,
            duration,
            "yes" if has_hf_token else "no",
        )

        try:
            text = str(model.transcribe(wav_path) or "").strip()
            segments_short = [ASRSegment(start=0.0, end=duration, text=text)]
            self.logger.info(
                "GigaAM transcribe done (mode=short segments=%d wall=%.2fs)",
                len(segments_short),
                time.monotonic() - t0,
            )
            return segments_short, {"duration": duration, "model_name": model_name}
        except ValueError as exc:
            if "Too long wav file" not in str(exc):
                raise

        # Prefer long-form VAD-based mode when available (requires `pyannote.audio` + HF_TOKEN).
        if has_hf_token:
            self.logger.info("GigaAM longform enabled (trying transcribe_longform)")
            try:
                parts = model.transcribe_longform(wav_path)
                segments_lf: List[ASRSegment] = []
                texts_lf: List[str] = []
                for p in parts or []:
                    t = str((p or {}).get("transcription") or "").strip()
                    b = (p or {}).get("boundaries")
                    if t:
                        texts_lf.append(t)
                    if isinstance(b, (tuple, list)) and len(b) == 2:
                        try:
                            start = float(b[0])
                            end = float(b[1])
                        except Exception:
                            start, end = 0.0, 0.0
                        segments_lf.append(ASRSegment(start=start, end=end, text=t))
                if segments_lf:
                    self.logger.info(
                        "GigaAM transcribe done (mode=longform segments=%d wall=%.2fs)",
                        len(segments_lf),
                        time.monotonic() - t0,
                    )
                    return segments_lf, {"duration": duration, "model_name": model_name, "longform": True}
                full_text = " ".join(texts_lf).strip()
                self.logger.info(
                    "GigaAM transcribe done (mode=longform segments=1 wall=%.2fs)",
                    time.monotonic() - t0,
                )
                return [ASRSegment(start=0.0, end=duration, text=full_text)], {
                    "duration": duration,
                    "model_name": model_name,
                    "longform": True,
                }
            except Exception as exc:
                msg = str(exc)
                self.logger.warning("GigaAM longform failed: %s", msg)

                # `pyannote.audio` may return `None` from `Pipeline.from_pretrained` when token
                # is missing/invalid or the model is gated and terms weren't accepted.
                likely_auth = (
                    "Could not download" in msg
                    or "use_auth_token" in msg
                    or "401" in msg
                    or "403" in msg
                    or "gated" in msg.lower()
                    or "NoneType" in msg and ".to" in msg
                )
                if likely_auth:
                    raise RuntimeError(
                        "GigaAM long-form не запустился через pyannote VAD.\n"
                        "Проверьте, что:\n"
                        "- `HF_TOKEN` задан и валиден\n"
                        "- вы приняли условия на https://hf.co/pyannote/voice-activity-detection\n"
                        "- вы приняли условия на https://hf.co/pyannote/segmentation\n"
                        "- установлены зависимости: `python3 -m pip install -r requirements-longform.txt`\n"
                        "\n"
                        "Если хотите, чтобы бот автоматически резал на чанки при проблемах с long-form,\n"
                        "задайте `GIGAAM_FALLBACK_CHUNKING=1`."
                    ) from exc

                # Allow fallback only when explicitly enabled.
                fallback = (os.environ.get("GIGAAM_FALLBACK_CHUNKING") or "").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "y",
                    "on",
                }
                if not fallback:
                    raise
                self.logger.info("GigaAM falling back to chunking (GIGAAM_FALLBACK_CHUNKING=1)")

        # Fallback: chunk audio into short windows and run regular `transcribe` per chunk.
        # This avoids `pyannote.audio`/HF_TOKEN and works well for voice notes.
        chunk_sec_env = (os.environ.get("GIGAAM_CHUNK_SEC") or "").strip()
        try:
            chunk_sec = float(chunk_sec_env) if chunk_sec_env else 20.0
        except Exception:
            chunk_sec = 20.0
        chunk_sec = min(24.0, max(5.0, chunk_sec))
        self.logger.info("GigaAM chunking enabled (chunk_sec=%.1fs)", chunk_sec)

        segments: List[ASRSegment] = []
        texts: List[str] = []
        with tempfile.TemporaryDirectory(prefix="gigaam_chunks_") as td:
            t = 0.0
            idx = 0
            while t < max(duration, 0.001):
                d = min(chunk_sec, max(0.0, duration - t))
                if d <= 0.01:
                    break
                chunk_path = os.path.join(td, f"chunk_{idx:04d}.wav")
                _ffmpeg_wav_chunk(src_wav=wav_path, dst_wav=chunk_path, start_sec=t, dur_sec=d)
                part_text = str(model.transcribe(chunk_path) or "").strip()
                if part_text:
                    texts.append(part_text)
                segments.append(ASRSegment(start=t, end=min(duration, t + d), text=part_text))
                t += d
                idx += 1

        full_text = " ".join(texts).strip()
        if not full_text and segments:
            full_text = " ".join((s.text or "").strip() for s in segments).strip()

        self.logger.info(
            "GigaAM transcribe done (mode=chunking chunks=%d wall=%.2fs)",
            len(segments),
            time.monotonic() - t0,
        )
        return segments, {
            "duration": duration,
            "model_name": model_name,
            "chunking": True,
            "chunk_sec": chunk_sec,
        }


def gigaam_transcribe_worker(
    *,
    wav_path: str,
    model_name: str,
    device: str,
    hf_token: Optional[str],
    out_queue,
) -> None:
    """
    Helper for running GigaAM transcription in a separate process.
    `out_queue` should be a multiprocessing.Queue-like object.
    """
    try:
        svc = GigaAMTranscriptionService(model_name=model_name, device=device, hf_token=hf_token)
        segments, info = svc.transcribe(wav_path)
        out_queue.put(
            {
                "ok": True,
                "segments": [(float(s.start), float(s.end), str(s.text)) for s in (segments or [])],
                "info": info or {},
            }
        )
    except Exception as exc:
        out_queue.put(
            {
                "ok": False,
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            }
        )
