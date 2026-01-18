"""Local minimal implementation used by dual_transcriber_gui and the Telegram bot.

This repo originally referenced an external `snapscript` package. For portability,
we ship the small subset of functionality needed for:
- audio preparation via ffmpeg
- Whisper transcription via faster-whisper
- GigaAM transcription via the `gigaam` package

Compatibility:
Some upstream libs still pass `use_auth_token=` into `huggingface_hub.hf_hub_download`,
but newer `huggingface_hub` versions renamed it to `token=`.
We provide a small shim so GigaAM long-form (pyannote) can work without pinning.
"""

from __future__ import annotations


def _patch_huggingface_hub_use_auth_token() -> None:
    try:
        import inspect
        import huggingface_hub
    except Exception:
        return

    try:
        hf_hub_download = getattr(huggingface_hub, "hf_hub_download", None)
        if hf_hub_download is None:
            return
        params = inspect.signature(hf_hub_download).parameters
        if "use_auth_token" in params:
            return  # already supported

        def _wrapped_hf_hub_download(*args, **kwargs):  # type: ignore[no-redef]
            if "use_auth_token" in kwargs and "token" not in kwargs:
                kwargs["token"] = kwargs.pop("use_auth_token")
            else:
                kwargs.pop("use_auth_token", None)
            return hf_hub_download(*args, **kwargs)

        huggingface_hub.hf_hub_download = _wrapped_hf_hub_download  # type: ignore[assignment]
    except Exception:
        return


_patch_huggingface_hub_use_auth_token()

