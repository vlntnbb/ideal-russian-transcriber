from __future__ import annotations

import json
import os
import subprocess
import threading
from dataclasses import dataclass
from typing import Optional

import httpx

from .env import float_env, int_env, opt_str_env, str_env, truthy_env


GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "deepseek-r1:8b"

_OLLAMA_PULL_LOCK = threading.Lock()


def get_system_prompt() -> tuple[str, str]:
    """
    Returns (prompt, source) where source is one of: file/env/default.
    Priority: GEMINI_SYSTEM_PROMPT_FILE -> GEMINI_SYSTEM_PROMPT -> gemini_system_prompt.md -> default constant.
    """
    default_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "gemini_system_prompt.md")

    def read_file(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    p = (os.environ.get("GEMINI_SYSTEM_PROMPT_FILE") or "").strip()
    candidate = p or default_path
    try:
        if candidate and os.path.exists(candidate):
            out = (read_file(candidate) or "").strip()
            if out:
                return out, f"file:{candidate}"
    except Exception:
        pass

    env_prompt = (os.environ.get("GEMINI_SYSTEM_PROMPT") or "").strip()
    if env_prompt:
        return env_prompt, "env"

    # Keep in sync (in spirit) with local_telegram_bot.DEFAULT_GEMINI_SYSTEM_PROMPT.
    fallback = (
        "вот две транскрибации одного аудио двумя разными моделями. они косячат в разных местах. "
        "твоя задача сделать итоговый текст, взяв из каждой модели пропущенные предложения и исправив явные ошибки "
        "и убрав повторы. когда какие-то слова будут отличаться выбирай тот вариант, который из контекста выглядит "
        "более правильным. если в обоих случаях текст выглядит как с ошибками - исправь исходя из контекста. "
        "ни в коем случае не выкидывай никакие смысловые фразы и предложения, тебе нужно соблюсти точность передачи текста. "
        "подсвети жирным те части текста, где тебе пришлось исправлять текст (где он отличается между моделями).\n\n"
        "обращай внимание на указания для контент-менеджера, что надо найти, исправить и уточнить по этому тексту если они есть. "
        "в конце отчитайся какие действия с текстами ты сделал\n\n"
        "вот такой формат выдачи (шаблон)\n\n"
        "### Итоговый текст:\n\n"
        "### Примечания для контент-менеджера:\n\n"
        "### Отчет о проделанных действиях:\n"
    ).strip()
    return fallback, "default"


def gemini_generate(*, api_key: str, model: str, prompt: str, system_prompt: Optional[str], generation_config: dict) -> str:
    url = f"{GEMINI_API_URL}/models/{model}:generateContent"
    params = {"key": api_key}
    payload = {
        "systemInstruction": {"role": "system", "parts": [{"text": system_prompt}]} if system_prompt else None,
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }
    payload = {k: v for k, v in payload.items() if v is not None}

    timeout = httpx.Timeout(connect=10.0, read=240.0, write=30.0, pool=30.0)
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, params=params, json=payload)
        r.raise_for_status()
        data = r.json()

    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini: empty response: {data!r}")
    content = (candidates[0] or {}).get("content") or {}
    parts = content.get("parts") or []
    texts = []
    for p in parts:
        t = (p or {}).get("text")
        if t:
            texts.append(str(t))
    out = "\n".join(texts).strip()
    if not out:
        raise RuntimeError("Gemini: empty text output")
    return out


def _ollama_base_url() -> str:
    return (os.environ.get("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL).strip().rstrip("/")


def _ollama_pick_best_local_model() -> str:
    base_url = _ollama_base_url()
    try:
        r = httpx.get(f"{base_url}/api/tags", timeout=10.0)
        r.raise_for_status()
        data = r.json() or {}
    except Exception:
        return ""

    models = data.get("models") or []
    if not isinstance(models, list):
        return ""

    def is_embedding(name: str) -> bool:
        n = (name or "").lower()
        return any(t in n for t in ("embedding", "embed", "text-embedding", "nomic-embed", "mxbai-embed", "bge-"))

    candidates = []
    for m in models:
        if not isinstance(m, dict):
            continue
        if (m.get("remote_host") or "").strip():
            continue
        name = (m.get("name") or m.get("model") or "").strip()
        if not name:
            continue
        if is_embedding(name):
            continue
        size = int(m.get("size") or 0)
        candidates.append((size, name))

    if not candidates:
        for m in models:
            if not isinstance(m, dict):
                continue
            if (m.get("remote_host") or "").strip():
                continue
            name = (m.get("name") or m.get("model") or "").strip()
            if not name:
                continue
            size = int(m.get("size") or 0)
            candidates.append((size, name))

    if not candidates:
        return ""
    candidates.sort()
    return candidates[-1][1]


def ollama_model() -> str:
    raw = (os.environ.get("OLLAMA_MODEL") or "").strip()
    if not raw or raw.lower() == "auto":
        picked = _ollama_pick_best_local_model()
        return picked or DEFAULT_OLLAMA_MODEL
    return raw


def _ollama_options_from_env() -> dict:
    perf = (os.environ.get("OLLAMA_PERF_PROFILE") or "auto").strip().lower()
    cpu_count = int(os.cpu_count() or 0)

    options = {
        "temperature": float_env("OLLAMA_TEMPERATURE", 0.7),
        "top_p": float_env("OLLAMA_TOP_P", 0.95),
        "num_ctx": int_env("OLLAMA_NUM_CTX", 8192),
    }

    raw_threads = (os.environ.get("OLLAMA_NUM_THREADS") or os.environ.get("OLLAMA_NUM_THREAD") or "").strip()
    raw_batch = (os.environ.get("OLLAMA_NUM_BATCH") or "").strip()
    if raw_threads.isdigit():
        options["num_thread"] = int(raw_threads)
    if raw_batch.isdigit():
        options["num_batch"] = int(raw_batch)

    if "num_thread" not in options:
        if perf in {"max", "aggressive", "fast"}:
            if cpu_count:
                options["num_thread"] = max(4, cpu_count - 2)
        elif perf == "auto":
            if cpu_count >= 12:
                options["num_thread"] = max(4, cpu_count - 2)

    if "num_batch" not in options:
        if perf in {"max", "aggressive", "fast"}:
            options["num_batch"] = 512
        elif perf == "auto":
            if cpu_count >= 12:
                options["num_batch"] = 256

    return options


def _ensure_ollama_model_present(model: str) -> None:
    base_url = _ollama_base_url()
    try:
        r = httpx.get(f"{base_url}/api/tags", timeout=10.0)
        r.raise_for_status()
        data = r.json() or {}
        models = data.get("models") or []
        if any((m or {}).get("name") == model for m in models):
            return
    except Exception:
        return

    if not truthy_env("OLLAMA_AUTO_PULL", True):
        return

    with _OLLAMA_PULL_LOCK:
        try:
            r = httpx.get(f"{base_url}/api/tags", timeout=10.0)
            r.raise_for_status()
            data = r.json() or {}
            models = data.get("models") or []
            if any((m or {}).get("name") == model for m in models):
                return
        except Exception:
            return
        try:
            subprocess.run(["ollama", "pull", model], check=False)
        except Exception:
            pass


def ollama_chat(*, model: str, prompt: str, system_prompt: Optional[str]) -> str:
    _ensure_ollama_model_present(model)
    url = f"{_ollama_base_url()}/api/chat"
    options = _ollama_options_from_env()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt or ""},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": options,
    }
    timeout = httpx.Timeout(connect=5.0, read=600.0, write=30.0, pool=30.0)
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        data = r.json() or {}
    msg = (data or {}).get("message") or {}
    out = str(msg.get("content") or "").strip()
    if not out:
        raise RuntimeError("Ollama: empty output")
    return out


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    error: Optional[str] = None


def llm_finalize(
    *,
    prompt: str,
    prefer_gemini: bool = True,
    prefer_ollama: bool = True,
) -> LLMResult:
    system_prompt, _src = get_system_prompt()

    gemini_api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    gemini_model = (os.environ.get("GEMINI_MODEL") or "gemini-3-pro-preview").strip()
    if prefer_gemini and gemini_api_key:
        generation_config = {
            "temperature": float_env("GEMINI_TEMPERATURE", 1.0),
            "topP": float_env("GEMINI_TOP_P", 0.95),
            "maxOutputTokens": int_env("GEMINI_MAX_OUTPUT_TOKENS", 65536),
        }
        try:
            out = gemini_generate(
                api_key=gemini_api_key,
                model=gemini_model,
                prompt=prompt,
                system_prompt=system_prompt,
                generation_config=generation_config,
            )
            return LLMResult(text=out, provider="gemini", model=gemini_model)
        except Exception as exc:
            if not prefer_ollama:
                raise
            gem_err = str(exc)
    else:
        gem_err = None

    if prefer_ollama and truthy_env("OLLAMA_ENABLED", True):
        model = ollama_model()
        try:
            out = ollama_chat(model=model, prompt=prompt, system_prompt=system_prompt)
            return LLMResult(text=out, provider="ollama", model=model, error=gem_err)
        except Exception as exc:
            raise RuntimeError(f"LLM failed (gemini_err={gem_err!r}): {exc}") from exc

    raise RuntimeError(f"No LLM available (gemini_err={gem_err!r})")

