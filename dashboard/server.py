from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import parse_qs, urlparse


ROOT_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().with_name("static")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def _parse_iso_dt(value: str) -> datetime:
    # usage_sessions.jsonl uses RFC3339-ish with timezone offset (+00:00)
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _floor_day(dt: datetime) -> date:
    return dt.date()


def _floor_hour(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if q <= 0:
        return float(sorted_vals[0])
    if q >= 1:
        return float(sorted_vals[-1])
    idx = (len(sorted_vals) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                items.append(obj)
    return items


def _user_label(user: dict[str, Any]) -> str:
    username = (user.get("username") or "").strip()
    first_name = (user.get("first_name") or "").strip()
    last_name = (user.get("last_name") or "").strip()
    name = " ".join([p for p in [first_name, last_name] if p]).strip()
    if username and name:
        return f"{name} (@{username})"
    if username:
        return f"@{username}"
    if name:
        return name
    uid = user.get("id")
    return f"user:{uid}" if uid is not None else "unknown"


def _chat_label(chat: dict[str, Any]) -> str:
    title = (chat.get("title") or "").strip()
    ctype = (chat.get("type") or "").strip()
    cid = chat.get("id")
    if title and ctype:
        return f"{title} ({ctype})"
    if title:
        return title
    if ctype and cid is not None:
        return f"{ctype}:{cid}"
    return str(cid) if cid is not None else "unknown"


def _coalesce_dt(
    started_at: datetime,
    ended_at: Optional[datetime],
    total_sec: Optional[float],
) -> datetime:
    if ended_at is not None:
        return ended_at
    if total_sec is not None and total_sec > 0:
        return started_at + timedelta(seconds=total_sec)
    return started_at


def _compute_peak_concurrency(windows: Iterable[tuple[datetime, datetime]]) -> tuple[int, Optional[datetime]]:
    events: list[tuple[datetime, int]] = []
    for start, end in windows:
        if end < start:
            start, end = end, start
        events.append((start, +1))
        events.append((end, -1))
    if not events:
        return 0, None
    # At the same timestamp, process end before start to avoid artificial spikes.
    events.sort(key=lambda x: (x[0], x[1]))

    active = 0
    peak = 0
    peak_at: Optional[datetime] = None
    for t, delta in events:
        active += delta
        if active > peak:
            peak = active
            peak_at = t
    return peak, peak_at


def _compute_analytics(
    sessions: list[dict[str, Any]],
    *,
    now: Optional[datetime] = None,
    days: Optional[int] = None,
) -> dict[str, Any]:
    if now is None:
        now = datetime.now(timezone.utc)
    since: Optional[datetime] = None
    if days is not None and days > 0:
        since = now - timedelta(days=days)

    filtered: list[dict[str, Any]] = []
    for s in sessions:
        try:
            started_at = _parse_iso_dt(str(s.get("started_at") or ""))
        except Exception:
            continue
        if since is not None and started_at < since:
            continue
        filtered.append(s)

    status_counts: dict[str, int] = {"ok": 0, "canceled": 0, "error": 0, "other": 0}
    triggers: dict[str, int] = {}
    chat_types: dict[str, int] = {}
    model_counts: dict[str, dict[str, int]] = {"whisper": {}, "gigaam": {}, "gemini": {}, "device": {}, "language": {}}
    users: dict[int, dict[str, Any]] = {}
    chats: dict[int, dict[str, Any]] = {}

    windows: list[tuple[datetime, datetime]] = []
    per_hour: dict[datetime, int] = {}
    per_day: dict[date, dict[str, int]] = {}

    internal_users: set[int] = set()
    external_users: set[int] = set()
    unknown_users: set[int] = set()
    internal_sessions = 0
    external_sessions = 0
    unknown_sessions = 0

    audio_secs: list[float] = []
    total_secs_ok: list[float] = []
    stage_secs_ok: dict[str, list[float]] = {
        "download_sec": [],
        "extract_wav_sec": [],
        "whisper_sec": [],
        "gigaam_sec": [],
        "gemini_sec": [],
        "total_sec": [],
    }

    for s in filtered:
        started_at = _parse_iso_dt(str(s.get("started_at")))
        ended_at = _parse_iso_dt(str(s.get("ended_at"))) if s.get("ended_at") else None

        timings = s.get("timings") if isinstance(s.get("timings"), dict) else {}
        total_sec = _safe_float((timings or {}).get("total_sec"))
        ended_at = _coalesce_dt(started_at, ended_at, total_sec)

        windows.append((started_at, ended_at))
        per_hour[_floor_hour(started_at)] = per_hour.get(_floor_hour(started_at), 0) + 1

        day = _floor_day(started_at)
        day_bucket = per_day.setdefault(day, {"total": 0, "ok": 0, "canceled": 0, "error": 0, "other": 0})
        day_bucket["total"] += 1

        status = str(s.get("status") or "other")
        if status in status_counts:
            status_counts[status] += 1
        else:
            status_counts["other"] += 1
        day_bucket[status if status in day_bucket else "other"] += 1

        trigger = str(s.get("trigger") or "unknown")
        triggers[trigger] = triggers.get(trigger, 0) + 1

        telegram = s.get("telegram") if isinstance(s.get("telegram"), dict) else {}
        user = (telegram or {}).get("user") if isinstance((telegram or {}).get("user"), dict) else {}
        chat = (telegram or {}).get("chat") if isinstance((telegram or {}).get("chat"), dict) else {}

        uid = _safe_int(user.get("id"))
        if uid is not None:
            u = users.setdefault(
                uid,
                {
                    "user_id": uid,
                    "label": _user_label(user),
                    "username": user.get("username"),
                    "first_name": user.get("first_name"),
                    "last_name": user.get("last_name"),
                    "language_code": user.get("language_code"),
                    "sessions_total": 0,
                    "sessions_ok": 0,
                    "sessions_canceled": 0,
                    "sessions_error": 0,
                    "first_seen_at": started_at,
                    "last_seen_at": started_at,
                },
            )
            u["sessions_total"] += 1
            if status in ("ok", "canceled", "error"):
                u[f"sessions_{status}"] += 1
            u["first_seen_at"] = min(u["first_seen_at"], started_at)
            u["last_seen_at"] = max(u["last_seen_at"], started_at)

        cid = _safe_int(chat.get("id"))
        if cid is not None:
            c = chats.setdefault(
                cid,
                {
                    "chat_id": cid,
                    "label": _chat_label(chat),
                    "type": chat.get("type"),
                    "title": chat.get("title"),
                    "sessions_total": 0,
                    "first_seen_at": started_at,
                    "last_seen_at": started_at,
                },
            )
            c["sessions_total"] += 1
            c["first_seen_at"] = min(c["first_seen_at"], started_at)
            c["last_seen_at"] = max(c["last_seen_at"], started_at)

        ctype = str(chat.get("type") or "unknown")
        chat_types[ctype] = chat_types.get(ctype, 0) + 1

        models = s.get("models") if isinstance(s.get("models"), dict) else {}
        for key in ("whisper", "gigaam", "gemini", "device", "language"):
            val = models.get(key)
            if val is None:
                continue
            sval = str(val)
            model_counts[key][sval] = model_counts[key].get(sval, 0) + 1

        auth = s.get("auth") if isinstance(s.get("auth"), dict) else {}
        if uid is not None:
            if "user_authorized" in (auth or {}):
                if bool((auth or {}).get("user_authorized")):
                    internal_users.add(uid)
                    internal_sessions += 1
                else:
                    external_users.add(uid)
                    external_sessions += 1
            else:
                unknown_users.add(uid)
                unknown_sessions += 1

        audio = s.get("audio") if isinstance(s.get("audio"), dict) else {}
        wav_sec = _safe_float((audio or {}).get("wav_sec"))
        if wav_sec is not None and wav_sec > 0:
            audio_secs.append(wav_sec)

        if status == "ok":
            if total_sec is not None and total_sec > 0:
                total_secs_ok.append(total_sec)
            for stage in stage_secs_ok.keys():
                sec = _safe_float((timings or {}).get(stage))
                if sec is not None and sec > 0:
                    stage_secs_ok[stage].append(sec)

    peak_concurrency, peak_concurrency_at = _compute_peak_concurrency(windows)
    if per_hour:
        peak_per_hour_at = max(per_hour, key=lambda k: per_hour[k])
        peak_per_hour = per_hour[peak_per_hour_at]
    else:
        peak_per_hour_at = None
        peak_per_hour = 0

    audio_secs.sort()
    total_secs_ok.sort()
    stage_stats: dict[str, Any] = {}
    for stage, vals in stage_secs_ok.items():
        vals.sort()
        stage_stats[stage] = {
            "avg": (sum(vals) / len(vals)) if vals else 0.0,
            "p50": _percentile(vals, 0.5),
            "p95": _percentile(vals, 0.95),
            "count": len(vals),
        }

    internal_users_count = len(internal_users)
    # A user may have both unknown + internal/external sessions; keep internal precedence.
    unknown_users_only = unknown_users - internal_users - external_users
    external_users_only = external_users - internal_users
    internal_users_count = len(internal_users)
    external_users_count = len(external_users_only)
    unknown_users_count = len(unknown_users_only)

    def _iso(dt: Optional[datetime]) -> Optional[str]:
        return dt.astimezone(timezone.utc).isoformat() if dt else None

    daily = [
        {
            "date": d.isoformat(),
            **per_day[d],
        }
        for d in sorted(per_day.keys())
    ]
    hourly = [
        {
            "hour": h.astimezone(timezone.utc).isoformat(),
            "count": per_hour[h],
        }
        for h in sorted(per_hour.keys())
    ]

    top_users = sorted(users.values(), key=lambda u: (-u["sessions_total"], u["label"]))[:20]
    top_chats = sorted(chats.values(), key=lambda c: (-c["sessions_total"], c["label"]))[:20]

    for u in top_users:
        u["first_seen_at"] = _iso(u["first_seen_at"])
        u["last_seen_at"] = _iso(u["last_seen_at"])
    for c in top_chats:
        c["first_seen_at"] = _iso(c["first_seen_at"])
        c["last_seen_at"] = _iso(c["last_seen_at"])

    return {
        "generated_at": now.astimezone(timezone.utc).isoformat(),
        "filter": {"days": days},
        "summary": {
            "sessions_total": len(filtered),
            "sessions_ok": status_counts["ok"],
            "sessions_canceled": status_counts["canceled"],
            "sessions_error": status_counts["error"],
            "unique_users": len(users),
            "unique_chats": len(chats),
            "internal_users": internal_users_count,
            "external_users": external_users_count,
            "unknown_users": unknown_users_count,
            "internal_sessions": internal_sessions,
            "external_sessions": external_sessions,
            "unknown_sessions": unknown_sessions,
            "peak_concurrency": peak_concurrency,
            "peak_concurrency_at": _iso(peak_concurrency_at),
            "peak_sessions_per_hour": peak_per_hour,
            "peak_sessions_per_hour_at": _iso(peak_per_hour_at),
        },
        "breakdowns": {
            "status": status_counts,
            "triggers": dict(sorted(triggers.items(), key=lambda kv: (-kv[1], kv[0]))),
            "chat_types": dict(sorted(chat_types.items(), key=lambda kv: (-kv[1], kv[0]))),
            "models": {k: dict(sorted(v.items(), key=lambda kv: (-kv[1], kv[0]))) for k, v in model_counts.items()},
        },
        "timings_ok": {
            "audio_sec": {
                "avg": (sum(audio_secs) / len(audio_secs)) if audio_secs else 0.0,
                "p50": _percentile(audio_secs, 0.5),
                "p95": _percentile(audio_secs, 0.95),
                "count": len(audio_secs),
            },
            "total_sec": {
                "avg": (sum(total_secs_ok) / len(total_secs_ok)) if total_secs_ok else 0.0,
                "p50": _percentile(total_secs_ok, 0.5),
                "p95": _percentile(total_secs_ok, 0.95),
                "count": len(total_secs_ok),
            },
            "stages": stage_stats,
        },
        "timeseries": {"daily": daily, "hourly": hourly},
        "top": {"users": top_users, "chats": top_chats},
    }


@dataclass
class _Cache:
    mtime_ns: int
    sessions: list[dict[str, Any]]


class DashboardServer:
    def __init__(self, *, usage_log_path: Path) -> None:
        self.usage_log_path = usage_log_path
        self._cache: Optional[_Cache] = None

    def load_sessions(self) -> list[dict[str, Any]]:
        try:
            st = self.usage_log_path.stat()
        except FileNotFoundError:
            self._cache = _Cache(mtime_ns=0, sessions=[])
            return []
        if self._cache and self._cache.mtime_ns == st.st_mtime_ns:
            return self._cache.sessions
        sessions = _read_jsonl(self.usage_log_path)
        self._cache = _Cache(mtime_ns=st.st_mtime_ns, sessions=sessions)
        return sessions


def _guess_content_type(path: Path) -> str:
    ctype, _ = mimetypes.guess_type(path.name)
    return ctype or "application/octet-stream"


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")


def make_handler(server: DashboardServer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            # Reduce noise; keep errors visible.
            sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

        def _send(self, *, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: Any, *, status: int = 200) -> None:
            self._send(status=status, body=_json_bytes(payload), content_type="application/json; charset=utf-8")

        def _send_text(self, text: str, *, status: int = 200) -> None:
            self._send(status=status, body=text.encode("utf-8"), content_type="text/plain; charset=utf-8")

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path or "/"

            if path == "/api/analytics":
                qs = parse_qs(parsed.query or "")
                days = None
                if "days" in qs:
                    try:
                        days = int(qs["days"][0])
                    except Exception:
                        days = None
                sessions = server.load_sessions()
                payload = _compute_analytics(sessions, days=days)
                self._send_json(payload)
                return

            if path == "/api/health":
                self._send_json({"ok": True})
                return

            # Static routing
            if path == "/":
                path = "/index.html"
            rel = path.lstrip("/")
            fs_path = (STATIC_DIR / rel).resolve()
            if not str(fs_path).startswith(str(STATIC_DIR.resolve())):
                self._send_text("Not found", status=HTTPStatus.NOT_FOUND)
                return
            if not fs_path.exists() or not fs_path.is_file():
                self._send_text("Not found", status=HTTPStatus.NOT_FOUND)
                return

            body = fs_path.read_bytes()
            self._send(status=HTTPStatus.OK, body=body, content_type=_guess_content_type(fs_path))

    return Handler


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Local analytics dashboard for Ideal Russian Transcriber usage_sessions.jsonl")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--usage-log",
        default=(os.environ.get("USAGE_LOG_PATH") or "").strip() or str(ROOT_DIR / "usage_sessions.jsonl"),
        help="Path to usage_sessions.jsonl (defaults to USAGE_LOG_PATH or ./usage_sessions.jsonl)",
    )
    args = parser.parse_args(argv)

    usage_log_path = Path(args.usage_log).expanduser()
    srv = DashboardServer(usage_log_path=usage_log_path)

    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(srv))
    print(f"Dashboard: http://{args.host}:{args.port}")
    print(f"Usage log: {usage_log_path}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
