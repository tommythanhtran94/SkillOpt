"""SkillOpt-Sleep — DeepSeek Harness (dsh) session harvesting.

dsh keeps one directory per session under ``~/.dsh/sessions``. The directory
name encodes the session's working directory, and the transcript is
``session.v4.jsonl.zstd``: an append-only v4 JSONL log whose batches are
concatenated Zstandard frames. Sessions written with compression disabled use
``session.v4.jsonl`` instead, and both layouts are read here.

The v4 shapes below were verified against real logs on a dsh host, not
inferred:

  * ``session`` — exactly one, first: ``version`` (4), ``id``, ``createdAt``
    (epoch milliseconds), ``cwd``, ``isSeeded``, ``delegationDepth`` and
    ``agentPreset``.
  * ``user/message`` — ``data.content`` blocks plus ``data.source``
    (``{kind, form}``). ``kind == "user"`` is typed/pasted input;
    ``kind == "dsh-superagent"`` with ``form == "relay"`` is the harness
    relaying a human request (a candidate card for the router, or the
    assignment card for a worker). The host also injects machine context on
    this channel — ``runtime-context``, ``skill-catalog``, ``time-context``
    and ``tool-jobs`` — and those never become user prompts.
  * ``assistant/message`` — ``data.message`` with ``role == "assistant"``;
    ``content`` blocks are ``text`` (what the user sees), ``reasoning``
    (private chain of thought) and ``tool-call``.
  * ``tool/call`` — ``data.name`` plus ``data.arguments`` (a JSON string);
    ``tool/result`` — ``data.message`` with ``isError`` and the tool output.
  * ``turn/start`` / ``step/start`` / ``system/message`` / ``developer/message``
    / ``job/registered`` / ``job/settled`` / ``request/header`` and the
    remaining bookkeeping records carry no user intent and are skipped.

What is kept: visible user text, visible assistant text, tool names, skill
targets, the count of mechanically failed tool results, and deployment
provenance (session id, agent preset, delegation depth, store path). Private
reasoning, system/developer prompts, tool arguments, tool output and file
payloads are never copied into a digest, so ``files_touched`` stays empty
(v4 logs expose no file list without reading tool arguments).

Data handling
-------------
The adapter only reads: it never writes to the session store and makes no
network call. Before harvested text can reach a backend or a staged proposal
it goes through the shared secret redactor plus a bounded pass that removes
e-mail addresses, credentials embedded in URL userinfo and well-known
credential directories. That pass is defence in depth, not a guarantee to
detect arbitrary personal data; the derived ``project`` path stays unredacted
because scoping needs it.

Harvesting is opt-in: ``allowed_projects`` restricts the walk to the project
paths a caller explicitly lists, even under ``scope="all"``. An empty
allowlist leaves the existing ``scope`` decision in place and logs a warning
when ``scope="all"`` would otherwise take every project in the store. Only
this store is read, so sessions written by other tools are never merged in.

Skill targets come from the ``skill`` tool call, whose arguments are
``{"name": "<skill>"}``; only that key is read, never the rest of the call.

Session kinds and why both are harvested
----------------------------------------
dsh writes router/orchestrator sessions (``agentPreset`` ending in
``-router``) and worker/subagent sessions (``-worker``, often with a positive
``delegationDepth``), plus ordinary interactive ``session-*`` sessions. Other
harvesters drop sub-agent transcripts because another tool's sub-agents are
implementation detail. In dsh the worker session is where the actual coding
work, tool use and verification happen, so excluding it would throw away the
richest learning source. This harvester therefore keeps every session kind
and only applies the shared relevance filters: project scope, the harvest
cutoff, sessions with no user or assistant turn, and the engine's own
headless replay calls. ``_is_agent_session`` is deliberately not applied.

Decompression
-------------
Reading a compressed log needs a Zstandard decoder. Python 3.14's
``compression.zstd`` (or the ``backports.zstd`` port) is used when present,
then the optional ``zstandard`` package, then the system ``libzstd`` through
``ctypes``. Without any of them the compressed sessions are skipped with one
warning instead of crashing the harvest, and a corrupt or truncated log is
treated as unreadable rather than aborting the walk.

This module performs NO writes and NO network calls.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import importlib
import io
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from skillopt_sleep.harvest import (
    _detect_feedback,
    _ended_before,
    _is_headless_replay,
    _is_meta_prompt,
    _project_matches,
    _text_from_content,
)
from skillopt_sleep.staging import redact_secrets
from skillopt_sleep.types import SessionDigest

# ``session.v4.jsonl.zstd`` first: a directory that somehow holds both the
# compressed and the plaintext log is read once, from the compressed one.
_SESSION_LOG_NAMES = ("session.v4.jsonl.zstd", "session.v4.jsonl")

# Host-injected ``user/message`` channels. They arrive with role "user" but
# the user typed none of it; a closed list, so an unknown channel keeps its
# text instead of losing a real request.
_META_USER_SOURCE_KINDS = frozenset({
    "runtime-context",
    "skill-catalog",
    "time-context",
    "tool-jobs",
})

_TOOL_NAME_RE = re.compile(r"[^A-Za-z0-9_.:-]+")

# Export-time personal-data passes (see "Data handling"). The credential
# directory list is closed on purpose: a path this pattern does not know is
# ordinary text, and over-redacting a task's context hides what it is about.
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_URL_USERINFO_RE = re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://)[^\s/@]+@")
_CREDENTIAL_PATH_RE = re.compile(
    r"(?i)(?:~|/(?:home|Users)/[^/\s]+)"
    r"/(?:\.ssh|\.aws|\.gnupg|\.kube|\.docker|\.netrc|\.npmrc|\.pypirc)\b[^\s\"'`)]*"
)

_DECODER_LIBS: Tuple[str, ...] = ("compression.zstd", "backports.zstd")

_PY_ZSTD_MODULE: Any = None
_PY_ZSTD_LOOKED_UP = False
_LIBZSTD: Any = None
_LIBZSTD_LOOKED_UP = False


# ── Zstandard decoding ────────────────────────────────────────────────────────

def _python_zstd_module() -> Any:
    """Return the stdlib/backports ``zstd`` module, or None."""
    global _PY_ZSTD_MODULE, _PY_ZSTD_LOOKED_UP
    if _PY_ZSTD_LOOKED_UP:
        return _PY_ZSTD_MODULE
    _PY_ZSTD_LOOKED_UP = True
    for name in _DECODER_LIBS:
        try:
            _PY_ZSTD_MODULE = importlib.import_module(name)
            return _PY_ZSTD_MODULE
        except Exception:
            continue
    return None


def _libzstd() -> Any:
    """Load ``libzstd`` once; return None when the system has no decoder."""
    global _LIBZSTD, _LIBZSTD_LOOKED_UP
    if _LIBZSTD_LOOKED_UP:
        return _LIBZSTD
    _LIBZSTD_LOOKED_UP = True
    try:
        name = ctypes.util.find_library("zstd")
        if not name:
            return None
        lib = ctypes.CDLL(name)
        lib.ZSTD_createDStream.restype = ctypes.c_void_p
        lib.ZSTD_initDStream.argtypes = [ctypes.c_void_p]
        lib.ZSTD_initDStream.restype = ctypes.c_size_t
        lib.ZSTD_decompressStream.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ]
        lib.ZSTD_decompressStream.restype = ctypes.c_size_t
        lib.ZSTD_freeDStream.argtypes = [ctypes.c_void_p]
        lib.ZSTD_isError.argtypes = [ctypes.c_size_t]
        lib.ZSTD_isError.restype = ctypes.c_uint
        _LIBZSTD = lib
    except Exception:
        _LIBZSTD = None
    return _LIBZSTD


class _ZstdBuffer(ctypes.Structure):
    """``ZSTD_inBuffer`` / ``ZSTD_outBuffer``: dst, size, pos."""

    _fields_ = [
        ("dst", ctypes.c_void_p),
        ("size", ctypes.c_size_t),
        ("pos", ctypes.c_size_t),
    ]


def _decompress_with_libzstd(lib: Any, raw: bytes) -> Optional[bytes]:
    """Stream every concatenated frame; keep the prefix before a torn tail.

    dsh appends each batch as a complete frame. A log being written at harvest
    time can end in a partially flushed frame, so decoding stops at the first
    incomplete or invalid frame and returns what was committed before it.
    """
    stream = lib.ZSTD_createDStream()
    if not stream:
        return None
    try:
        if lib.ZSTD_isError(lib.ZSTD_initDStream(stream)):
            return None
        source = ctypes.create_string_buffer(raw)
        in_buffer = _ZstdBuffer(ctypes.cast(source, ctypes.c_void_p), len(raw), 0)
        out = bytearray()
        chunk = 1 << 17
        while in_buffer.pos < in_buffer.size:
            consumed_before = in_buffer.pos
            target = ctypes.create_string_buffer(chunk)
            out_buffer = _ZstdBuffer(
                ctypes.cast(target, ctypes.c_void_p), chunk, 0,
            )
            remaining = lib.ZSTD_decompressStream(
                stream, ctypes.byref(out_buffer), ctypes.byref(in_buffer),
            )
            if lib.ZSTD_isError(remaining):
                break
            out += target.raw[:out_buffer.pos]
            if remaining == 0 and in_buffer.pos < in_buffer.size:
                # Frame complete; the next appended batch starts a new frame.
                if lib.ZSTD_isError(lib.ZSTD_initDStream(stream)):
                    break
            elif in_buffer.pos >= in_buffer.size:
                break
            elif in_buffer.pos == consumed_before and out_buffer.pos == 0:
                # Corrupt input that neither consumes nor emits bytes.
                break
        return bytes(out)
    finally:
        lib.ZSTD_freeDStream(stream)


def _module_zstd_decompress(module: Any, raw: bytes) -> Optional[bytes]:
    """Decode every concatenated frame with a stdlib/backports zstd module.

    ``compression.zstd`` (and the ``backports.zstd`` port) expose a zlib-style
    decompress object: one call decodes a frame and leaves the next frame in
    ``unused_data``. Looping over those frames mirrors the libzstd streaming
    path, and a truncated tail yields the committed prefix instead of losing
    the whole session.
    """
    out = bytearray()
    remaining = raw
    while remaining:
        decoder = module.ZstdDecompressor()
        try:
            out += decoder.decompress(remaining)
        except Exception:
            return bytes(out) if out else None
        idle = 0
        while not decoder.eof and not decoder.needs_input:
            try:
                more = decoder.decompress(b"")
            except Exception:
                return bytes(out) if out else None
            if not more:
                idle += 1
                if idle > 1:
                    break
            out += more
        if not decoder.eof:
            # Input ended inside a frame; keep what committed frames produced.
            return bytes(out) if out else None
        remaining = decoder.unused_data
    return bytes(out)


def _decompress_zstd(raw: bytes) -> Optional[bytes]:
    """Decode a dsh Zstandard log, or return None if it cannot be decoded."""
    module = _python_zstd_module()
    if module is not None:
        if hasattr(module, "ZstdDecompressor"):
            decoded = _module_zstd_decompress(module, raw)
        else:
            try:
                decoded = bytes(module.decompress(raw))
            except Exception:
                decoded = None
        if decoded:
            return decoded
    try:
        import zstandard  # type: ignore[import-not-found]
    except Exception:
        zstandard = None  # type: ignore[assignment]
    if zstandard is not None:
        try:
            with zstandard.ZstdDecompressor().stream_reader(
                io.BytesIO(raw), read_across_frames=True,
            ) as reader:
                return bytes(reader.read())
        except Exception:
            pass
    lib = _libzstd()
    if lib is not None:
        decoded = _decompress_with_libzstd(lib, raw)
        if decoded:
            return decoded
    return None


def _zstd_decoder_available() -> bool:
    """True when some Zstandard decoder can be used on this host."""
    if _python_zstd_module() is not None:
        return True
    if _libzstd() is not None:
        return True
    try:
        import zstandard  # type: ignore[import-not-found]  # noqa: F401
    except Exception:
        return False
    return True


# ── Record parsing ────────────────────────────────────────────────────────────

def _sanitize_tool_name(name: str) -> str:
    return _TOOL_NAME_RE.sub("_", str(name))[:80]


def _iso_from_epoch_ms(value: Any) -> str:
    """Convert a dsh epoch-millisecond timestamp to a UTC ISO-8601 string."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return ""
    try:
        stamp = datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return ""
    return stamp.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _read_records(path: str) -> Optional[List[Dict[str, Any]]]:
    """Read one v4 session log into JSON objects, or None if unreadable.

    A line that is not a JSON object is skipped (a log can be cut mid-line
    while it is being appended to); a file that cannot be read or decompressed
    is rejected as a whole.
    """
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except (OSError, ValueError):
        return None
    if path.endswith(".zstd"):
        decoded = _decompress_zstd(raw)
        if decoded is None:
            return None
        raw = decoded
    records: List[Dict[str, Any]] = []
    for line in raw.split(b"\n"):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line.decode("utf-8", "replace"))
        except Exception:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _redact_export_text(text: str) -> str:
    """Remove credentials and contact details before text leaves the adapter.

    ``redact_secrets`` already covers API keys, bearer tokens, private keys
    and password assignments. The passes below additionally drop e-mail
    addresses, credentials embedded in a URL's userinfo, and the well-known
    credential directories. This is a bounded pass, not a claim to detect
    arbitrary personal data.
    """
    text = redact_secrets(text)
    text = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = _URL_USERINFO_RE.sub(r"\1[REDACTED]@", text)
    return _CREDENTIAL_PATH_RE.sub("[REDACTED_CREDENTIAL_PATH]", text)


def _relayed_text(text: str) -> str:
    """Unwrap a dsh-superagent relay card to the human text it carries.

    The harness delivers a router candidate card (``text`` at the top level)
    or a worker assignment card (``trigger.text``) as the user turn. Miners
    need the request, not the envelope; a card with neither key, or text that
    is not JSON, is kept verbatim.
    """
    try:
        card = json.loads(text)
    except (TypeError, ValueError):
        return text
    if not isinstance(card, dict):
        return text
    candidates = [card.get("text")]
    trigger = card.get("trigger")
    if isinstance(trigger, dict):
        candidates.append(trigger.get("text"))
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return text


def _user_message_text(message: Dict[str, Any]) -> Optional[str]:
    """The user text of one ``user/message`` record, or None if it is not one."""
    source = message.get("source")
    source = source if isinstance(source, dict) else {}
    kind = source.get("kind")
    if isinstance(kind, str):
        if kind in _META_USER_SOURCE_KINDS:
            return None
        if kind == "dsh-superagent" and source.get("form") == "notice":
            return None
    text = _text_from_content(message.get("content"))
    if kind == "dsh-superagent":
        text = _relayed_text(text)
    text = _redact_export_text(text).strip()
    if not text or _is_meta_prompt(text):
        return None
    return text


def _skill_name(tool_name: Any, arguments: Any) -> str:
    """Skill target of a dsh ``skill`` call; only its ``name`` argument is read."""
    if tool_name != "skill":
        return ""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (TypeError, ValueError):
            return ""
    if not isinstance(arguments, dict):
        return ""
    name = arguments.get("name")
    if not isinstance(name, str):
        return ""
    return name.strip()


def _dedup(values: List[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def digest_dsh_session(path: str, project: str = "") -> Optional[SessionDigest]:
    """Build a :class:`SessionDigest` from one dsh v4 session log."""
    records = _read_records(path)
    if not records:
        return None

    header = next((rec for rec in records if rec.get("type") == "session"), None)
    if not isinstance(header, dict) or header.get("version") != 4:
        return None

    started = _iso_from_epoch_ms(header.get("createdAt"))
    ended = started
    session_project = header.get("cwd")
    session_project = session_project if isinstance(session_project, str) else ""
    session_id = header.get("id")
    if not isinstance(session_id, str) or not session_id:
        session_id = os.path.basename(os.path.dirname(path))
    agent_preset = header.get("agentPreset")
    agent_preset = agent_preset if isinstance(agent_preset, str) else ""
    delegation_depth = header.get("delegationDepth")
    if isinstance(delegation_depth, bool) or not isinstance(delegation_depth, int):
        delegation_depth = 0

    user_prompts: List[str] = []
    assistant_finals: List[str] = []
    tools: List[str] = []
    skills: List[str] = []
    feedback: List[str] = []
    n_user = 0
    n_asst = 0
    n_tool_failures = 0

    for record in records:
        record_type = record.get("type")
        stamp = _iso_from_epoch_ms(record.get("time"))
        if stamp:
            ended = stamp
        data = record.get("data")
        data = data if isinstance(data, dict) else {}

        if record_type == "user/message":
            text = _user_message_text(data)
            if text is None:
                continue
            n_user += 1
            user_prompts.append(text)
            feedback.extend(_detect_feedback(text))

        elif record_type == "assistant/message":
            message = data.get("message")
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            n_asst += 1
            content = message.get("content")
            text = _text_from_content(content)
            if text.strip():
                assistant_finals.append(_redact_export_text(text).strip())
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict) or part.get("type") != "tool-call":
                        continue
                    name = part.get("name")
                    if isinstance(name, str) and name:
                        tools.append(_sanitize_tool_name(name))
                        skill = _skill_name(name, part.get("arguments"))
                        if skill:
                            skills.append(skill)

        elif record_type == "tool/call":
            name = data.get("name")
            if isinstance(name, str) and name:
                tools.append(_sanitize_tool_name(name))
                skill = _skill_name(name, data.get("arguments"))
                if skill:
                    skills.append(skill)

        elif record_type == "tool/result":
            message = data.get("message")
            if isinstance(message, dict) and message.get("isError") is True:
                n_tool_failures += 1

    if n_user == 0 and n_asst == 0:
        return None
    if project and not _project_matches(session_project, "invoked", project):
        return None

    digest = SessionDigest(
        session_id=session_id,
        project=session_project,
        started_at=started,
        ended_at=ended,
        user_prompts=user_prompts,
        assistant_finals=assistant_finals[-5:],
        tools_used=_dedup(tools),
        skills_used=_dedup(skills),
        files_touched=[],  # v4 logs carry no file list without tool arguments
        feedback_signals=_dedup(feedback),
        n_user_turns=n_user,
        n_assistant_turns=n_asst,
        raw_path=path,
        n_tool_failures=n_tool_failures,
        agent_preset=agent_preset,
        delegation_depth=delegation_depth,
    )
    if _is_headless_replay(digest):
        return None
    return digest


# ── Entry point ───────────────────────────────────────────────────────────────

def _session_logs(sessions_dir: str) -> List[Tuple[float, str]]:
    """Every session log below the store, newest first, one log per session."""
    logs: List[Tuple[float, str]] = []
    found_zstd = False
    for root, _dirs, files in os.walk(sessions_dir):
        for name in _SESSION_LOG_NAMES:
            if name not in files:
                continue
            path = os.path.join(root, name)
            if name.endswith(".zstd"):
                found_zstd = True
            try:
                logs.append((os.path.getmtime(path), path))
            except OSError:
                # A session can disappear while dsh rotates its store.
                break
            break
    if found_zstd and not _zstd_decoder_available():
        logging.getLogger("skillopt_sleep").warning(
            "dsh sessions are Zstandard-compressed but no decoder is "
            "available; install the 'zstandard' package (or run Python 3.14+) "
            "to harvest them"
        )
    logs.sort(key=lambda item: (-item[0], item[1]))
    return logs


def _normalize_allowlist(value: Any) -> List[str]:
    """Absolute project paths a caller opts in to, or [] when unrestricted."""
    if not value:
        return []
    if isinstance(value, str):
        value = [value]
    allowed: List[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            allowed.append(os.path.abspath(os.path.expanduser(text)))
    return allowed


def _allowed_project(project: str, allowed: List[str]) -> bool:
    """True when ``project`` is one of the allowed paths or below one."""
    if not allowed:
        return True
    candidate = os.path.abspath(project)
    return any(
        candidate == entry or candidate.startswith(entry + os.sep)
        for entry in allowed
    )


def harvest_dsh(
    sessions_dir: str,
    *,
    scope: Any = "all",
    invoked_project: str = "",
    since_iso: Optional[str] = None,
    limit: int = 0,
    allowed_projects: Any = None,
) -> List[SessionDigest]:
    """Walk a dsh session store (``~/.dsh/sessions``) and return digests.

    Parameters mirror :func:`skillopt_sleep.harvest.harvest`, plus
    ``allowed_projects``: an opt-in allowlist of project paths. When it is
    non-empty a session is harvested only if its recorded ``cwd`` equals one of
    those paths or lives below one, even under ``scope="all"``. Sessions are
    returned newest first, with ties broken by path, and a session id is
    harvested at most once. Both router and worker sessions are included; see
    the module docstring.
    """
    digests: List[SessionDigest] = []
    if not os.path.isdir(sessions_dir):
        return digests

    allowed = _normalize_allowlist(allowed_projects)
    if scope == "all" and not allowed:
        logging.getLogger("skillopt_sleep").warning(
            "harvesting every dsh project because dsh_allowed_projects is "
            "empty; list the projects to mine to opt in explicitly"
        )

    seen: set = set()
    project_hint = invoked_project if scope == "invoked" else ""
    for _mtime, path in _session_logs(sessions_dir):
        digest = digest_dsh_session(path, project=project_hint)
        if digest is None:
            continue
        if not _allowed_project(digest.project or "", allowed):
            continue
        if not _project_matches(digest.project or "", scope, invoked_project):
            continue
        if digest.session_id in seen:
            continue
        if since_iso and digest.ended_at and _ended_before(digest.ended_at, since_iso):
            continue
        seen.add(digest.session_id)
        digests.append(digest)
        if limit and len(digests) >= limit:
            break
    return digests
