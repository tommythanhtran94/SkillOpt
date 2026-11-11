"""Tests for the DeepSeek Harness (dsh) session harvester.

Session logs are written as plaintext ``session.v4.jsonl`` here. To exercise
the Zstandard path without shipping an opaque binary fixture, the tests build
a real Zstandard frame that stores its payload as a *raw* (uncompressed) block.
That is a valid zstd frame — including the "content size unknown" header dsh
writes — so every decoder the harvester can use must accept it.
"""
from __future__ import annotations

import json
import logging
import os

import pytest

from skillopt_sleep.config import SleepConfig
from skillopt_sleep.harvest_dsh import (
    _decompress_zstd,
    _zstd_decoder_available,
    digest_dsh_session,
    harvest_dsh,
)
from skillopt_sleep.harvest_sources import harvest_for_config
from skillopt_sleep.mine import heuristic_mine

PROJECT = "/home/u/proj"
OTHER_PROJECT = "/other/place"

START_MS = 1_700_000_000_000  # 2023-11-14T22:13:20Z
END_MS = 1_700_000_010_000    # 2023-11-14T22:13:30Z


def _zstd_raw_frame(payload: bytes) -> bytes:
    """A minimal valid zstd frame: magic, header, raw blocks."""
    frame = b"\x28\xb5\x2f\xfd"  # frame magic
    frame += b"\x00"            # no content size, no checksum, no dictionary
    frame += b"\x30"            # 64 KiB window
    for offset in range(0, len(payload), 65536):
        block = payload[offset:offset + 65536]
        last = 1 if offset + 65536 >= len(payload) else 0
        frame += ((last & 1) | (len(block) << 3)).to_bytes(3, "little")
        frame += block
    return frame


def _user(seq: int, time_ms: int, text: str, *, kind: str = "user", form: str | None = None) -> dict:
    source: dict = {"kind": kind}
    if form:
        source["form"] = form
    return {
        "type": "user/message",
        "seq": seq,
        "time": time_ms,
        "data": {
            "id": f"msg-{seq}",
            "role": "user",
            "content": [{"type": "text", "text": text}],
            "source": source,
        },
    }


def _assistant(seq: int, time_ms: int, parts: list) -> dict:
    return {
        "type": "assistant/message",
        "seq": seq,
        "time": time_ms,
        "data": {
            "turn": 1,
            "step": 1,
            "message": {"role": "assistant", "content": parts},
        },
    }


def _tool_call(seq: int, time_ms: int, name: str, arguments: str) -> dict:
    return {
        "type": "tool/call",
        "seq": seq,
        "time": time_ms,
        "data": {
            "turn": 1,
            "step": 1,
            "callId": f"call-{seq}",
            "name": name,
            "arguments": arguments,
        },
    }


def _tool_result(seq: int, time_ms: int, is_error: bool) -> dict:
    return {
        "type": "tool/result",
        "seq": seq,
        "time": time_ms,
        "data": {
            "turn": 1,
            "step": 1,
            "message": {
                "role": "tool",
                "toolCallId": f"call-{seq}",
                "isError": is_error,
                "content": [{"type": "text", "text": "exit code: 1" if is_error else "ok"}],
            },
        },
    }


def _relay_card(text: str, *, nested: bool) -> str:
    if nested:
        return json.dumps({
            "ticket_id": "t1",
            "title": "Build the thing",
            "assignment": "routing note",
            "trigger": {"trigger_id": "x", "text": text},
        })
    return json.dumps({"kind": "candidate", "trigger_id": "x", "text": text, "chars": len(text)})


def _records(
    *,
    project: str = PROJECT,
    session_id: str = "worker-abc",
    preset: str = "dsh-superagent-worker",
    depth: int = 0,
) -> list:
    return [
        {
            "type": "session",
            "version": 4,
            "id": session_id,
            "createdAt": START_MS,
            "cwd": project,
            "isSeeded": False,
            "delegationDepth": depth,
            "agentPreset": preset,
        },
        {"type": "session/title", "seq": 0, "time": START_MS + 100,
         "data": {"title": "parser work", "source": {"kind": "user"}}},
        _user(1, START_MS + 1_000, "Fix the parser bug"),
        # Host-injected context must never become a user prompt.
        _user(2, START_MS + 1_100, "Current working directory: /home/u/proj",
              kind="runtime-context", form="snapshot"),
        _user(3, START_MS + 1_200, "A skill catalog reminder", kind="skill-catalog", form="catalog"),
        _user(4, START_MS + 1_300, "Time sampled while preparing turn 1", kind="time-context", form="snapshot"),
        _user(5, START_MS + 1_400, "background job bash-123 finished", kind="tool-jobs", form="notice"),
        _user(6, START_MS + 1_500, '{"settings_saved":"source:github"}',
              kind="dsh-superagent", form="notice"),
        _assistant(7, START_MS + 2_000, [
            {"type": "reasoning", "text": "private reasoning about the parser"},
            {"type": "text", "text": "Running the suite."},
            {"type": "tool-call", "id": "call-a", "name": "bash", "arguments": "{\"command\":\"pytest\"}"},
            {"type": "tool-call", "id": "call-b", "name": "skill",
             "arguments": "{\"name\":\"dsh-superagent-github\"}"},
        ]),
        _tool_call(8, START_MS + 2_100, "bash", "{\"command\":\"pytest\"}"),
        _tool_result(9, START_MS + 2_200, False),
        _tool_call(10, START_MS + 2_300, "skill", "{\"name\":\"dsh-superagent-github\"}"),
        _tool_result(11, START_MS + 2_400, True),
        # Routed work arrives as a dsh-superagent relay card; the human text is
        # inside the envelope, so the envelope must not bury it.
        _user(12, START_MS + 3_000, _relay_card("Add the missing test", nested=True),
              kind="dsh-superagent", form="relay"),
        _assistant(13, START_MS + 4_000, [{"type": "text", "text": "Added the test."}]),
        _user(14, START_MS + 5_000, _relay_card("Route the new request", nested=False),
              kind="dsh-superagent", form="relay"),
        _assistant(15, END_MS - 1_000, [{"type": "text", "text": "Routed it."}]),
        _user(16, END_MS, "thanks, that works now"),
        {"type": "turn/end", "seq": 17, "time": END_MS,
         "data": {"turn": 1, "reason": {"kind": "completed"}}},
    ]


def _write_session(root, records, *, session_id="worker-abc", project_dir="--home-u-proj--",
                   name="session.v4.jsonl") -> str:
    directory = os.path.join(str(root), project_dir, session_id)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


def _write_compressed(root, records, *, session_id="worker-zstd", payload: bytes | None = None) -> str:
    directory = os.path.join(str(root), "--home-u-proj--", session_id)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "session.v4.jsonl.zstd")
    if payload is None:
        payload = "".join(json.dumps(record) + "\n" for record in records).encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(_zstd_raw_frame(payload))
    return path


# ── digest ────────────────────────────────────────────────────────────────────

def test_digest_extracts_the_conversation(tmp_path):
    path = _write_session(tmp_path, _records())

    digest = digest_dsh_session(path)

    assert digest is not None
    assert digest.session_id == "worker-abc"
    assert digest.project == PROJECT
    assert digest.started_at == "2023-11-14T22:13:20.000Z"
    assert digest.ended_at == "2023-11-14T22:13:30.000Z"
    assert digest.raw_path == path


def test_digest_keeps_only_real_user_turns(tmp_path):
    path = _write_session(tmp_path, _records())

    digest = digest_dsh_session(path)

    assert digest.n_user_turns == 4
    assert digest.user_prompts == [
        "Fix the parser bug",
        "Add the missing test",   # worker assignment relay, unwrapped
        "Route the new request",  # router candidate relay, unwrapped
        "thanks, that works now",
    ]
    assert any(signal.startswith("pos:") for signal in digest.feedback_signals)


def test_digest_ignores_private_reasoning_and_system_records(tmp_path):
    path = _write_session(tmp_path, _records())

    digest = digest_dsh_session(path)

    assert digest.n_assistant_turns == 3
    assert digest.assistant_finals == [
        "Running the suite.", "Added the test.", "Routed it.",
    ]
    assert "private reasoning" not in " ".join(digest.assistant_finals)


def test_digest_collects_tools_skills_and_tool_outcomes(tmp_path):
    path = _write_session(tmp_path, _records())

    digest = digest_dsh_session(path)

    assert digest.tools_used == ["bash", "skill"]
    assert digest.skills_used == ["dsh-superagent-github"]
    # v4 logs expose no file list without tool arguments, which are not kept.
    assert digest.files_touched == []
    # One isError result; mechanical failures are counted, not user feedback.
    assert digest.n_tool_failures == 1
    assert all("exit code" not in signal for signal in digest.feedback_signals)


def test_unknown_user_source_kind_keeps_its_text(tmp_path):
    records = _records()
    records.append(_user(17, END_MS + 1_000, "keep this request", kind="some-future-host"))
    path = _write_session(tmp_path, records)

    digest = digest_dsh_session(path)

    assert digest.user_prompts[-1] == "keep this request"


def test_digest_rejects_unsupported_session_version(tmp_path):
    records = _records()
    records[0] = dict(records[0], version=3)
    path = _write_session(tmp_path, records)

    assert digest_dsh_session(path) is None


def test_digest_skips_unreadable_and_corrupt_logs(tmp_path):
    assert digest_dsh_session(str(tmp_path / "missing.jsonl")) is None

    path = os.path.join(str(tmp_path), "session.v4.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("not json\n")
        handle.write("[1, 2, 3]\n")
    assert digest_dsh_session(path) is None

    corrupt = os.path.join(str(tmp_path), "session.v4.jsonl.zstd")
    with open(corrupt, "wb") as handle:
        handle.write(b"this is not a zstd frame")
    assert digest_dsh_session(corrupt) is None


def test_digest_skips_a_line_cut_mid_write(tmp_path):
    records = _records()
    path = _write_session(tmp_path, records)
    with open(path, "ab") as handle:
        handle.write(b'{"type":"user/message","data":{"content":[{"type":"text","te')

    digest = digest_dsh_session(path)

    assert digest is not None
    assert digest.n_user_turns == 4


def test_single_short_session_is_not_a_headless_replay(tmp_path):
    records = [
        {"type": "session", "version": 4, "id": "s", "createdAt": START_MS, "cwd": PROJECT},
        _user(1, START_MS, "Fix the parser bug"),
        _assistant(2, END_MS, [{"type": "text", "text": "Fixed."}]),
    ]
    path = _write_session(tmp_path, records)

    assert digest_dsh_session(path) is not None


# ── harvest walk ──────────────────────────────────────────────────────────────

def test_harvest_missing_directory_is_empty(tmp_path):
    assert harvest_dsh(str(tmp_path / "no-such-dir")) == []


def test_harvest_scope_and_order(tmp_path):
    _write_session(tmp_path, _records(project=PROJECT), session_id="worker-abc")
    _write_session(
        tmp_path, _records(project=OTHER_PROJECT, session_id="worker-other"),
        session_id="worker-other", project_dir="--other-place--",
    )

    assert len(harvest_dsh(str(tmp_path), scope="all")) == 2
    invoked = harvest_dsh(str(tmp_path), scope="invoked", invoked_project=PROJECT)
    assert [digest.session_id for digest in invoked] == ["worker-abc"]


def test_harvest_limit(tmp_path):
    _write_session(tmp_path, _records(session_id="worker-a"), session_id="worker-a")
    _write_session(tmp_path, _records(session_id="worker-b"), session_id="worker-b")

    digests = harvest_dsh(str(tmp_path), scope="all", limit=1)

    assert len(digests) == 1


def test_harvest_since_cutoff_compares_instants(tmp_path):
    _write_session(tmp_path, _records())

    included = harvest_dsh(str(tmp_path), scope="all", since_iso="2023-11-14T22:13:25Z")
    excluded = harvest_dsh(str(tmp_path), scope="all", since_iso="2023-11-14T22:13:35Z")

    assert len(included) == 1
    assert excluded == []


def test_harvest_skips_a_corrupt_compressed_log_without_crashing(tmp_path):
    _write_session(tmp_path, _records())
    directory = os.path.join(str(tmp_path), "--broken--", "worker-broken")
    os.makedirs(directory)
    with open(os.path.join(directory, "session.v4.jsonl.zstd"), "wb") as handle:
        handle.write(b"\x28\xb5\x2f\xfd garbage")

    digests = harvest_dsh(str(tmp_path), scope="all")

    assert [digest.session_id for digest in digests] == ["worker-abc"]


# ── Zstandard logs ────────────────────────────────────────────────────────────

pytestmark_zstd = pytest.mark.skipif(
    not _zstd_decoder_available(), reason="no Zstandard decoder on this host",
)


@pytestmark_zstd
def test_decompress_reads_a_dsh_shaped_frame():
    payload = b'{"type":"session","version":4}\n'

    assert _decompress_zstd(_zstd_raw_frame(payload)) == payload


@pytestmark_zstd
def test_decompress_reads_concatenated_frames():
    payload = b'{"type":"session","version":4}\n'

    assert _decompress_zstd(_zstd_raw_frame(payload) * 2) == payload * 2


@pytestmark_zstd
def test_decompress_keeps_frames_before_a_torn_tail():
    payload = b'{"type":"session","version":4}\n'

    assert _decompress_zstd(_zstd_raw_frame(payload) + b"\x28\xb5\x2f\xfd\x00") == payload


@pytestmark_zstd
def test_harvest_reads_a_compressed_session(tmp_path):
    _write_compressed(tmp_path, _records())

    digests = harvest_dsh(str(tmp_path), scope="all")

    assert [digest.session_id for digest in digests] == ["worker-abc"]
    assert digests[0].user_prompts[0] == "Fix the parser bug"


@pytestmark_zstd
def test_harvest_reads_a_session_being_appended(tmp_path):
    payload = "".join(json.dumps(record) + "\n" for record in _records()).encode("utf-8")
    frame = _zstd_raw_frame(payload)
    _write_compressed(tmp_path, [], payload=payload, session_id="worker-live")
    path = os.path.join(str(tmp_path), "--home-u-proj--", "worker-live", "session.v4.jsonl.zstd")
    with open(path, "wb") as handle:
        handle.write(frame + frame[: len(frame) // 2])

    digests = harvest_dsh(str(tmp_path), scope="all")

    assert [digest.session_id for digest in digests] == ["worker-abc"]


# ── registration ──────────────────────────────────────────────────────────────

def test_dsh_is_a_configurable_source(tmp_path):
    _write_session(tmp_path, _records())
    cfg = SleepConfig()
    cfg.data["transcript_source"] = "dsh"
    cfg.data["dsh_sessions_dir"] = str(tmp_path)
    cfg.data["projects"] = "all"

    digests = harvest_for_config(cfg)

    assert [digest.session_id for digest in digests] == ["worker-abc"]
    assert cfg.dsh_sessions_dir == str(tmp_path)


def test_dsh_sessions_dir_defaults_below_the_home_directory():
    cfg = SleepConfig()

    assert cfg.dsh_sessions_dir == os.path.join(os.path.expanduser("~"), ".dsh", "sessions")


# ── provenance and data handling ──────────────────────────────────────────────

def test_digest_records_worker_provenance(tmp_path):
    path = _write_session(tmp_path, _records(preset="dsh-superagent-worker"))

    digest = digest_dsh_session(path)

    assert digest.session_id == "worker-abc"
    assert digest.agent_preset == "dsh-superagent-worker"
    assert digest.delegation_depth == 0
    assert digest.raw_path == path


def test_digest_records_router_provenance_and_delegation(tmp_path):
    path = _write_session(
        tmp_path,
        _records(session_id="orchestrator-x", preset="dsh-superagent-router", depth=2),
        session_id="orchestrator-x",
    )

    digest = digest_dsh_session(path)

    assert digest.agent_preset == "dsh-superagent-router"
    assert digest.delegation_depth == 2


def test_mined_tasks_carry_session_profile_and_lineage(tmp_path):
    path = _write_session(
        tmp_path,
        _records(session_id="worker-lineage", preset="dsh-superagent-worker", depth=2),
        session_id="worker-lineage",
    )
    digest = digest_dsh_session(path)

    [task] = heuristic_mine([digest], max_tasks=1)

    assert task.source_sessions == ["worker-lineage"]
    assert "profile:dsh-superagent-worker" in task.tags
    assert "lineage:delegation=2" in task.tags


def test_exported_text_redacts_credentials_and_contact_details(tmp_path):
    github_token = "ghp_" + "a" * 30
    secret_text = (
        "Mail tommy@example.com or open https://alice:hunter2@example.com/x "
        f"with {github_token} and ~/.ssh/id_rsa"
    )
    records = [
        {"type": "session", "version": 4, "id": "s", "createdAt": START_MS, "cwd": PROJECT},
        _user(1, START_MS, secret_text),
        _assistant(2, END_MS, [{"type": "text", "text": secret_text}]),
    ]

    digest = digest_dsh_session(_write_session(tmp_path, records))

    exported = " ".join(digest.user_prompts + digest.assistant_finals)
    assert "tommy@example.com" not in exported
    assert "hunter2" not in exported
    assert github_token not in exported
    assert "~/.ssh/id_rsa" not in exported
    assert "[REDACTED_EMAIL]" in exported
    assert "[REDACTED_CREDENTIAL_PATH]" in exported


def test_harvest_is_deterministic_and_deduplicates_session_ids(tmp_path):
    original = _write_session(
        tmp_path, _records(session_id="worker-a"), session_id="worker-a",
    )
    middle = _write_session(
        tmp_path, _records(session_id="worker-b"), session_id="worker-b",
        project_dir="--b--",
    )
    duplicate = _write_session(
        tmp_path, _records(session_id="worker-a"), session_id="worker-a-dup",
        project_dir="--dup--",
    )
    os.utime(original, (1_000, 1_000))
    os.utime(middle, (2_000, 2_000))
    os.utime(duplicate, (3_000, 3_000))

    first = [digest.session_id for digest in harvest_dsh(str(tmp_path), scope="all")]
    second = [digest.session_id for digest in harvest_dsh(str(tmp_path), scope="all")]

    assert first == second
    assert first[0] == "worker-a"  # the newest copy wins
    assert sorted(first) == ["worker-a", "worker-b"]
    assert len(first) == 2


def test_harvest_breaks_mtime_ties_by_path(tmp_path):
    first = _write_session(tmp_path, _records(session_id="worker-a"), session_id="worker-a")
    second = _write_session(
        tmp_path, _records(session_id="worker-b"), session_id="worker-b",
        project_dir="--a--",
    )
    for path in (first, second):
        os.utime(path, (5_000, 5_000))
        os.utime(os.path.dirname(path), (5_000, 5_000))

    ids = [digest.session_id for digest in harvest_dsh(str(tmp_path), scope="all")]

    assert ids == ["worker-b", "worker-a"]  # "--a--" sorts before "--home-u-proj--"


def test_allowlist_limits_scope_all_and_warns_when_absent(tmp_path, caplog):
    _write_session(tmp_path, _records())
    _write_session(
        tmp_path, _records(project=OTHER_PROJECT, session_id="worker-other"),
        session_id="worker-other", project_dir="--other--",
    )

    with caplog.at_level(logging.WARNING, logger="skillopt_sleep"):
        everything = harvest_dsh(str(tmp_path), scope="all")
    assert len(everything) == 2
    assert "dsh_allowed_projects" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="skillopt_sleep"):
        limited = harvest_dsh(str(tmp_path), scope="all", allowed_projects=[PROJECT])
    assert [digest.session_id for digest in limited] == ["worker-abc"]
    assert "dsh_allowed_projects" not in caplog.text


def test_allowlist_matches_a_parent_project_path(tmp_path):
    _write_session(tmp_path, _records())

    digests = harvest_dsh(str(tmp_path), scope="all", allowed_projects=["/home/u"])

    assert [digest.session_id for digest in digests] == ["worker-abc"]


def test_config_passes_the_allowlist_to_the_harvester(tmp_path):
    _write_session(tmp_path, _records())
    _write_session(
        tmp_path, _records(project=OTHER_PROJECT, session_id="worker-other"),
        session_id="worker-other", project_dir="--other--",
    )
    cfg = SleepConfig()
    cfg.data["transcript_source"] = "dsh"
    cfg.data["dsh_sessions_dir"] = str(tmp_path)
    cfg.data["projects"] = "all"
    cfg.data["dsh_allowed_projects"] = [PROJECT]

    digests = harvest_for_config(cfg)

    assert [digest.session_id for digest in digests] == ["worker-abc"]
