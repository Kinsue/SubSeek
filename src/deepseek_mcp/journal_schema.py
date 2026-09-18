"""Strict on-disk schema for the durable mutation-intent journal.

Kept separate from the secure storage backend so both stay within source
budgets. Version 1 records predate job attribution and load as "unknown".
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

MAX_RECORDS, MAX_RECORD_BYTES, MAX_WARNING_BYTES = 128, 64 * 1024, 4096
_TOOLS = frozenset({"Write", "Edit", "NotebookEdit"})
_ID = re.compile(r"[0-9a-f]{32}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_IDENTITY = re.compile(r"(?:[0-9a-f]{2}){1,4096}")
_JOB_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
_AGENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
_KEYS = frozenset(
    {"version", "transaction_id", "workspace_identity", "tool", "path", "sha256", "warnings"}
)
_ATTRIBUTION_KEYS = ("job_id", "agent", "started_at")
_KEYS_V2 = _KEYS | frozenset(_ATTRIBUTION_KEYS)


class TransactionJournalError(RuntimeError):
    pass


class JournalUpdatePublishedWarning(TransactionJournalError):
    pass


@dataclass(frozen=True)
class _StoredRecord:
    transaction_id: str
    workspace_identity: str
    tool: str
    path: str
    sha256: str
    warnings: tuple[str, ...] = ()
    job_id: str = ""
    agent: str = ""
    started_at: float | None = None

    def storage_payload(self) -> dict[str, object]:
        return {"version": 2, "transaction_id": self.transaction_id,
            "workspace_identity": self.workspace_identity, "tool": self.tool,
            "path": self.path, "sha256": self.sha256, "warnings": list(self.warnings),
            "job_id": self.job_id, "agent": self.agent, "started_at": self.started_at}

    def public_payload(self, status: str) -> dict[str, object]:
        return {"transaction_id": self.transaction_id, "tool": self.tool,
            "path": self.path, "sha256": self.sha256, "status": status,
            "warnings": list(self.warnings), "job_id": self.job_id,
            "agent": self.agent, "started_at": self.started_at}


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise TransactionJournalError("journal record contains a duplicate key")
        value[key] = item
    return value


def _reject_constant(_value: str) -> None:
    raise ValueError("non-finite JSON number")


def _utf8_size(value: str, label: str) -> int:
    try:
        return len(value.encode("utf-8", "strict"))
    except UnicodeEncodeError:
        raise TransactionJournalError(f"{label} must be valid Unicode") from None


def _validate_warning(value: object) -> str:
    if not isinstance(value, str):
        raise TransactionJournalError("journal warning must be a string")
    if _utf8_size(value, "journal warning") > MAX_WARNING_BYTES:
        raise TransactionJournalError("journal warning exceeds 4096 UTF-8 bytes")
    return value


def _matched(value: object, pattern: re.Pattern[str], message: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise TransactionJournalError(message)
    return value


def _target_fields(tool: object, path: object) -> tuple[str, str]:
    if tool not in _TOOLS or not isinstance(tool, str) or not isinstance(path, str) or not path:
        raise TransactionJournalError("journal mutation target is invalid")
    if Path(path).is_absolute() or ".." in Path(path).parts:
        raise TransactionJournalError("journal mutation path is not relative")
    return tool, path


def _attribution(value: dict) -> tuple[str, str, float | None]:
    job_id = value.get("job_id", "")
    agent = value.get("agent", "")
    started = value.get("started_at")
    if not isinstance(job_id, str) or (job_id and _JOB_ID.fullmatch(job_id) is None):
        raise TransactionJournalError("journal job attribution is invalid")
    if not isinstance(agent, str) or (agent and _AGENT.fullmatch(agent) is None):
        raise TransactionJournalError("journal agent attribution is invalid")
    if started is not None and (
        isinstance(started, bool) or not isinstance(started, (int, float))
    ):
        raise TransactionJournalError("journal start attribution is invalid")
    return job_id, agent, started


def _validate_stored(value: object) -> _StoredRecord:
    if (
        not isinstance(value, dict)
        or type(value.get("version")) is not int
        or value.get("version") not in (1, 2)
    ):
        raise TransactionJournalError("journal record has an invalid schema")
    version = value["version"]
    if set(value) != (_KEYS if version == 1 else _KEYS_V2):
        raise TransactionJournalError("journal record has an invalid schema")
    transaction_id = _matched(value.get("transaction_id"), _ID, "journal transaction id is invalid")
    identity = _matched(value.get("workspace_identity"), _IDENTITY, "journal workspace identity is invalid")
    tool, path = _target_fields(value.get("tool"), value.get("path"))
    digest = _matched(value.get("sha256"), _DIGEST, "journal mutation digest is invalid")
    warnings = value.get("warnings")
    if not isinstance(warnings, list):
        raise TransactionJournalError("journal warnings are invalid")
    checked = tuple(_validate_warning(item) for item in warnings)
    job_id, agent, started_at = _attribution(value)
    return _StoredRecord(
        transaction_id, identity, tool, path, digest, checked, job_id, agent, started_at,
    )


def _encode(record: _StoredRecord) -> bytes:
    try:
        encoded = json.dumps(record.storage_payload(), separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise TransactionJournalError("journal record cannot be encoded") from error
    if len(encoded) > MAX_RECORD_BYTES:
        raise TransactionJournalError("journal record exceeds 64 KiB")
    return encoded


def _decode(data: bytes) -> _StoredRecord:
    if len(data) > MAX_RECORD_BYTES:
        raise TransactionJournalError("journal record exceeds 64 KiB")
    try:
        text = data.decode("utf-8", "strict")
        value = json.loads(
            text, object_pairs_hook=_strict_object, parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise TransactionJournalError("journal record is not strict UTF-8 JSON") from None
    return _validate_stored(value)
