"""User-configurable run budgets shared by config loading and the agent loop.

Each budget has a module-level default, a config key in
``~/.deepseek-mcp/config.json``, and an environment variable override
(environment wins over the file, mirroring ``DEEPSEEK_API_KEY`` semantics).
There are no client-side hard maxima: the provider is the authority and rejects
over-limit requests. Values only need to be positive integers, and cross-key
consistency is still enforced so a smaller limit cannot exceed a larger one.
"""
from __future__ import annotations

import os

DEFAULT_MAX_TOTAL_TOKENS_PER_RUN = 1_000_000
DEFAULT_MAX_HISTORY_TOKENS = 98_304
DEFAULT_MAX_OUTPUT_TOKENS_PER_REQUEST = 16_384
DEFAULT_MAX_TOOL_CALLS_PER_TURN = 32
DEFAULT_MAX_TOOL_CALLS_PER_RUN = 128
DEFAULT_MAX_MUTATION_BYTES_PER_RUN = 64 * 1024 * 1024
# (config key, environment variable, default)
BUDGET_LIMIT_FIELDS = (
    (
        "max_total_tokens_per_run",
        "DEEPSEEK_MAX_TOTAL_TOKENS_PER_RUN",
        DEFAULT_MAX_TOTAL_TOKENS_PER_RUN,
    ),
    (
        "max_history_tokens",
        "DEEPSEEK_MAX_HISTORY_TOKENS",
        DEFAULT_MAX_HISTORY_TOKENS,
    ),
    (
        "max_output_tokens_per_request",
        "DEEPSEEK_MAX_OUTPUT_TOKENS_PER_REQUEST",
        DEFAULT_MAX_OUTPUT_TOKENS_PER_REQUEST,
    ),
    (
        "max_tool_calls_per_turn",
        "DEEPSEEK_MAX_TOOL_CALLS_PER_TURN",
        DEFAULT_MAX_TOOL_CALLS_PER_TURN,
    ),
    (
        "max_tool_calls_per_run",
        "DEEPSEEK_MAX_TOOL_CALLS_PER_RUN",
        DEFAULT_MAX_TOOL_CALLS_PER_RUN,
    ),
    (
        "max_mutation_bytes_per_run",
        "DEEPSEEK_MAX_MUTATION_BYTES_PER_RUN",
        DEFAULT_MAX_MUTATION_BYTES_PER_RUN,
    ),
)
# Owner-friendly alias -> canonical budget key (provider-axis naming).
BUDGET_KEY_ALIASES = {
    "max_input_token": "max_history_tokens",
    "max_output_token": "max_output_tokens_per_request",
}
_ALIAS_BY_KEY = {target: alias for alias, target in BUDGET_KEY_ALIASES.items()}
BUDGET_CONFIG_KEYS = frozenset(key for key, _env, _default in BUDGET_LIMIT_FIELDS)


def validate_budget_limit(key: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RuntimeError(f"{key} must be a positive integer")
    return value


def load_optional_int_env(env_name: str) -> int | None:
    raw = os.getenv(env_name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError:
        raise RuntimeError(f"{env_name} must be a positive integer") from None
    if value < 1:
        raise RuntimeError(f"{env_name} must be a positive integer")
    return value


def validate_budget_cross_limits(limits: dict) -> None:
    """Reject budget combinations whose smaller limit can never be used."""
    if limits["max_output_tokens_per_request"] > limits["max_total_tokens_per_run"]:
        raise RuntimeError(
            "max_output_tokens_per_request must not exceed "
            "max_total_tokens_per_run"
        )
    if limits["max_tool_calls_per_turn"] > limits["max_tool_calls_per_run"]:
        raise RuntimeError(
            "max_tool_calls_per_turn must not exceed max_tool_calls_per_run"
        )


def _file_budget_value(data: dict, key: str) -> tuple[object, bool]:
    alias = _ALIAS_BY_KEY.get(key)
    canonical_present = key in data
    alias_present = alias is not None and alias in data
    if canonical_present and alias_present:
        raise RuntimeError(f"{key} and {alias} cannot both be set")
    if canonical_present:
        return data[key], True
    if alias_present:
        return data[alias], True
    return None, False


def load_budget_limits(data: dict) -> dict:
    limits: dict = {}
    for key, env_name, default in BUDGET_LIMIT_FIELDS:
        value, present = _file_budget_value(data, key)
        env_value = load_optional_int_env(env_name)
        if env_value is not None:
            limits[key] = env_value
        elif present:
            limits[key] = validate_budget_limit(key, value)
        else:
            limits[key] = validate_budget_limit(key, default)
    validate_budget_cross_limits(limits)
    return limits


SAME_WORKSPACE_WRITERS_ENV = "DEEPSEEK_SAME_WORKSPACE_WRITERS"
SAME_WORKSPACE_WRITERS_ALLOW = "allow"
SAME_WORKSPACE_WRITERS_EXCLUSIVE = "exclusive"
DEFAULT_SAME_WORKSPACE_WRITERS = SAME_WORKSPACE_WRITERS_ALLOW
SAME_WORKSPACE_WRITERS_OPTIONS = (
    SAME_WORKSPACE_WRITERS_ALLOW,
    SAME_WORKSPACE_WRITERS_EXCLUSIVE,
)


def validate_same_workspace_writers(value: object) -> str:
    if not isinstance(value, str) or value not in SAME_WORKSPACE_WRITERS_OPTIONS:
        options = ", ".join(SAME_WORKSPACE_WRITERS_OPTIONS)
        raise RuntimeError(f"same_workspace_writers must be one of: {options}")
    return value


def load_same_workspace_writers(data: dict) -> str:
    """Resolve the same-workspace writer mode (env wins; empty env is unset)."""
    raw = os.getenv(SAME_WORKSPACE_WRITERS_ENV)
    if raw is None or not raw.strip():
        raw = data.get("same_workspace_writers", DEFAULT_SAME_WORKSPACE_WRITERS)
    return validate_same_workspace_writers(raw)
