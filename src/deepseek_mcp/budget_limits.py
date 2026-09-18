"""User-configurable run budgets shared by config loading and the agent loop.

Each budget has a module-level default, an inclusive hard maximum, a config
key in ``~/.deepseek-mcp/config.json``, and an environment variable override
(environment wins over the file, mirroring ``DEEPSEEK_API_KEY`` semantics).
"""
from __future__ import annotations

import os

DEFAULT_MAX_TOTAL_TOKENS_PER_RUN = 1_000_000
HARD_MAX_TOTAL_TOKENS_PER_RUN = 8_000_000
DEFAULT_MAX_HISTORY_TOKENS = 98_304
HARD_MAX_HISTORY_TOKENS = 1_048_576
DEFAULT_MAX_OUTPUT_TOKENS_PER_REQUEST = 16_384
HARD_MAX_OUTPUT_TOKENS_PER_REQUEST = 65_536
DEFAULT_MAX_TOOL_CALLS_PER_TURN = 32
HARD_MAX_TOOL_CALLS_PER_TURN = 256
DEFAULT_MAX_TOOL_CALLS_PER_RUN = 128
HARD_MAX_TOOL_CALLS_PER_RUN = 1_024
DEFAULT_MAX_MUTATION_BYTES_PER_RUN = 64 * 1024 * 1024
HARD_MAX_MUTATION_BYTES_PER_RUN = 1024 ** 3
# (config key, environment variable, default, inclusive hard maximum)
BUDGET_LIMIT_FIELDS = (
    (
        "max_total_tokens_per_run",
        "DEEPSEEK_MAX_TOTAL_TOKENS_PER_RUN",
        DEFAULT_MAX_TOTAL_TOKENS_PER_RUN,
        HARD_MAX_TOTAL_TOKENS_PER_RUN,
    ),
    (
        "max_history_tokens",
        "DEEPSEEK_MAX_HISTORY_TOKENS",
        DEFAULT_MAX_HISTORY_TOKENS,
        HARD_MAX_HISTORY_TOKENS,
    ),
    (
        "max_output_tokens_per_request",
        "DEEPSEEK_MAX_OUTPUT_TOKENS_PER_REQUEST",
        DEFAULT_MAX_OUTPUT_TOKENS_PER_REQUEST,
        HARD_MAX_OUTPUT_TOKENS_PER_REQUEST,
    ),
    (
        "max_tool_calls_per_turn",
        "DEEPSEEK_MAX_TOOL_CALLS_PER_TURN",
        DEFAULT_MAX_TOOL_CALLS_PER_TURN,
        HARD_MAX_TOOL_CALLS_PER_TURN,
    ),
    (
        "max_tool_calls_per_run",
        "DEEPSEEK_MAX_TOOL_CALLS_PER_RUN",
        DEFAULT_MAX_TOOL_CALLS_PER_RUN,
        HARD_MAX_TOOL_CALLS_PER_RUN,
    ),
    (
        "max_mutation_bytes_per_run",
        "DEEPSEEK_MAX_MUTATION_BYTES_PER_RUN",
        DEFAULT_MAX_MUTATION_BYTES_PER_RUN,
        HARD_MAX_MUTATION_BYTES_PER_RUN,
    ),
)
BUDGET_CONFIG_KEYS = frozenset(key for key, _env, _default, _hard in BUDGET_LIMIT_FIELDS)


def validate_budget_limit(key: str, value: object, hard_max: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{key} must be an integer")
    if not 1 <= value <= hard_max:
        raise RuntimeError(f"{key} must be between 1 and {hard_max}")
    return value


def _load_optional_int_env(env_name: str, hard_max: int) -> int | None:
    raw = os.getenv(env_name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        raise RuntimeError(f"{env_name} must be an integer") from None
    if not 1 <= value <= hard_max:
        raise RuntimeError(f"{env_name} must be between 1 and {hard_max}")
    return value


def load_budget_limits(data: dict) -> dict:
    limits: dict = {}
    for key, env_name, default, hard_max in BUDGET_LIMIT_FIELDS:
        env_value = _load_optional_int_env(env_name, hard_max)
        if env_value is not None:
            limits[key] = env_value
        else:
            limits[key] = validate_budget_limit(key, data.get(key, default), hard_max)
    return limits
