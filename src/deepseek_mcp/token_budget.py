"""Usage-based token accounting for the agent loop.

The provider reports ``prompt_tokens`` for every request and ``completion_tokens``
for every response. Every prompt already contains the full prior conversation,
so summing prompt tokens would double-count history. The run total is therefore
the most recent prompt plus the accumulated completions. Encoded byte sizes are
never mixed into token sums; they are only used for a conservative estimate
before the first provider response and as an internal transport guard.
"""
from __future__ import annotations

import json

from .budget_limits import (
    DEFAULT_MAX_HISTORY_TOKENS,
    DEFAULT_MAX_OUTPUT_TOKENS_PER_REQUEST,
    DEFAULT_MAX_TOTAL_TOKENS_PER_RUN,
)
from .provider_retry import AgentLoopError

# Internal transport guard only: never configurable, never a token estimate.
MAX_PROVIDER_HISTORY_BYTES = 12 * 1024 * 1024
# Conservative bytes-per-token ratio for the pre-first-response estimate.
BYTES_PER_TOKEN = 4


class BudgetExceededError(AgentLoopError):
    """A configured run budget was exhausted."""


_BUDGET_LOG_KEYS = (
    "max_turns",
    "max_run_seconds",
    "max_total_tokens_per_run",
    "max_history_tokens",
    "max_output_tokens_per_request",
    "max_tool_calls_per_turn",
    "max_tool_calls_per_run",
    "max_mutation_bytes_per_run",
)
_MAX_ERROR_MESSAGE_CHARS = 500


def is_budget_error(error: BaseException) -> bool:
    # The substring half deliberately nets budget-class errors that cross a
    # subprocess boundary and lose their BudgetExceededError type. The message
    # surface reaching server.py is fixed internal strings, so keep this as-is;
    # do not widen it or "simplify" it to isinstance-only.
    return isinstance(error, BudgetExceededError) or "budget" in str(error).lower()


def bounded_budget_message(error: BaseException) -> str:
    return f"ERROR: {str(error)[:_MAX_ERROR_MESSAGE_CHARS]}"


def log_effective_budgets(log) -> None:
    """Log the effective numeric budgets once (no secrets)."""
    try:
        from .config import Config

        config = Config.load()
    except Exception as error:
        log.debug("effective budgets unavailable: %s", type(error).__name__)
        return
    log.info(
        "effective budgets: %s",
        " ".join(f"{key}={getattr(config, key)}" for key in _BUDGET_LOG_KEYS),
    )


def config_int(source, key: str, default: int) -> int:
    return getattr(source, key, default)


def run_usage(state) -> int:
    # History is append-only (no compaction/truncation), so the most recent
    # reported prompt_tokens is monotone and run_usage can never undercount.
    # If context compaction is ever added, this accounting must be revisited.
    last_prompt = getattr(state, "last_prompt_tokens", None) or 0
    return last_prompt + state.total_completion_tokens


def encoded_size(value: object) -> int:
    encoder = json.JSONEncoder(separators=(",", ":"), ensure_ascii=True)
    return sum(len(chunk.encode("utf-8")) for chunk in encoder.iterencode(value))


def check_run_token_budget(state) -> None:
    used = run_usage(state)
    limit = config_int(
        state.config, "max_total_tokens_per_run", DEFAULT_MAX_TOTAL_TOKENS_PER_RUN
    )
    if used > limit:
        raise BudgetExceededError(
            f"run token budget exceeded: used={used} limit={limit} "
            f"(config: max_total_tokens_per_run)"
        )


def ensure_request_budget(state) -> None:
    last_prompt = getattr(state, "last_prompt_tokens", None)
    if last_prompt is None:
        history_tokens = encoded_size(state.messages) // BYTES_PER_TOKEN
    else:
        history_tokens = last_prompt
    limit = config_int(
        state.config, "max_total_tokens_per_run", DEFAULT_MAX_TOTAL_TOKENS_PER_RUN
    )
    output = config_int(
        state.config,
        "max_output_tokens_per_request",
        DEFAULT_MAX_OUTPUT_TOKENS_PER_REQUEST,
    )
    reserved = history_tokens + output
    used = run_usage(state)
    remaining = limit - used
    if reserved > remaining:
        raise BudgetExceededError(
            f"run token budget cannot cover another provider request: "
            f"used={used} limit={limit} reserved={reserved} "
            f"(config: max_total_tokens_per_run)"
        )


def enforce_history_budget(state) -> int:
    """Reject oversized history before the next provider request.

    The token check lags one round-trip by design: it compares the previous
    request's reported prompt_tokens, so messages appended since then are caught
    on the next round, or bounded meanwhile by the exact byte guard below.
    """
    size = encoded_size(state.messages)
    if size > MAX_PROVIDER_HISTORY_BYTES:
        raise BudgetExceededError(
            "provider conversation history budget exceeded "
            f"(byte cap {MAX_PROVIDER_HISTORY_BYTES})"
        )
    last_prompt = getattr(state, "last_prompt_tokens", None)
    if last_prompt is None:
        history_tokens = size // BYTES_PER_TOKEN
    else:
        history_tokens = last_prompt
    limit = config_int(state.config, "max_history_tokens", DEFAULT_MAX_HISTORY_TOKENS)
    if history_tokens > limit:
        raise BudgetExceededError(
            f"conversation history budget exceeded: tokens={history_tokens} "
            f"limit={limit} (config: max_history_tokens)"
        )
    return size
