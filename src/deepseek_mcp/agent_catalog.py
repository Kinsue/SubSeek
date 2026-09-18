"""Data-defined agent catalog: built-ins, validation, resolution, rendering.

Lives outside ``config`` so it stays import-cycle free (``config`` imports this
module, never the reverse) and so ``config``/``server`` keep their line budgets.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

DEFAULT_ALLOWED_TOOLS = [
    "Read",
    "Write",
    "Edit",
    "Bash",
    "Glob",
    "Grep",
    "NotebookEdit",
]
KNOWN_TOOLS = frozenset(
    {"Read", "Write", "Edit", "Bash", "Glob", "Grep", "NotebookEdit"}
)
MUTATION_TOOLS = frozenset({"Write", "Edit", "NotebookEdit"})
READONLY_TOOLS = ("Read", "Glob", "Grep")
AGENT_CAPABILITIES = ("coding", "readonly")
DEFAULT_TOOLS_BY_CAPABILITY = {
    "coding": tuple(DEFAULT_ALLOWED_TOOLS),
    "readonly": READONLY_TOOLS,
}
AGENT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9-]{1,31}$")
MAX_DESCRIPTION_CHARS = 200


@dataclass(frozen=True)
class AgentSpec:
    """One selectable sub-agent profile exposed through MCP tool arguments."""

    id: str
    description: str
    capability: str
    allowed_tools: tuple[str, ...]
    model: str | None = None


BUILT_IN_AGENTS = (
    AgentSpec(
        id="coding",
        description=(
            "Full coding sub-agent: reads, writes, edits, runs bounded Bash, searches"
        ),
        capability="coding",
        allowed_tools=tuple(DEFAULT_ALLOWED_TOOLS),
    ),
    AgentSpec(
        id="readonly",
        description=(
            "Read-only analysis sub-agent: reads and searches without mutating"
        ),
        capability="readonly",
        allowed_tools=READONLY_TOOLS,
    ),
)
DEFAULT_CATALOG = {agent.id: agent for agent in BUILT_IN_AGENTS}


def validate_model(value: object, field_name: str = "model") -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{field_name} must be a non-empty string")
    if value != value.strip() or any(ord(char) < 32 for char in value):
        raise RuntimeError(
            f"{field_name} must not contain surrounding or control whitespace"
        )
    return value


def _agent_tools(raw: object, capability: str, label: str) -> list[str]:
    if raw is None:
        return list(DEFAULT_TOOLS_BY_CAPABILITY[capability])
    if not isinstance(raw, list) or not all(isinstance(tool, str) for tool in raw):
        raise RuntimeError(f"{label} allowed_tools must be a list of tool names")
    unknown = sorted(set(raw) - KNOWN_TOOLS)
    if unknown:
        raise RuntimeError(
            f"{label} allowed_tools contains unknown tools: {', '.join(unknown)}"
        )
    if len(raw) != len(set(raw)):
        raise RuntimeError(f"{label} allowed_tools must not contain duplicates")
    return list(raw)


def _parse_agent(entry: object, index: int) -> AgentSpec:
    label = f"agents[{index}]"
    if not isinstance(entry, dict):
        raise RuntimeError(f"{label} must be an object")
    agent_id = entry.get("id")
    if not isinstance(agent_id, str) or not AGENT_ID_PATTERN.match(agent_id):
        raise RuntimeError(
            f"{label} id must be a slug matching ^[a-z][a-z0-9-]{{1,31}}$"
        )
    description = entry.get("description")
    if not isinstance(description, str) or not description.strip():
        raise RuntimeError(f"{label} '{agent_id}' description must be non-empty")
    if len(description) > MAX_DESCRIPTION_CHARS:
        raise RuntimeError(
            f"{label} '{agent_id}' description must be at most "
            f"{MAX_DESCRIPTION_CHARS} characters"
        )
    capability = entry.get("capability")
    if capability not in AGENT_CAPABILITIES:
        raise RuntimeError(
            f"{label} '{agent_id}' capability must be one of: "
            + ", ".join(AGENT_CAPABILITIES)
        )
    agent_label = f"{label} '{agent_id}'"
    tools = _agent_tools(entry.get("allowed_tools"), capability, agent_label)
    model = entry.get("model")
    if model is not None:
        model = validate_model(model, f"{label} '{agent_id}' model")
    mutations = sorted(MUTATION_TOOLS.intersection(tools))
    if mutations and capability != "coding":
        raise RuntimeError(
            f"{label} '{agent_id}' capability is '{capability}' but allowed_tools "
            f"include mutation tools: {', '.join(mutations)}; mutation tools "
            "require capability 'coding'"
        )
    return AgentSpec(
        id=agent_id,
        description=description.strip(),
        capability=capability,
        allowed_tools=tuple(tools),
        model=model,
    )


def parse_agents(raw: object) -> tuple[AgentSpec, ...]:
    """Validate the ``agents`` config list and return built-ins plus user agents."""
    if raw is None:
        return BUILT_IN_AGENTS
    if not isinstance(raw, list):
        raise RuntimeError("agents must be a list of agent objects")
    catalog = {agent.id: agent for agent in BUILT_IN_AGENTS}
    for index, entry in enumerate(raw):
        agent = _parse_agent(entry, index)
        if agent.id in catalog:
            raise RuntimeError(
                f"agents[{index}] id '{agent.id}' conflicts with an existing agent id"
            )
        catalog[agent.id] = agent
    return tuple(catalog.values())


def catalog_from_config(config: object) -> dict[str, AgentSpec]:
    agents = getattr(config, "agents", None) or BUILT_IN_AGENTS
    return {agent.id: agent for agent in agents}


def resolve_agent(
    catalog: Mapping[str, AgentSpec], agent_id: str, default_id: str
) -> AgentSpec:
    """Resolve an explicit or default agent id, listing available ids on failure."""
    target = agent_id or default_id
    agent = catalog.get(target)
    if agent is None:
        available = ", ".join(sorted(catalog))
        raise RuntimeError(
            f"unknown agent '{target}'; available agents: {available}"
        )
    return agent


def bind_agent(
    config: Any, agent_id: str, api_capability: str, tool_model: str = "flash"
) -> Any:
    """Apply an agent's capability/tools/model to a per-call config.

    Precedence for the provider model: explicit ``model`` tool arg > agent model
    > config default. The agent model applies only when the host left the tool
    arg at its default ``flash`` choice.
    """
    agent = resolve_agent(catalog_from_config(config), agent_id, api_capability)
    if agent.capability != api_capability:
        raise RuntimeError(
            f"agent '{agent.id}' has capability '{agent.capability}' but this API "
            f"requires capability '{api_capability}'"
        )
    updated = replace(
        config,
        delegation_capability=agent.capability,
        allowed_tools=list(agent.allowed_tools),
        active_agent=agent.id,
    )
    if agent.model is not None and tool_model == "flash":
        updated.model = agent.model
    return updated


def render_catalog_docstring(
    catalog: Mapping[str, AgentSpec] | None = None,
) -> str:
    """Render a compact ``id: description`` list sorted by agent id."""
    specs: Iterable[AgentSpec] = (catalog or DEFAULT_CATALOG).values()
    return "\n".join(
        f"{agent.id}: {agent.description}"
        for agent in sorted(specs, key=lambda spec: spec.id)
    )


MODEL_PRECEDENCE_LINE = (
    "Model precedence: explicit model argument > agent model > config default "
    "(an agent model applies only when the model argument is left at its default)."
)


def tool_docstring(
    description: str, catalog: Mapping[str, AgentSpec] | None = None
) -> str:
    """Build one delegation-tool docstring with the catalog and precedence note."""
    return (
        f"{description}\n\nAgents:\n{render_catalog_docstring(catalog)}"
        f"\n\n{MODEL_PRECEDENCE_LINE}"
    )


def register_agent_tool(
    mcp_tool: Any, annotations: Any, description: str, fn: Any
) -> Any:
    """Register an MCP tool with the catalog embedded in its docstring."""
    fn.catalog_description = description
    fn.__doc__ = tool_docstring(description)
    return mcp_tool(annotations=annotations)(fn)


def refresh_tool_docstrings(config: object, functions: Iterable) -> None:
    """Overwrite delegation docstrings with the config catalog; never raises.

    A missing or broken config leaves the built-in-only docstrings in place.
    """
    if config is None:
        try:
            from .config import Config

            config = Config.load()
        except Exception:
            config = None
    try:
        catalog = catalog_from_config(config)
    except Exception:
        try:
            catalog = {agent.id: agent for agent in parse_agents(getattr(config, "agents", None))}
        except Exception:
            catalog = DEFAULT_CATALOG
    for function in functions:
        description = getattr(function, "catalog_description", None)
        if not isinstance(description, str) or not description:
            description = "DeepSeek delegation sub-agent."
        function.__doc__ = tool_docstring(description, catalog)
