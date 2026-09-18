from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from deepseek_mcp import agent_catalog
from deepseek_mcp.agent_catalog import (
    BUILT_IN_AGENTS,
    DEFAULT_ALLOWED_TOOLS,
    DEFAULT_CATALOG,
    AgentSpec,
    bind_agent,
    catalog_from_config,
    parse_agents,
    render_catalog_docstring,
    resolve_agent,
)
from deepseek_mcp.config import Config


def _entry(agent_id: str, capability: str = "coding", **extra) -> dict:
    entry = {
        "id": agent_id,
        "description": f"{agent_id} agent",
        "capability": capability,
    }
    entry.update(extra)
    return entry


class AgentCatalogTests(unittest.TestCase):
    def _config(self, agents: list | None = None) -> Config:
        tmpdir = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, tmpdir, True)
        data: dict = {"workspace": tmpdir, "allowed_tools": ["Read"]}
        if agents is not None:
            data["agents"] = agents
        return Config._from_data(data, "credential")

    def test_builtins_are_the_default_catalog(self) -> None:
        self.assertEqual(parse_agents(None), BUILT_IN_AGENTS)
        self.assertEqual(set(DEFAULT_CATALOG), {"coding", "readonly"})
        self.assertEqual(
            DEFAULT_CATALOG["coding"].allowed_tools, tuple(DEFAULT_ALLOWED_TOOLS)
        )
        self.assertEqual(
            DEFAULT_CATALOG["readonly"].allowed_tools, ("Read", "Glob", "Grep")
        )

    def test_parse_rejects_bad_ids_and_clashes(self) -> None:
        cases = (
            ([_entry("coding")], "conflicts"),
            ([_entry("rev"), _entry("rev")], "conflicts"),
            ([_entry("Bad")], "slug"),
            ([{"id": "rev", "description": "x", "capability": "nope"}], "capability"),
            ([{"id": "rev", "description": "", "capability": "readonly"}], "non-empty"),
            ([_entry("rev", allowed_tools=["Read", "Nope"])], "unknown tools"),
        )
        for agents, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RuntimeError, message):
                    parse_agents(agents)

    def test_parse_rejects_mutation_tools_for_readonly(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Write") as raised:
            parse_agents(
                [_entry("rev", "readonly", allowed_tools=["Read", "Write"])]
            )
        self.assertIn("coding", str(raised.exception))

    def test_parse_fills_capability_defaults_and_model(self) -> None:
        specs = parse_agents(
            [
                _entry("rev", "readonly", model="deepseek-v4-pro"),
                _entry("worker", "coding"),
            ]
        )
        catalog = {spec.id: spec for spec in specs}
        self.assertEqual(catalog["rev"].allowed_tools, ("Read", "Glob", "Grep"))
        self.assertEqual(catalog["rev"].model, "deepseek-v4-pro")
        self.assertEqual(
            catalog["worker"].allowed_tools, tuple(DEFAULT_ALLOWED_TOOLS)
        )

    def test_resolve_agent_defaults_and_lists_available(self) -> None:
        catalog = DEFAULT_CATALOG
        self.assertEqual(resolve_agent(catalog, "", "readonly").id, "readonly")
        with self.assertRaisesRegex(RuntimeError, "available agents"):
            resolve_agent(catalog, "ghost", "coding")

    def test_render_catalog_docstring_is_sorted(self) -> None:
        rendered = render_catalog_docstring(
            {
                "zeta": AgentSpec("zeta", "last", "coding", ()),
                "alpha": AgentSpec("alpha", "first", "readonly", ()),
            }
        )
        self.assertEqual(rendered, "alpha: first\nzeta: last")

    def test_bind_agent_applies_capability_tools_and_model(self) -> None:
        config = self._config(
            [_entry("reviewer", "readonly", model="deepseek-v4-pro")]
        )
        bound = bind_agent(config, "reviewer", "readonly", "flash")

        self.assertEqual(bound.delegation_capability, "readonly")
        self.assertEqual(set(bound.allowed_tools), {"Read", "Glob", "Grep"})
        self.assertEqual(bound.active_agent, "reviewer")
        self.assertEqual(bound.model, "deepseek-v4-pro")

    def test_bind_agent_model_precedence(self) -> None:
        config = self._config(
            [_entry("reviewer", "readonly", model="deepseek-v4-pro")]
        )
        config.model = "default-model"

        self.assertEqual(
            bind_agent(config, "reviewer", "readonly", "flash").model,
            "deepseek-v4-pro",
        )
        self.assertEqual(
            bind_agent(config, "reviewer", "readonly", "pro").model,
            "default-model",
        )

    def test_bind_agent_rejects_capability_contradiction(self) -> None:
        config = self._config([_entry("reviewer", "readonly")])
        with self.assertRaisesRegex(RuntimeError, "readonly.*coding"):
            bind_agent(config, "reviewer", "coding", "flash")

    def test_bind_agent_unknown_id_is_transparent(self) -> None:
        config = self._config()
        with self.assertRaisesRegex(RuntimeError, "available agents"):
            bind_agent(config, "ghost", "coding", "flash")

    def test_catalog_from_config_uses_configured_agents(self) -> None:
        config = self._config([_entry("reviewer", "readonly")])
        catalog = catalog_from_config(config)
        self.assertIn("reviewer", catalog)
        self.assertIn("coding", catalog)


if __name__ == "__main__":
    unittest.main()
