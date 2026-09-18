from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from unittest.mock import patch

from mcp.server.fastmcp import FastMCP

from deepseek_mcp.agent_catalog import (
    MODEL_PRECEDENCE_LINE,
    parse_agents,
    refresh_registered_tool_descriptions,
    register_agent_tool,
)
from deepseek_mcp.config import Config

ROOT = Path(__file__).resolve().parents[1]


def _served_description(tools, name: str) -> str:
    for tool in tools:
        if tool.name == name:
            return tool.description or ""
    raise AssertionError(f"tool {name} was not registered")


class AgentDocstringTests(unittest.TestCase):
    def _config(self) -> Config:
        config = Config("sk-test", ROOT, allowed_tools=["Read"])
        config.agents = parse_agents(
            [
                {
                    "id": "reviewer",
                    "description": "Reads only",
                    "capability": "readonly",
                }
            ]
        )
        return config

    def _server(self):
        server = FastMCP("agent-docstring-test")

        def delegate() -> str:
            return "x"

        register_agent_tool(server.tool, None, "Run a delegation.", delegate)
        return server, delegate

    def test_refresh_updates_the_served_tool_description(self) -> None:
        server, delegate = self._server()
        name = delegate.__name__
        before = _served_description(asyncio.run(server.list_tools()), name)
        self.assertNotIn("reviewer", before)

        refresh_registered_tool_descriptions(server, (delegate,), self._config())

        after = _served_description(asyncio.run(server.list_tools()), name)
        self.assertIn("reviewer", after)
        self.assertIn("Reads only", after)
        self.assertIn(MODEL_PRECEDENCE_LINE, after)

    def test_refresh_failure_keeps_the_builtin_description(self) -> None:
        server, delegate = self._server()
        name = delegate.__name__

        with patch(
            "deepseek_mcp.config.Config.load",
            side_effect=RuntimeError("no config"),
        ):
            refresh_registered_tool_descriptions(server, (delegate,), None)

        description = _served_description(asyncio.run(server.list_tools()), name)
        self.assertIn("coding:", description)
        self.assertIn(MODEL_PRECEDENCE_LINE, description)
        self.assertNotIn("reviewer", description)

    def test_missing_tool_manager_is_a_silent_noop(self) -> None:
        class _FakeServer:
            pass

        def delegate() -> str:
            return "x"

        delegate.catalog_description = "Run a delegation."

        refresh_registered_tool_descriptions(
            _FakeServer(), (delegate,), self._config()
        )

    def test_register_agent_tool_embeds_precedence(self) -> None:
        server, delegate = self._server()

        self.assertEqual(delegate.catalog_description, "Run a delegation.")
        self.assertIn(MODEL_PRECEDENCE_LINE, delegate.__doc__)


if __name__ == "__main__":
    unittest.main()
