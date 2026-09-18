from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from deepseek_mcp.agent_catalog import (
    MODEL_PRECEDENCE_LINE,
    parse_agents,
    refresh_tool_docstrings,
    register_agent_tool,
)


def _mcp_tool(annotations=None):
    def decorate(fn):
        return fn

    return decorate


class AgentDocstringTests(unittest.TestCase):
    def _config(self) -> SimpleNamespace:
        return SimpleNamespace(
            agents=parse_agents(
                [
                    {
                        "id": "reviewer",
                        "description": "Reads only",
                        "capability": "readonly",
                    }
                ]
            )
        )

    def test_refresh_renders_user_agents_and_precedence(self) -> None:
        def tool() -> str:
            """old docstring"""
            return "x"

        refresh_tool_docstrings(self._config(), (tool,))

        self.assertIn("reviewer: Reads only", tool.__doc__)
        self.assertIn("Model precedence:", tool.__doc__)
        self.assertIn(MODEL_PRECEDENCE_LINE, tool.__doc__)

    def test_refresh_without_config_falls_back_to_builtins(self) -> None:
        def tool() -> str:
            """old docstring"""
            return "x"

        with patch(
            "deepseek_mcp.config.Config.load",
            side_effect=RuntimeError("no config"),
        ):
            refresh_tool_docstrings(None, (tool,))

        self.assertIn("coding:", tool.__doc__)
        self.assertIn("Model precedence:", tool.__doc__)
        self.assertNotIn("reviewer", tool.__doc__)

    def test_register_agent_tool_embeds_precedence(self) -> None:
        def tool() -> str:
            return "x"

        registered = register_agent_tool(
            _mcp_tool, None, "Run a coding delegation.", tool
        )

        self.assertIs(registered, tool)
        self.assertIn("Run a coding delegation.", tool.__doc__)
        self.assertIn("Model precedence:", tool.__doc__)
        self.assertEqual(tool.catalog_description, "Run a coding delegation.")


if __name__ == "__main__":
    unittest.main()
