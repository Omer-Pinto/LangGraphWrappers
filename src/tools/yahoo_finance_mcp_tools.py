import os

from tools_wrapper import ToolsWrapper
from langchain_mcp_adapters.client import MultiServerMCPClient


class YahooFinanceMCPTools(ToolsWrapper):
    def __init__(self):
        super().__init__()
        self.client = None

    async def setup(self):
        self.client = MultiServerMCPClient(
            {
                "yahoo_finance": {
                    "command": "uv",
                    "args": ["run", "--active", "./server.py"],
                    "transport": "stdio",
                    "cwd": "./mcps/yahoo_finance_mcp",
                    "env": {"UV_PYTHON": "3.12", **os.environ},
                }
            }
        )
        self.tools = await self.client.get_tools()

    async def cleanup(self):
        for name, session in getattr(self.client, "clients", {}).items():
            try:
                await session.close()
            except Exception as e:
                pass