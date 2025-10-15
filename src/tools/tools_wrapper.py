from abc import ABC, abstractmethod

class ToolsWrapper:
    def __init__(self, name: str):
        self.name = name
        self.tools = []

    async def setup(self):
        """Optional async setup hook (override if needed)."""
        pass

    async def cleanup(self):
        """Optional async teardown hook (override if needed)."""
        pass