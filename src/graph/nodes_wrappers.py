from typing import Callable, Any, Dict, List, Optional, Awaitable
from langchain_core.tools.base import BaseTool
from langgraph.prebuilt import ToolNode
from langgraph.typing import StateT
from models.model_wrapper import ModelWrapper


class BaseNodeWrapper:
    def __init__(self, name: str):
        self.name = name
    def get_node_metadata(self) -> Dict[str, Any]:
        pass

class NodeWrapper(BaseNodeWrapper):
    def __init__(self, name: str,
                 action: Callable[[StateT], Awaitable[Dict[str, Any]]],
                 model_wrapper: ModelWrapper,
                 router: Optional[Callable[[Any], Any]]= None):
        super().__init__(name)
        self.action = action
        self.model_wrapper = model_wrapper
        self.router = router
        # inject model into node action
        self.action.__model__ = model_wrapper.model

    def get_node_metadata(self) -> Dict[str, Any]:
        return {"node": self.name, "action": self.action}

class PythonNodeWrapper(BaseNodeWrapper):
    def __init__(self, name: str,
                 action: Callable[[StateT], Awaitable[Dict[str, Any]]]):
        super().__init__(name)
        self.action = action

    def get_node_metadata(self) -> Dict[str, Any]:
        return {"node": self.name, "action": self.action}

class ToolNodeWrapper(BaseNodeWrapper):
    def __init__(self, name: str, tools: List[BaseTool]):
        super().__init__(name)
        self.tools = tools

    def get_node_metadata(self) -> Dict[str, Any]:
        return {"node": self.name, "action": ToolNode(tools=self.tools)}
