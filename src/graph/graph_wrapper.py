import uuid
from pathlib import Path
from typing import List, Type, Tuple, cast
from langchain_core.tools import BaseTool

from models.model_wrapper import ModelWrapper
from graph.nodes_wrappers import BaseNodeWrapper, NodeWrapper
from langgraph.typing import StateT
from langgraph.graph import StateGraph
from enum import Enum
from langgraph.checkpoint.memory import MemorySaver

class EdgeType(str, Enum):
    EDGE = "->"
    CONDITIONAL_EDGE = "-?>"

class GraphWrapper:
    def __init__(self,
                 state_type: Type[StateT],
                 models: List[ModelWrapper],
                 nodes: List[BaseNodeWrapper],
                 edges_info: List[Tuple[str, EdgeType, List[str]]]
                 ):
        self.models = models
        self.nodes = nodes
        self.edges_info = edges_info
        self.state_type = state_type
        self.graph = None
        self.memory = MemorySaver()
        self.sidekick_id = str(uuid.uuid4())

    def _get_router(self, node_name: str):
        for node in self.nodes:
            if node.name == node_name:
                if not isinstance(node, NodeWrapper):
                    raise Exception(f"Node {node_name} must be an instance of NodeWrapper in order to have a router")
                return cast(NodeWrapper, node).router
        raise Exception(f"Node {node_name} not found")


    def get_graph_drawing(self):
        mermaid = self.graph.get_graph().draw_mermaid()
        mermaid = mermaid.replace("&nbsp;", " ").replace("<p>", "").replace("</p>", "")
        Path("graph.mmd").write_text(mermaid, encoding="utf-8")
        print("Saved to graph.mmd — open it in mermaid.live to verify synt.")

    def build_and_compile_graph(self):
        graph_builder = StateGraph(self.state_type)

        # Add nodes
        for node in self.nodes:
            graph_builder.add_node(**node.get_node_metadata())
        for source, arrow, target_list in self.edges_info:
                if arrow == EdgeType.EDGE:
                    for target in target_list:
                        graph_builder.add_edge(source, target)
                else:
                    graph_builder.add_conditional_edges(source, self._get_router(source), target_list)

        # Compile the graph
        self.graph = graph_builder.compile(checkpointer=self.memory)

    async def run_superstep(self, initial_state: StateT) -> StateT:
        config = {"configurable": {"thread_id": self.sidekick_id}}
        result = await self.graph.ainvoke(initial_state, config=config)
        return result
