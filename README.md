# LangGraphWrappers

### Lightweight DSL & Wrapper Layer Around [LangGraph](https://github.com/langchain-ai/langgraph)

LangGraphWrappers is a small Python library designed to make working with **LangGraph** simpler, cleaner, and less repetitive.  
It abstracts away the verbose and sometimes complex initialization steps required to define and wire together **LangGraph nodes, edges, models, and tools**, letting you focus purely on your **business logic**.

This project provides a compact **Domain-Specific Language (DSL)** to declaratively describe graphs and their components, plus helpers for structured outputs, model and tool initialization, and integration with MCP servers.

---

## 🌟 Features

- **Concise DSL for Nodes and Edges**  
  Simplifies definition and linking of LangGraph components.
  
- **Unified Wrappers for Models, Tools, and Nodes**  
  - `ModelWrapper` for consistent initialization and schema integration  
  - `NodeWrapper` for easy node creation and attachment  
  - `ToolsWrapper` for tool lifecycle management (setup/teardown)
  
- **Graph Composition Made Simple**  
  Define graphs using a single `GraphWrapper`, attaching nodes and edges declaratively.

- **Structured Outputs (Pydantic)**  
  Integrate strongly-typed responses with `BaseModel` schemas.

- **Integrated Tool Management**  
  Built-in wrappers for tools like:
  - LangChain’s `FileManagementToolkit`
  - Custom notification tools (e.g., Pushbullet/Pushover)
  - MCP server utilities (under `tools/mcps/`)

- **Async Setup & Cleanup**  
  Hooks to initialize or teardown connections (e.g., MCP servers or remote APIs).

---

## 📁 Repository Structure

```
src/
├── graph/
│   ├── graph_wrapper.py       # High-level Graph creation and connection management
│   └── nodes_wrappers.py      # Node wrappers and base classes for node definitions
├── messages/
│   └── control_message.py     # A new type of message for control messaging between nodes using state as their data channel
├── models/
│   ├── model_wrapper.py       # Handles ChatOpenAI + LangGraph model setup and schemas
│   └── models.py              # Model definitions, URLs, and API key retrieval helpers
├── tools/
│   ├── langgraph_tools.py     # Tool wrappers (e.g. FileManagementToolkit, PushNotification)
│   ├── tools_wrapper.py       # Base class for all tool wrappers
│   └── mcps/                  # Example MCP tool server (Yahoo Finance)
│       └── yahoo_finance_mcp/
│           ├── server.py      # MCP server implementation
│           ├── pyproject.toml # Independent MCP server setup
│           └── README.md      # Additional documentation for this mcp server
```

---

## 🚀 Quick Start

### 1. Installation


```bash
uv sync
```

---

### 2. Define a Model

```python
from models.model_wrapper import ModelWrapper
from models.models import Model
# a simple model. See below for more advanced setup of tools and/or structured outputs
model = ModelWrapper(model=Model.GPT_4O, name="primary-model")
```

---

### 3. Create Nodes

```python
from graph.nodes_wrappers import NodeWrapper

async def my_action(state):
    # perform inference or logic here
    return {"result": "Hello, World!"}

node = NodeWrapper(name="hello_node", action=my_action, model_wrapper=model)
```

---

### 4. Compose the Graph

```python
from graph.graph_wrapper import GraphWrapper, EdgeType

graph = GraphWrapper(
    state_type=dict,
    models=[model],
    nodes=[node],
    edges_info=[("hello_node", EdgeType.EDGE, ["end_node"])]
)
```

---

### 5. Add Tools (Optional)

```python
from tools.langgraph_tools import FileManagementToolkitWrapper

file_tools = FileManagementToolkitWrapper(root_dir="/tmp")
await file_tools.setup()
# use file_tools.tools inside nodes or agents
```

---

### 6. Integrate Structured Outputs

You can associate **Pydantic schemas** with a model for validated responses:

```python
from pydantic import BaseModel

class OutputSchema(BaseModel):
    result: str

model = ModelWrapper(model=Model.GPT_4O, name="schema-model", schema=OutputSchema)
```

---

## ⚙️ Internals Overview

| Component | Purpose                                                                                                                                                |
|------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| `GraphWrapper` | Simplifies LangGraph graph construction with nodes, edges, and state types.                                                                            |
| `NodeWrapper` | Wraps async callables into LangGraph nodes, injecting the associated model automatically, possibly attaching a router function(also a LangGraph node). 
| `ModelWrapper` | Wraps LangChain models (e.g., `ChatOpenAI`) and configures temperature, tools, and schema integration.                                                 |
| `ToolsWrapper` | Manages reusable toolkits (e.g., file management, MCP servers, notification systems), including handle their setup/teardown.                           |
| `langgraph_tools.py` | Prebuilt wrappers for LangChain community tools and Push notification integration (to be extended and solidified).                                     |

---

## 💡 Special Note

- I wanted to separate the construction of the graph (as a data structure) from the "node actions" - the async function 
each node should call once activated by LangGraph infrastructure.
- In that case, on a project that uses this infrastructure, I used a client-side graph manager that helps me build the graph,
providing GraphWrapper with what it needs to initiate (node wrappers, edge info, model wrappers, tools).
- In addition I have a node_actions file with all node actions. Because these are "detached" from graph code, they don't have
the model available to call from them.
- The model is being handled in the graph. The actions are passed to the graph as part of node wrappers definitions.
- I decided to "plant" the model within the action itself, so each node wrapper that defines a model wrapper, and an action, 
within its action method, model could be accessed as `func_name.__model__` as in:
```python
async def hedge_fund_manager(state: State) -> Dict[str, Any]:
    # set hedge fund manager LangGraph messages (system, ai, user)
    ....

    # Invoke the func-injected LLM
    response = await hedge_fund_manager.__model__.ainvoke(hedge_fund_manager_messages)
    return {
        "messages": [ControlMessage(content="hedge fund manager made an investment decision successfully")],
        "investment_decision": response
    }
```

---

## 🧩 Example: Building a graph with nodes, edges, tools & more

- You should look at the code under **src /market_analyst_graph** folder in https://github.com/Omer-Pinto/StocksMarketRecommender.
- It shows a basic graph construction using the infrastructure layer code implemented in this repository.

---

## 🧠 Philosophy

This library isn’t meant to **replace LangGraph** — it’s meant to **wrap it**.  
The goal is **clarity and brevity**, making complex LangGraph setups look more like **declarative configuration** than boilerplate.

---

## 📜 License

MIT License © 2025 [Omer Pinto](https://github.com/Omer-Pinto)

---

## 🧭 Future Directions

- Yet to be determined. Depending on how much production systems I can built using LangGraph & LangSmith (absolutely love these)..
