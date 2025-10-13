from typing import List, Optional, Literal
from models.models import Model
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langchain_core.tools.base import BaseTool

class ModelWrapper:
    def __init__(self,
                 model:Model,
                 name: str,
                 method: Literal["function_calling", "json_mode", "json_schema"] = "json_schema",
                 tools: Optional[List[BaseTool]]=None,
                 schema: Optional[BaseModel] = None,):
        self.name = name
        temp = ChatOpenAI(model=model)
        if tools:
            self.model = temp.bind_tools(tools)
        elif schema:
            self.model = temp.with_structured_output(schema=schema, method=method)
        else:
            self.model = temp