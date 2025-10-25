from typing import List, Optional, Literal, Type, Dict, Any
from models.models import Model, ModelUrl, get_base_url_for_model, get_api_key_for_model
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langchain_core.tools.base import BaseTool

class ModelWrapper:
    def __init__(self,
                 model:Model,
                 name: str,
                 method: Literal["function_calling", "json_mode", "json_schema"] = "json_schema",
                 temperature: float = 0.0,
                 tools: Optional[List[BaseTool]]=None,
                 schema: Optional[Type[BaseModel]] = None,):
        self.name = name
        kwargs: Dict[Any, Any] = {"temperature": temperature}
        base_url = get_base_url_for_model(model)
        if base_url:
            kwargs['base_url'] = base_url
        api_key = get_api_key_for_model(model)
        if api_key:
            kwargs['api_key'] = api_key
        temp = ChatOpenAI(model=model, **kwargs)
        if tools:
            self.model = temp.bind_tools(tools)
        elif schema:
            self.model = temp.with_structured_output(schema=schema, method=method)
        else:
            self.model = temp