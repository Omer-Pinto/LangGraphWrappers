from typing import Optional, Union, Any
from langchain_core.messages import BaseMessage

class ControlMessage(BaseMessage):
    type: str = "control"
    meta: Optional[dict] = None

    def __init__(
        self,
        content: Union[str, list[Union[str, dict]]],
        **kwargs: Any,
    ) -> None:
        """Initialize ``AIMessage``.

        Args:
            content: The content of the message.
            kwargs: Additional arguments to pass to the parent class.
        """
        super().__init__(content=content, **kwargs)