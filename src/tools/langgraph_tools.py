import os

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from tools.tools_wrapper import ToolsWrapper
from langchain_community.agent_toolkits import FileManagementToolkit
import requests
from langchain.agents import Tool

class FileManagementToolkitWrapper(ToolsWrapper):
    def __init__(self, root_dir: str):
        super().__init__("file_management")
        self.tools = FileManagementToolkit(root_dir=root_dir).get_tools()

class PushArgs(BaseModel):
    text: str = Field(..., description="The message body of the notification.")
    title: str = Field(..., description="The title of the notification.")


class PushNotificationTool(ToolsWrapper):
    PUSHOVER_URL = "https://api.pushover.net/1/messages.json"
    def __init__(self):
        super().__init__("push_notification")
        self.pushover_token = os.getenv("PUSHOVER_TOKEN")
        self.pushover_user = os.getenv("PUSHOVER_USER")

        self.tools = [
                        StructuredTool.from_function(
                            name="send_push_notification",
                            func=self.push,
                            description="Use this tool when you want to send a push notification - using text and title",
                            args_schema=PushArgs,),
                     ]

    def push(self, text: str, title: str) -> None:
        """
        Send a push notification to the user using text and title.
        Args:
            text: str
                The text of the push notification.
            title : str
                the title of the push notification.
        """
        requests.post(self.PUSHOVER_URL, data={"token": self.pushover_token, "user": self.pushover_user, "message": text, "title": title})