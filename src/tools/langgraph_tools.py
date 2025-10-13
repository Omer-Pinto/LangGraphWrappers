import os

from tools_wrapper import ToolsWrapper
from langchain_community.agent_toolkits import FileManagementToolkit
import requests

class FileManagementToolkitWrapper(ToolsWrapper):
    def __init__(self, root_dir: str):
        super().__init__()
        self.tools = FileManagementToolkit(root_dir=root_dir).get_tools()

class PushNotificationTool(ToolsWrapper):
    PUSHOVER_URL = "https://api.pushover.net/1/messages.json"
    def __init__(self, pushover_token: str, pushover_user: str):
        super().__init__()
        self.pushover_token = os.getenv("PUSHOVER_TOKEN")
        self.pushover_user = os.getenv("PUSHOVER_USER")

    def push(self, text: str) -> None:
        """Send a push notification to the user"""
        requests.post(self.PUSHOVER_URL, data={"token": self.pushover_token, "user": self.pushover_user, "message": text})