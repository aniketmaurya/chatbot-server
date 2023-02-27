import lightning as L
import lightning.app.frontend as frontend

from chatserver.llm_serve import LLMServe
from chatserver.ui import main


class ChatBotApp(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.llm_serve = LLMServe()
        self.llm_url = ""

    def run(self):
        self.llm_serve.run()
        if self.llm_serve.url:
            self.llm_url = self.llm_serve.url

    def configure_layout(self):
        return frontend.StreamlitFrontend(render_fn=main.run)


app = L.LightningApp(ChatBotApp())
