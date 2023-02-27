import lightning as L
import lightning.app.frontend as frontend

from chatserver.components import LLMServe
from chatserver.ui import ui_render_fn


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
        return frontend.StreamlitFrontend(render_fn=ui_render_fn)


app = L.LightningApp(ChatBotApp())
