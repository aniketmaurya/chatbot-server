from lightning.app.components import PythonServer, Text
import lightning as L
from pydantic import BaseModel, Field

from typing import Any

class PromptSchema(BaseModel):
    prompt: str = Field(title="Your msg to chatbot", max_length=300, min_length=1)

class LLMServer(PythonServer):
    def __init__(self, **kwargs):
        super().__init__(input_type=PromptSchema, output_type=Text, **kwargs)

    def setup(self, *args, **kwargs) -> None:
        from chatserver.llm import load_hf_llm

        self._model = load_hf_llm()

    def predict(self, request: PromptSchema) -> Any:
        return {"text": self._model(request.prompt)}

app = L.LightningApp(LLMServer())
