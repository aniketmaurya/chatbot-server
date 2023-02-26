"""Wrapper around Lightning App."""
import logging
from typing import Any, Dict, List, Mapping, Optional

from pydantic import BaseModel

from langchain.llms.base import LLM
import requests

logger = logging.getLogger(__name__)


class LitServer(LLM, BaseModel):
    url: str = ""

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Run the LLM on the given prompt and input."""
        if self.url =="":
            raise Exception("Server URL not set!")
        response =  requests.get(url=self.url, data={"prompt": prompt})
        response.raise_for_status()
        return response.json()["text"]
    
    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "Lightning"
