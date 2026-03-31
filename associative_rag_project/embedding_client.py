"""Thin embedding client with a small in-memory query cache."""

import numpy as np
from openai import OpenAI

from .config import load_llm_config


class OpenAICompatibleEmbeddingClient:
    """Wrapper around an OpenAI-compatible embeddings endpoint."""

    def __init__(self, llm_config=None):
        self.config = llm_config or load_llm_config()
        self.client = OpenAI(
            api_key=self.config["embedding_api_key"],
            base_url=self.config["embedding_base_url"],
            timeout=self.config.get("timeout", 120),
            max_retries=self.config.get("max_retries", 3),
        )
        self.model = self.config["embedding_model"]
        self._cache = {}

    def embed_text(self, text):
        """Embed one query string; repeated calls are cached during a run."""
        if text in self._cache:
            return self._cache[text]
        response = self.client.embeddings.create(model=self.model, input=[text])
        vector = np.asarray(response.data[0].embedding, dtype=np.float32)
        self._cache[text] = vector
        return vector
