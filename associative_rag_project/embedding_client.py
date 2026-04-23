"""Embedding client for dense retrieval.

Embedding is served through an OpenAI-compatible API. The runner process does
not load local BGE/transformer weights.
"""

from __future__ import annotations

import numpy as np
from openai import OpenAI

from .config import load_llm_config


class OpenAICompatibleEmbeddingClient:
    """Wrapper around an OpenAI-compatible embeddings endpoint.

    封装对 OpenAI 兼容 embeddings 接口的调用，并在本次运行中缓存结果。
    """

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
        response = self.client.embeddings.create(model=self.model, input=[text], encoding_format="float")
        vector = np.asarray(response.data[0].embedding, dtype=np.float32)
        self._cache[text] = vector
        return vector

    def embed_texts(self, texts):
        """Embed a batch of query strings while preserving order and cache hits."""
        texts = [str(text) for text in texts]
        missing = []
        seen_missing = set()
        for text in texts:
            if text not in self._cache and text not in seen_missing:
                missing.append(text)
                seen_missing.add(text)
        if missing:
            response = self.client.embeddings.create(model=self.model, input=missing, encoding_format="float")
            ordered_items = sorted(response.data, key=lambda item: item.index)
            for text, item in zip(missing, ordered_items):
                self._cache[text] = np.asarray(item.embedding, dtype=np.float32)
        return [self._cache[text] for text in texts]


def build_embedding_client(llm_config=None):
    """Instantiate the configured embedding provider.

    根据配置选择 OpenAI 兼容嵌入客户端。
    """
    config = llm_config or load_llm_config()
    provider = (config.get("embedding_provider") or "openai_compatible").strip().lower()
    if provider in {"openai_compatible", "openai", "api", "bge_m3_server", "bge-m3-server"}:
        return OpenAICompatibleEmbeddingClient(config)
    raise ValueError(
        f"Unsupported embedding provider '{provider}'. "
        "Expected an OpenAI-compatible server provider, e.g. openai_compatible or bge_m3_server."
    )
