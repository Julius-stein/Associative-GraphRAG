"""Embedding clients for dense retrieval.

提供向量化查询的两种后端：OpenAI 兼容 API 和本地 BGE 模型。
"""

from __future__ import annotations

import os
from typing import Iterable

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
        response = self.client.embeddings.create(model=self.model, input=[text])
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
            response = self.client.embeddings.create(model=self.model, input=missing)
            ordered_items = sorted(response.data, key=lambda item: item.index)
            for text, item in zip(missing, ordered_items):
                self._cache[text] = np.asarray(item.embedding, dtype=np.float32)
        return [self._cache[text] for text in texts]


class LocalBGEM3EmbeddingClient:
    """Local dense embedding client backed by Hugging Face `BAAI/bge-m3`.

    本地推理嵌入客户端，支持 MPS/CUDA/CPU 设备调度。
    """

    def __init__(self, llm_config=None):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Local BGE-M3 embedding requires `torch` and `transformers`. "
                "Install them, then rerun with --embedding-provider bge_m3_local."
            ) from exc
        self._torch = torch
        self.config = llm_config or load_llm_config()
        self.model_name = self.config.get("local_embedding_model", "BAAI/bge-m3")
        self.batch_size = int(self.config.get("local_embedding_batch_size", 16))
        self.query_instruction = (self.config.get("local_embedding_query_instruction") or "").strip()
        self.max_length = int(self.config.get("local_embedding_max_length", 8192))
        self.device = self._resolve_device(self.config.get("local_embedding_device", "auto"))

        # Allow fully offline reuse when model weights are already cached.
        local_only = os.getenv("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes"}
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            local_files_only=local_only,
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            local_files_only=local_only,
        )
        self.model.to(self.device)
        self.model.eval()
        self._cache = {}

    def _resolve_device(self, raw_device):
        device = (raw_device or "auto").lower()
        torch = self._torch
        if device != "auto":
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-12)
        return summed / counts

    def _normalize_inputs(self, texts: Iterable[str]):
        prefix = f"{self.query_instruction} " if self.query_instruction else ""
        return [prefix + str(text) for text in texts]

    def _embed_uncached_texts(self, texts):
        prepared = self._normalize_inputs(texts)
        vectors = []
        with self._torch.inference_mode():
            for start in range(0, len(prepared), self.batch_size):
                batch = prepared[start : start + self.batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                outputs = self.model(**encoded)
                pooled = self._mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
                normalized = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.append(normalized.detach().cpu().to(self._torch.float32).numpy())
        if not vectors:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack(vectors)

    def embed_text(self, text):
        """Embed one query string; repeated calls are cached during a run."""
        if text in self._cache:
            return self._cache[text]
        vector = self._embed_uncached_texts([text])[0]
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
            vectors = self._embed_uncached_texts(missing)
            for text, vector in zip(missing, vectors):
                self._cache[text] = vector
        return [self._cache[text] for text in texts]


def build_embedding_client(llm_config=None):
    """Instantiate the configured embedding provider.

    根据配置选择 OpenAI 兼容或本地 BGE 模型嵌入客户端。
    """
    config = llm_config or load_llm_config()
    provider = (config.get("embedding_provider") or "openai_compatible").strip().lower()
    if provider in {"openai_compatible", "openai", "api"}:
        return OpenAICompatibleEmbeddingClient(config)
    if provider in {"bge_m3_local", "bge-m3-local", "local_bge_m3"}:
        return LocalBGEM3EmbeddingClient(config)
    raise ValueError(
        f"Unsupported embedding provider '{provider}'. "
        "Expected one of: openai_compatible, bge_m3_local"
    )
