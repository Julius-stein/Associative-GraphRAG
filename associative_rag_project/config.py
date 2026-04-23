"""Project-local configuration loader.

Generation and embedding credentials are inherited from the existing LightRAG
config so this project can reuse the same API setup.

加载本项目的模型与嵌入配置，支持环境变量覆盖。
"""

import os
from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _env_int(name, default):
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got: {raw}") from exc


def _load_scheme4_config():
    """Load the shared OpenAI-compatible config from the original repo.

    从上层 `llm_config.py` 中读取通用生成与嵌入配置。
    """
    module_path = Path(__file__).resolve().parents[1] / "llm_config.py"
    spec = spec_from_file_location("assoc_llm_config", module_path)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module.SCHEME4_LLM_CONFIG


def load_llm_config():
    """Load generation + embedding config, letting env vars override secrets.

    加载 LLM 配置，并允许环境变量覆盖关键参数。
    """
    config = deepcopy(_load_scheme4_config())
    env_name = config.get("api_key_env")
    if env_name and os.getenv(env_name):
        config["api_key"] = os.getenv(env_name)
    config["embedding_provider"] = (
        os.getenv("LIGHTRAG_EMBED_PROVIDER")
        or os.getenv("OPENAI_EMBEDDING_PROVIDER")
        or config.get("embedding_provider")
        or "openai_compatible"
    )
    config["embedding_model"] = (
        os.getenv("LIGHTRAG_EMBED_MODEL")
        or os.getenv("OPENAI_EMBEDDING_MODEL")
        or config.get("embedding_model")
        or "text-embedding-3-small"
    )
    config["embedding_base_url"] = (
        os.getenv("LIGHTRAG_EMBED_BASE_URL")
        or os.getenv("OPENAI_EMBEDDING_BASE_URL")
        or config.get("embedding_base_url")
        or config.get("base_url")
    )
    config["embedding_api_key"] = (
        os.getenv("LIGHTRAG_EMBED_API_KEY")
        or os.getenv("OPENAI_EMBEDDING_API_KEY")
        or config.get("embedding_api_key")
        or config.get("api_key")
        or "EMPTY"
    )
    if "embedding_dim" not in config:
        raw_dim = os.getenv("LIGHTRAG_EMBED_DIM") or os.getenv("OPENAI_EMBEDDING_DIM") or os.getenv("EMBEDDING_DIM")
        config["embedding_dim"] = int(raw_dim) if raw_dim else 1536
    else:
        raw_dim = os.getenv("LIGHTRAG_EMBED_DIM") or os.getenv("OPENAI_EMBEDDING_DIM") or os.getenv("EMBEDDING_DIM")
        config["embedding_dim"] = int(raw_dim) if raw_dim else int(config["embedding_dim"])
    config["embedding_batch_size"] = _env_int("LIGHTRAG_EMBED_BATCH_SIZE", int(config.get("embedding_batch_size", 16)))
    return config


def load_judge_config():
    """Use a separate judge model so generation and evaluation can diverge.

    为评分过程加载独立模型配置，避免与生成阶段共用同一模型。
    """
    config = load_llm_config()
    config["model"] = os.getenv("OPENAI_JUDGE_MODEL", "gpt-5.4-mini")
    config["max_concurrency"] = int(os.getenv("OPENAI_JUDGE_MAX_CONCURRENCY", str(config.get("max_concurrency", 4))))
    return config
