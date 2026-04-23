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
    config.setdefault("embedding_provider", "openai_compatible")
    config.setdefault("embedding_model", os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    config.setdefault("embedding_base_url", os.getenv("OPENAI_EMBEDDING_BASE_URL", config.get("base_url")))
    config.setdefault("embedding_api_key", os.getenv("OPENAI_EMBEDDING_API_KEY", config.get("api_key")))
    if "embedding_dim" not in config:
        raw_dim = os.getenv("OPENAI_EMBEDDING_DIM") or os.getenv("EMBEDDING_DIM")
        config["embedding_dim"] = int(raw_dim) if raw_dim else 1536
    else:
        config["embedding_dim"] = int(config["embedding_dim"])
    config.setdefault("local_embedding_model", os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-m3"))
    config.setdefault("local_embedding_device", os.getenv("LOCAL_EMBEDDING_DEVICE", "auto"))
    config.setdefault("local_embedding_batch_size", _env_int("LOCAL_EMBEDDING_BATCH_SIZE", 16))
    config.setdefault("local_embedding_max_length", _env_int("LOCAL_EMBEDDING_MAX_LENGTH", 8192))
    config.setdefault("local_embedding_query_instruction", os.getenv("LOCAL_EMBEDDING_QUERY_INSTRUCTION", ""))
    return config


def load_judge_config():
    """Use a separate judge model so generation and evaluation can diverge.

    为评分过程加载独立模型配置，避免与生成阶段共用同一模型。
    """
    config = load_llm_config()
    config["model"] = os.getenv("OPENAI_JUDGE_MODEL", "gpt-5.4-mini")
    config["max_concurrency"] = int(os.getenv("OPENAI_JUDGE_MAX_CONCURRENCY", str(config.get("max_concurrency", 4))))
    return config
