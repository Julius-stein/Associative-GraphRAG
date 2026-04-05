"""Project-local configuration loader.

Generation and embedding credentials are inherited from the existing LightRAG
config so this project can reuse the same API setup.
"""

import os
from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_scheme4_config():
    """Load the shared OpenAI-compatible config from the original repo."""
    module_path = Path(__file__).resolve().parents[1] / "llm_config.py"
    spec = spec_from_file_location("assoc_llm_config", module_path)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module.SCHEME4_LLM_CONFIG


def load_llm_config():
    """Load generation + embedding config, letting env vars override secrets."""
    config = deepcopy(_load_scheme4_config())
    env_name = config.get("api_key_env")
    if env_name and os.getenv(env_name):
        config["api_key"] = os.getenv(env_name)
    config.setdefault("embedding_model", os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    config.setdefault("embedding_base_url", os.getenv("OPENAI_EMBEDDING_BASE_URL", config.get("base_url")))
    config.setdefault("embedding_api_key", os.getenv("OPENAI_EMBEDDING_API_KEY", config.get("api_key")))
    return config


def load_judge_config():
    """Use a separate judge model so generation and evaluation can diverge."""
    config = load_llm_config()
    config["model"] = os.getenv("OPENAI_JUDGE_MODEL", "gpt-5.4-mini")
    config["max_concurrency"] = int(os.getenv("OPENAI_JUDGE_MAX_CONCURRENCY", str(config.get("max_concurrency", 4))))
    return config
