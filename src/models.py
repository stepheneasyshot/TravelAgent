import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from .config import Config, config

log = logging.getLogger(__name__)


def create_llm(cfg: Config | None = None) -> BaseChatModel:
    """LLM 工厂：根据 provider 返回对应的聊天模型。

    支持:
    - ollama: 本地模型，通过 ChatOllama
    - deepseek: DeepSeek 云端 API，通过 ChatOpenAI（OpenAI 兼容）
    - openai: 任意 OpenAI 兼容 API
    """
    if cfg is None:
        cfg = config

    if cfg.provider == "ollama":
        log.info("创建 ChatOllama(model=%s, temperature=%.2f, num_predict=%d)",
                 cfg.model, cfg.temperature, cfg.max_tokens)
        return ChatOllama(
            model=cfg.model,
            temperature=cfg.temperature,
            num_predict=cfg.max_tokens,
        )

    if cfg.provider in ("deepseek", "openai"):
        model_name = cfg.model if cfg.provider == "openai" else "deepseek-chat"

        # ChatOpenAI 传空字符串会跳过 env var 回退，None 才会触发
        api_key = cfg.deepseek_api_key or None
        if api_key is None:
            raise ValueError(
                "DeepSeek API Key 未设置。请设置环境变量 DEEPSEEK_API_KEY，"
                "或在 src/config.py 中配置 deepseek_api_key。"
            )

        log.info("创建 ChatOpenAI(model=%s, base_url=%s)", model_name, cfg.deepseek_base_url)
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=cfg.deepseek_base_url,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

    raise ValueError(f"不支持的 provider: {cfg.provider}")


def create_llm_with_structured_output(
    schema: type,
    cfg: Config | None = None,
    method: str = "json_mode",
) -> BaseChatModel:
    """创建带结构化输出的 LLM。

    对支持 JSON mode 的 provider（deepseek/openai）使用 json_mode，
    对 ollama 使用 json_mode 兜底。
    """
    if cfg is None:
        cfg = config

    llm = create_llm(cfg)

    if cfg.provider == "ollama":
        # Ollama 不支持原生 json_mode，使用 json_schema 兜底
        log.info("使用 with_structured_output(schema=%s, method=json_schema)", schema.__name__)
        return llm.with_structured_output(schema, method="json_schema")

    log.info("使用 with_structured_output(schema=%s, method=%s)", schema.__name__, method)
    return llm.with_structured_output(schema, method=method)
