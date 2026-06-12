import logging
import os
from dataclasses import dataclass, field


@dataclass
class Config:
    """Agent 全局配置"""

    # 模型提供商: "ollama" | "deepseek" | "openai"
    provider: str = "deepseek"
    model: str = "deepseek-chat"
    temperature: float = 0.0
    max_tokens: int = 8192
    max_search_results: int = 10
    log_level: int = logging.INFO

    # DeepSeek / OpenAI 兼容 API 配置
    deepseek_api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    deepseek_base_url: str = "https://api.deepseek.com/v1"


config = Config()


def setup_logging(level: int | None = None) -> None:
    """初始化日志系统"""
    logging.basicConfig(
        level=level or config.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
