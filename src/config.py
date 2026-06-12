import logging
from dataclasses import dataclass


@dataclass
class Config:
    """Agent 全局配置"""

    model: str = "qwen3.5:4b-mlx"
    temperature: float = 0.0
    max_tokens: int = 4096             # 最大生成 token 数 (关键：Ollama 默认仅 128!)
    max_search_results: int = 10
    log_level: int = logging.INFO


config = Config()


def setup_logging(level: int | None = None) -> None:
    """初始化日志系统"""
    logging.basicConfig(
        level=level or config.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
