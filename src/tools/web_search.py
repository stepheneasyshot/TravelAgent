import logging

from langchain_core.tools import tool
from web_search_mcp import MultiSourceSearcher
from src.config import config

log = logging.getLogger(__name__)

DEFAULT_SOURCES = ["baidu", "sogou"]


@tool
def web_search(query: str, max_results: int = config.max_search_results) -> str:
    """通过百度搜索互联网，自动故障转移到搜狗，返回结果的标题、链接和摘要。

    适用场景：查找实时新闻、近期事件、事实信息、股票价格等需要最新数据的场景。
    如果搜索结果摘要信息不足，可使用 fetch_webpage 工具获取页面全文。
    """
    log.info("web_search: query=%r, max_results=%d", query, max_results)
    try:
        searcher = MultiSourceSearcher()
        response = searcher.search(
            query=query,
            max_results=max_results,
            sources=DEFAULT_SOURCES,
        )
        if response.error:
            return f"搜索失败: {response.error}"
        if not response.results:
            return "未找到相关搜索结果，请尝试使用其他关键词。"
        return response.to_text()
    except Exception as e:
        log.error("web_search 失败: %s", e)
        return f"搜索失败: {str(e)}"


@tool
def fetch_webpage(url: str, max_length: int = 3000) -> str:
    """获取指定网页的文本内容。当搜索结果摘要不够详细时，用此工具阅读全文。

    参数 url: 网页链接地址
    参数 max_length: 返回的最大字符数，默认 3000
    """
    log.info("fetch_webpage: url=%r", url)
    try:
        import httpx
        from lxml import html as lxml_html

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            )
        }
        resp = httpx.get(url, headers=headers, timeout=15, follow_redirects=True)
        resp.raise_for_status()

        tree = lxml_html.fromstring(resp.content)
        for tag in tree.xpath("//script | //style | //nav | //footer | //header"):
            tag.getparent().remove(tag)

        text = tree.text_content()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        content = "\n".join(lines)

        if len(content) > max_length:
            content = content[:max_length] + f"\n\n... (内容已截断，共 {len(content)} 字符)"

        log.info("fetch_webpage: 成功获取 %d 字符", len(content))
        return content
    except Exception as e:
        log.error("fetch_webpage 失败: %s", e)
        return f"获取网页失败: {str(e)}"
