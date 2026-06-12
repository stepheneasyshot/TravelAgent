#!/usr/bin/env python3
"""CustomAgent Web GUI — 基于 FastAPI + WebSocket 的聊天界面

启动方式:
    python gui.py
    然后浏览器打开 http://127.0.0.1:8000
"""

import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from langchain_core.messages import HumanMessage

from src.agent import create_agent
from src.config import config, setup_logging

setup_logging()
log = logging.getLogger("gui")
agent = create_agent()
app = FastAPI(title="CustomAgent")


@app.get("/")
async def index():
    return HTMLResponse(HTML_TEMPLATE)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    log.info("WebSocket 客户端已连接")
    await ws.send_json({
        "type": "status",
        "content": f"已连接 | 模型: {config.model}",
    })

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "content": "消息格式错误"})
                continue

            if msg.get("type") != "chat":
                continue

            user_input = msg.get("content", "").strip()
            if not user_input:
                continue

            log.info("用户: %s", user_input)

            try:
                tool_outputs = []

                async for event in agent.astream_events(
                    {"messages": [HumanMessage(content=user_input)]},
                    version="v2",
                ):
                    kind = event["event"]

                    if kind == "on_chat_model_stream":
                        chunk = event["data"]["chunk"]
                        content = chunk.content
                        if content and isinstance(content, str):
                            await ws.send_json({"type": "token", "content": content})

                    elif kind == "on_tool_start":
                        name = event["name"]
                        args = event["data"].get("input", {})
                        summary = _tool_summary(name, args)
                        await ws.send_json({
                            "type": "tool_call",
                            "name": name,
                            "summary": summary,
                        })

                    elif kind == "on_tool_end":
                        output = event["data"].get("output", "")
                        name = event["name"]
                        tool_outputs.append({"name": name, "output": str(output) if output else ""})
                        await ws.send_json({
                            "type": "tool_result",
                            "name": name,
                        })

            except Exception as e:
                log.error("Agent 运行出错: %s", e, exc_info=True)
                await ws.send_json({"type": "error", "content": str(e)})

            if tool_outputs:
                refs = _build_references(tool_outputs)
                if refs:
                    await ws.send_json({"type": "references", "refs": refs})

            await ws.send_json({"type": "done"})

    except WebSocketDisconnect:
        log.info("WebSocket 客户端断开连接")
    except Exception as e:
        log.error("WebSocket 严重错误: %s", e, exc_info=True)


def _tool_summary(name: str, args: dict) -> str:
    """生成工具调用的简短描述，用于前端轻量指示器"""
    if name == "web_search":
        query = args.get("query", "")
        return f"正在搜索: {query}" if query else "正在搜索..."
    if name == "fetch_webpage":
        url = args.get("url", "")
        short = url[:60] + "..." if len(url) > 60 else url
        return f"正在抓取: {short}" if url else "正在抓取网页..."
    if name == "get_current_time":
        return "正在获取时间..."
    return f"正在调用 {name}..."


def _build_references(tool_outputs: list[dict]) -> list[dict]:
    """从工具输出中提取参考来源列表"""
    refs = []
    for item in tool_outputs:
        name = item["name"]
        output = item["output"]
        if name == "web_search":
            refs.extend(_parse_search_results(output))
        elif name == "fetch_webpage":
            m = re.search(r'Webpage content \((https?://[^)]+)\)', output)
            url = m.group(1) if m else ""
            if url:
                refs.append({"title": url, "url": url, "source": "网页抓取"})
    return refs


def _parse_search_results(output: str) -> list[dict]:
    """解析搜索输出为结构化参考列表"""
    refs = []
    lines = output.split("\n")
    current_title = ""
    current_url = ""
    for line in lines:
        title_match = re.match(r"^\d+\.\s+(.+)", line)
        url_match = re.match(r"\s*URL:\s*(.+)", line)
        if title_match:
            if current_title and current_url:
                refs.append({"title": current_title, "url": current_url})
            current_title = title_match.group(1).strip()
            current_url = ""
        elif url_match and current_title:
            current_url = url_match.group(1).strip()
    if current_title and current_url:
        refs.append({"title": current_title, "url": current_url})
    return refs


# ============================================================
# HTML / CSS / JavaScript 前端模板
# ============================================================

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CustomAgent — 本地 AI 助手</title>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root {
            --bg-base: #0d1117;
            --bg-surface: #161b22;
            --bg-elevated: #21262d;
            --bg-input: #1c2128;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --text-muted: #6e7681;
            --accent: #58a6ff;
            --accent-hover: #4090f0;
            --success: #3fb950;
            --warning: #d29922;
            --error: #f85149;
            --border: #30363d;
            --user-bubble: #1f6feb;
            --ai-bubble: #21262d;
            --tool-card-bg: #1a1f27;
            --tool-border: #d29922;
            --font: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", sans-serif;
            --font-mono: "SF Mono", "Fira Code", "JetBrains Mono", monospace;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: var(--font);
            background: var(--bg-base);
            color: var(--text-primary);
            height: 100vh;
            overflow: hidden;
        }

        .app {
            display: grid;
            grid-template-columns: 1fr 320px;
            height: 100vh;
            max-width: 1400px;
            margin: 0 auto;
        }

        /* ---- 主面板 (聊天) ---- */
        .main-panel {
            display: grid;
            grid-template-rows: auto 1fr auto;
            overflow: hidden;
            border-right: 1px solid var(--border);
        }

        .chat-header {
            padding: 1rem 1.25rem;
            background: var(--bg-surface);
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .chat-header .logo {
            width: 34px;
            height: 34px;
            border-radius: 10px;
            background: linear-gradient(135deg, var(--accent), #a371f7);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.1rem;
            font-weight: 700;
            color: #fff;
        }

        .chat-header .title { font-weight: 600; font-size: 1.05rem; }
        .chat-header .subtitle { font-size: 0.8rem; color: var(--text-secondary); }

        .chat-messages {
            overflow-y: auto;
            padding: 1.25rem;
            display: flex;
            flex-direction: column;
            gap: 0.875rem;
            scroll-behavior: smooth;
        }

        .chat-messages::-webkit-scrollbar { width: 6px; }
        .chat-messages::-webkit-scrollbar-track { background: transparent; }
        .chat-messages::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

        /* 消息气泡 */
        .message {
            max-width: 78%;
            padding: 0.75rem 1rem;
            border-radius: 1rem;
            line-height: 1.55;
            font-size: 0.925rem;
            animation: fadeIn 0.25s ease;
            word-break: break-word;
        }

        .message.user {
            align-self: flex-end;
            background: var(--user-bubble);
            color: #fff;
            border-bottom-right-radius: 0.25rem;
        }

        .message.ai {
            align-self: flex-start;
            background: var(--ai-bubble);
            color: var(--text-primary);
            border: 1px solid var(--border);
            border-bottom-left-radius: 0.25rem;
        }

        .message.error {
            align-self: flex-start;
            background: #2d1219;
            color: var(--error);
            border: 1px solid #5c1f28;
            font-size: 0.85rem;
        }

        /* ---- Markdown 渲染样式 (.message.ai 内部) ---- */
        .message.ai h1, .message.ai h2, .message.ai h3,
        .message.ai h4, .message.ai h5, .message.ai h6 {
            margin: 0.75rem 0 0.4rem 0;
            line-height: 1.3;
            font-weight: 600;
        }
        .message.ai h1:first-child, .message.ai h2:first-child,
        .message.ai h3:first-child, .message.ai h4:first-child { margin-top: 0; }
        .message.ai h1 { font-size: 1.35rem; border-bottom: 1px solid var(--border); padding-bottom: 0.3rem; }
        .message.ai h2 { font-size: 1.15rem; }
        .message.ai h3 { font-size: 1.05rem; }
        .message.ai h4 { font-size: 0.95rem; color: var(--text-secondary); }

        .message.ai p { margin: 0.35rem 0; }
        .message.ai p:first-child { margin-top: 0; }
        .message.ai p:last-child { margin-bottom: 0; }

        .message.ai strong, .message.ai b { font-weight: 600; color: #f0f6fc; }
        .message.ai em, .message.ai i { font-style: italic; }

        .message.ai a {
            color: var(--accent);
            text-decoration: none;
            border-bottom: 1px solid transparent;
            transition: border-color 0.15s;
        }
        .message.ai a:hover { border-bottom-color: var(--accent); }

        .message.ai code {
            font-family: var(--font-mono);
            font-size: 0.85em;
            background: var(--bg-surface);
            padding: 0.15em 0.35em;
            border-radius: 0.25rem;
            color: #f2a65a;
        }

        .message.ai pre {
            margin: 0.5rem 0;
            background: #0d1117;
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            overflow-x: auto;
        }
        .message.ai pre code {
            display: block;
            padding: 0.75rem 0.875rem;
            background: transparent;
            color: var(--text-primary);
            font-size: 0.82rem;
            line-height: 1.5;
            border-radius: 0;
        }

        .message.ai ul, .message.ai ol {
            margin: 0.35rem 0;
            padding-left: 1.5rem;
        }
        .message.ai li { margin: 0.15rem 0; }
        .message.ai li::marker { color: var(--text-muted); }

        .message.ai blockquote {
            margin: 0.5rem 0;
            padding: 0.4rem 0.75rem;
            border-left: 3px solid var(--accent);
            background: var(--bg-elevated);
            border-radius: 0 0.375rem 0.375rem 0;
            color: var(--text-secondary);
        }

        .message.ai table {
            margin: 0.5rem 0;
            border-collapse: collapse;
            font-size: 0.82rem;
            width: 100%;
        }
        .message.ai th, .message.ai td {
            padding: 0.4rem 0.625rem;
            border: 1px solid var(--border);
            text-align: left;
        }
        .message.ai th {
            background: var(--bg-elevated);
            font-weight: 600;
        }
        .message.ai tr:nth-child(even) td {
            background: rgba(255,255,255,0.02);
        }

        .message.ai hr {
            margin: 0.6rem 0;
            border: none;
            border-top: 1px solid var(--border);
        }

        .message.ai img {
            max-width: 100%;
            border-radius: 0.5rem;
            margin: 0.35rem 0;
        }

        /* 流式输出光标闪烁 */
        .message.ai.streaming::after {
            content: '';
            display: inline-block;
            width: 7px;
            height: 15px;
            background: var(--accent);
            margin-left: 1px;
            vertical-align: text-bottom;
            animation: blink-cursor 0.8s infinite;
        }

        @keyframes blink-cursor {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0; }
        }

        /* ---- 搜索状态指示器 (轻量) ---- */
        .search-indicator {
            align-self: flex-start;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.45rem 0.75rem;
            font-size: 0.8rem;
            color: var(--text-secondary);
            background: var(--bg-surface);
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            animation: fadeIn 0.2s ease;
        }

        .search-indicator .si-dot {
            width: 8px; height: 8px;
            border-radius: 50%;
            background: var(--warning);
            animation: pulse-dot 1.2s infinite;
        }

        .search-indicator.done .si-dot {
            background: var(--success);
            animation: none;
        }

        /* ---- 参考来源 ---- */
        .references {
            align-self: flex-start;
            max-width: 85%;
            margin-top: -0.25rem;
            font-size: 0.8rem;
        }

        .references details {
            background: var(--bg-surface);
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            overflow: hidden;
        }

        .references summary {
            padding: 0.5rem 0.75rem;
            cursor: pointer;
            color: var(--text-secondary);
            font-weight: 500;
            user-select: none;
            transition: color 0.15s;
        }

        .references summary:hover { color: var(--text-primary); }

        .references ul {
            list-style: none;
            padding: 0 0.75rem 0.6rem 0.75rem;
            margin: 0;
        }

        .references li {
            padding: 0.35rem 0;
            border-top: 1px solid var(--border);
            line-height: 1.4;
        }

        .references li:first-child { border-top: none; }

        .references a {
            color: var(--accent);
            text-decoration: none;
            font-size: 0.82rem;
            word-break: break-all;
        }

        .references a:hover { text-decoration: underline; }

        .references .ref-title {
            display: block;
            color: var(--text-primary);
            font-size: 0.8rem;
            margin-bottom: 0.1rem;
        }

        /* 正在输入动画 */
        .typing-indicator {
            align-self: flex-start;
            display: flex;
            gap: 4px;
            padding: 0.75rem 1rem;
            background: var(--ai-bubble);
            border: 1px solid var(--border);
            border-radius: 1rem;
            border-bottom-left-radius: 0.25rem;
        }

        .typing-indicator span {
            width: 7px; height: 7px;
            background: var(--text-muted);
            border-radius: 50%;
            animation: bounce 1.4s infinite ease-in-out both;
        }

        .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
        .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
        .typing-indicator span:nth-child(3) { animation-delay: 0s; }

        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0.4); }
            40% { transform: scale(1); }
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(6px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        /* 空状态 */
        .empty-state {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: var(--text-muted);
            text-align: center;
            gap: 0.5rem;
        }
        .empty-state .welcome-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
        .empty-state h2 { font-weight: 500; font-size: 1.2rem; color: var(--text-secondary); }
        .empty-state p { font-size: 0.85rem; max-width: 280px; }

        /* 输入区域 */
        .chat-input-area {
            padding: 0.875rem 1.25rem;
            background: var(--bg-surface);
            border-top: 1px solid var(--border);
            display: flex;
            gap: 0.625rem;
            align-items: center;
        }

        .chat-input-area input {
            flex: 1;
            padding: 0.7rem 0.875rem;
            background: var(--bg-input);
            border: 1px solid var(--border);
            border-radius: 0.75rem;
            color: var(--text-primary);
            font-size: 0.925rem;
            font-family: var(--font);
            outline: none;
            transition: border-color 0.2s;
        }

        .chat-input-area input:focus { border-color: var(--accent); }
        .chat-input-area input::placeholder { color: var(--text-muted); }

        .chat-input-area button {
            padding: 0.7rem 1.25rem;
            background: var(--accent);
            color: #fff;
            border: none;
            border-radius: 0.75rem;
            font-size: 0.9rem;
            font-weight: 600;
            font-family: var(--font);
            cursor: pointer;
            transition: background 0.2s, opacity 0.2s;
            white-space: nowrap;
        }

        .chat-input-area button:hover { background: var(--accent-hover); }
        .chat-input-area button:disabled { opacity: 0.45; cursor: not-allowed; }

        /* ---- 侧面板 (运行状态) ---- */
        .side-panel {
            display: grid;
            grid-template-rows: auto 1fr;
            overflow: hidden;
            background: var(--bg-surface);
        }

        .side-header {
            padding: 1rem 1.25rem;
            border-bottom: 1px solid var(--border);
            font-weight: 600;
            font-size: 0.9rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .side-header .dot {
            width: 8px; height: 8px;
            border-radius: 50%;
            background: var(--success);
            animation: pulse-dot 2s infinite;
        }

        @keyframes pulse-dot {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }

        .status-log {
            overflow-y: auto;
            padding: 0.75rem;
            display: flex;
            flex-direction: column;
            gap: 2px;
        }

        .status-log::-webkit-scrollbar { width: 4px; }
        .status-log::-webkit-scrollbar-track { background: transparent; }
        .status-log::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

        .status-entry {
            padding: 0.35rem 0.5rem;
            border-radius: 0.3rem;
            font-size: 0.78rem;
            font-family: var(--font-mono);
            animation: fadeIn 0.15s ease;
            display: flex;
            gap: 0.5rem;
            line-height: 1.4;
        }

        .status-entry .time {
            color: var(--text-muted);
            flex-shrink: 0;
        }
        .status-entry .text { flex: 1; }
        .status-entry.info    { color: var(--text-secondary); }
        .status-entry.tool    { color: var(--warning); }
        .status-entry.success { color: var(--success); }
        .status-entry.error   { color: var(--error); background: rgba(248,81,73,0.08); }

        /* 响应式 */
        @media (max-width: 800px) {
            .app { grid-template-columns: 1fr; grid-template-rows: 1fr auto; }
            .side-panel {
                max-height: 200px;
                border-top: 1px solid var(--border);
            }
            .message { max-width: 90%; }
        }
    </style>
</head>
<body>
    <div class="app">
        <!-- 主面板：聊天 -->
        <div class="main-panel">
            <div class="chat-header">
                <div class="logo">智</div>
                <div>
                    <div class="title">CustomAgent</div>
                    <div class="subtitle">本地 AI · 联网搜索</div>
                </div>
            </div>

            <div class="chat-messages" id="chat-messages">
                <div class="empty-state" id="empty-state">
                    <div class="welcome-icon">&#129302;</div>
                    <h2>你好，我是小智</h2>
                    <p>基于本地大模型的 AI 助手，支持联网搜索。尽管问我任何问题。</p>
                </div>
            </div>

            <div class="chat-input-area">
                <input type="text" id="chat-input" placeholder="输入你的问题…" autofocus>
                <button id="send-btn" onclick="sendMessage()">发送</button>
            </div>
        </div>

        <!-- 侧面板：运行状态 -->
        <div class="side-panel">
            <div class="side-header">
                <div class="dot"></div> 运行状态
            </div>
            <div class="status-log" id="status-log">
                <div class="status-entry info">
                    <span class="time">--:--</span>
                    <span class="text">等待连接…</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        // ---- 状态 ----
        let isSending = false;
        let currentSearchIndicator = null;
        let collectedRefs = [];
        let currentStreamBubble = null;
        let currentStreamText = '';
        let ws = null;
        let reconnectTimer = null;

        // ---- DOM 引用 ----
        const chatMessages = document.getElementById('chat-messages');
        const emptyState = document.getElementById('empty-state');
        const statusLog = document.getElementById('status-log');
        const chatInput = document.getElementById('chat-input');
        const sendBtn = document.getElementById('send-btn');

        // ---- WebSocket 连接 ----
        function connectWS() {
            if (ws && ws.readyState === WebSocket.OPEN) return;
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(protocol + '//' + location.host + '/ws');

            ws.onopen = function() {
                if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
                addStatus('已连接', 'success');
            };

            ws.onmessage = function(event) {
                try {
                    handleMessage(JSON.parse(event.data));
                } catch (e) {
                    console.error('消息解析失败:', e);
                }
            };

            ws.onclose = function() {
                addStatus('连接断开，3秒后重连…', 'error');
                reconnectTimer = setTimeout(connectWS, 3000);
            };

            ws.onerror = function() {
                addStatus('连接出错', 'error');
            };
        }

        // ---- 消息处理 ----
        function handleMessage(data) {
            switch (data.type) {
                case 'status':
                    addStatus(data.content, 'info');
                    break;

                case 'token':
                    hideTyping();
                    hideSearchIndicator();
                    appendStreamToken(data.content);
                    break;

                case 'tool_call':
                    finalizeStreamBubble();
                    showSearchIndicator(data.summary);
                    addStatus(data.summary, 'tool');
                    break;

                case 'tool_result':
                    hideSearchIndicator();
                    addStatus('完成: ' + (TOOL_LABELS[data.name] || data.name), 'success');
                    break;

                case 'references':
                    addReferences(data.refs);
                    break;

                case 'message':
                    hideTyping();
                    finalizeStreamBubble();
                    addAIBubble(data.content);
                    addStatus('已回复', 'info');
                    break;

                case 'error':
                    hideTyping();
                    finalizeStreamBubble();
                    hideSearchIndicator();
                    addErrorBubble(data.content);
                    addStatus('错误: ' + data.content, 'error');
                    isSending = false;
                    updateSendBtn();
                    break;

                case 'done':
                    finalizeStreamBubble();
                    hideSearchIndicator();
                    hideTyping();
                    isSending = false;
                    updateSendBtn();
                    collectedRefs = [];
                    break;
            }
            scrollAll();
        }

        // ---- 发送消息 ----
        function sendMessage() {
            if (isSending) return;
            const text = chatInput.value.trim();
            if (!text) return;

            if (ws.readyState !== WebSocket.OPEN) {
                addStatus('未连接，正在重连…', 'error');
                connectWS();
                return;
            }

            hideEmptyState();
            addUserBubble(text);
            showTyping();
            isSending = true;
            updateSendBtn();
            currentSearchIndicator = null;
            collectedRefs = [];
            currentStreamBubble = null;
            currentStreamText = '';

            ws.send(JSON.stringify({ type: 'chat', content: text }));
            chatInput.value = '';
            chatInput.focus();
        }

        // ---- DOM 操作 ----
        function hideEmptyState() {
            if (emptyState) emptyState.style.display = 'none';
        }

        function addUserBubble(text) {
            var div = document.createElement('div');
            div.className = 'message user';
            div.textContent = text;
            chatMessages.appendChild(div);
        }

        function addAIBubble(text) {
            var div = document.createElement('div');
            div.className = 'message ai';
            var html = marked.parse(text);
            div.innerHTML = html;
            div.querySelectorAll('a').forEach(function(a) {
                a.setAttribute('target', '_blank');
                a.setAttribute('rel', 'noopener noreferrer');
            });
            chatMessages.appendChild(div);
        }

        function appendStreamToken(token) {
            if (!currentStreamBubble) {
                currentStreamBubble = document.createElement('div');
                currentStreamBubble.className = 'message ai streaming';
                chatMessages.appendChild(currentStreamBubble);
                currentStreamText = '';
            }
            currentStreamText += token;
            var html = marked.parse(currentStreamText);
            currentStreamBubble.innerHTML = html;
        }

        function finalizeStreamBubble() {
            if (!currentStreamBubble) return;
            currentStreamBubble.classList.remove('streaming');
            currentStreamBubble.querySelectorAll('a').forEach(function(a) {
                a.setAttribute('target', '_blank');
                a.setAttribute('rel', 'noopener noreferrer');
            });
            currentStreamBubble = null;
            currentStreamText = '';
        }

        function addErrorBubble(text) {
            var div = document.createElement('div');
            div.className = 'message error';
            div.textContent = '错误: ' + text;
            chatMessages.appendChild(div);
        }

        var TOOL_ICONS = {
            'web_search': '&#128270;',
            'fetch_webpage': '&#127760;',
            'get_current_time': '&#128340;',
        };
        var TOOL_LABELS = {
            'web_search': '百度搜索',
            'fetch_webpage': '抓取网页',
            'get_current_time': '获取时间',
        };

        function showSearchIndicator(summary) {
            hideSearchIndicator();
            var div = document.createElement('div');
            div.className = 'search-indicator';
            div.id = 'search-ind';
            div.innerHTML = '<span class="si-dot"></span><span>' + escapeHtml(summary) + '</span>';
            chatMessages.appendChild(div);
            currentSearchIndicator = div;
        }

        function hideSearchIndicator() {
            if (currentSearchIndicator) {
                currentSearchIndicator.classList.add('done');
                currentSearchIndicator = null;
            }
            var el = document.getElementById('search-ind');
            if (el && el !== currentSearchIndicator) el.remove();
        }

        function addReferences(refs) {
            if (!refs || refs.length === 0) return;
            var container = document.createElement('div');
            container.className = 'references';
            var details = document.createElement('details');
            var summary = document.createElement('summary');
            summary.textContent = '参考来源 · ' + refs.length + ' 条';
            details.appendChild(summary);

            var ul = document.createElement('ul');
            refs.forEach(function(ref) {
                var li = document.createElement('li');
                if (ref.title && ref.title !== ref.url) {
                    var span = document.createElement('span');
                    span.className = 'ref-title';
                    span.textContent = ref.title;
                    li.appendChild(span);
                }
                if (ref.url) {
                    var a = document.createElement('a');
                    a.href = ref.url;
                    a.target = '_blank';
                    a.rel = 'noopener noreferrer';
                    a.textContent = ref.url;
                    li.appendChild(a);
                }
                ul.appendChild(li);
            });
            details.appendChild(ul);
            container.appendChild(details);
            chatMessages.appendChild(container);
        }

        function showTyping() {
            hideTyping();
            var div = document.createElement('div');
            div.className = 'typing-indicator';
            div.id = 'typing';
            div.innerHTML = '<span></span><span></span><span></span>';
            chatMessages.appendChild(div);
        }

        function hideTyping() {
            var el = document.getElementById('typing');
            if (el) el.remove();
        }

        function addStatus(text, level) {
            var now = new Date();
            var time = now.getHours().toString().padStart(2, '0') + ':' +
                       now.getMinutes().toString().padStart(2, '0') + ':' +
                       now.getSeconds().toString().padStart(2, '0');
            var entry = document.createElement('div');
            entry.className = 'status-entry ' + (level || 'info');
            entry.innerHTML = '<span class="time">' + time + '</span>' +
                              '<span class="text">' + escapeHtml(text) + '</span>';
            // 插入到最前面
            if (statusLog.firstChild) {
                statusLog.insertBefore(entry, statusLog.firstChild);
            } else {
                statusLog.appendChild(entry);
            }
            // 最多保留 100 条
            while (statusLog.children.length > 100) {
                statusLog.removeChild(statusLog.lastChild);
            }
        }

        function updateSendBtn() {
            sendBtn.disabled = isSending;
            sendBtn.textContent = isSending ? '思考中…' : '发送';
        }

        function scrollAll() {
            chatMessages.scrollTop = chatMessages.scrollHeight;
            statusLog.scrollTop = 0;  // 最新状态在顶部
        }

        function escapeHtml(text) {
            var div = document.createElement('div');
            div.appendChild(document.createTextNode(text));
            return div.innerHTML;
        }

        // ---- 键盘事件 ----
        chatInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        // ---- 初始化 ----
        marked.setOptions({ breaks: true, gfm: true });
        connectWS();
        chatInput.focus();
    </script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    print(f"""
╔══════════════════════════════════════════╗
║         CustomAgent Web GUI             ║
║                                         ║
║  打开浏览器访问: http://127.0.0.1:8000  ║
║  按 Ctrl+C 停止服务                      ║
╚══════════════════════════════════════════╝
    """)
    uvicorn.run("gui:app", host="127.0.0.1", port=8000, reload=False)
