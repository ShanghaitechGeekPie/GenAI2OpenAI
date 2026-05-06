import argparse
import base64
import hashlib
import json
import logging
import mimetypes
import os
import uuid
from datetime import datetime
from urllib.parse import urlparse

import requests
from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS
from rich.console import Console
from rich.logging import RichHandler

app = Flask(__name__)
CORS(app)

# 解析命令行参数
parser = argparse.ArgumentParser(description='GenAI Flask API Server')
parser.add_argument('--token', type=str, default='eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE3NjMzNjA2MzgsInVzZXJuYW1lIjoiMjAyNDEzNDAyMiJ9.b4E5VzUxkn0Kc1pxkKVipybRFCw47NcppBognTD39e8',
                    help='GenAI API Access Token')
parser.add_argument('--upload-token', type=str, default='2ea38f293adb4abca21132feba61eaa3',
                    help='GenAI image upload API token header value')
parser.add_argument('--port', type=int, default=5000,
                    help='Flask server port (default: 5000)')
parser.add_argument('--log-level', type=str, default='INFO',
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                    help='Console log level (default: INFO)')
args = parser.parse_args()

console = Console()
logging.basicConfig(
    level=getattr(logging, args.log_level.upper(), logging.INFO),
    format='%(message)s',
    datefmt='[%X]',
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger('genai-proxy')

# 进程内图片去重缓存：image_sha256 -> {imageUrl, width, height}
IMAGE_UPLOAD_CACHE = {}

# GenAI API 配置
GENAI_URL = "https://genai.shanghaitech.edu.cn/htk/chat/start/chat"
GENAI_UPLOAD_URL = "https://genaipic.shanghaitech.edu.cn//sys/common/upload"
GENAI_IMAGE_STATIC_URL = "https://genaipic.shanghaitech.edu.cn//sys/common/static/"
GENAI_HEADERS = {
    "Accept": "*/*, text/event-stream",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Content-Type": "application/json",
    "Origin": "https://genai.shanghaitech.edu.cn",
    "Referer": "https://genai.shanghaitech.edu.cn/dialogue",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
    "X-Access-Token": args.token,
    "sec-ch-ua": '"Chromium";v="142", "Google Chrome";v="142", "Not_A Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
}

MODEL_SPECS = [
    {
        "public_id": "deepseek-r1",
        "request_id": "deepseek-r1:671b",
        "actual_id": "deepseek-r1:671b",
        "root_ai_type": "xinference",
    },
    {
        "public_id": "deepseek-v3",
        "request_id": "deepseek-v3:671b",
        "actual_id": "deepseek-v3:671b",
        "root_ai_type": "xinference",
    },
    {
        "public_id": "glm-5.1",
        "request_id": "chatglm",
        "actual_id": "glm-chat",
        "root_ai_type": "xinference",
    },
    {
        "public_id": "minimax-m1",
        "request_id": "MiniMax-M1",
        "actual_id": "minimax",
        "root_ai_type": "xinference",
    },
    {
        "public_id": "qwen3.5-397b-a17b",
        "request_id": "qwen-instruct",
        "actual_id": "qwen-instruct",
        "root_ai_type": "xinference",
    },
    {
        "public_id": "gpt-5.5",
        "request_id": "GPT-5.5",
        "actual_id": "gpt-5.5-2026-04-24",
        "root_ai_type": "azure",
    },
    {
        "public_id": "gpt-5.4",
        "request_id": "GPT-5.4",
        "actual_id": "gpt-5.4-2026-03-05",
        "root_ai_type": "azure",
    },
    {
        "public_id": "gpt-5.2",
        "request_id": "GPT-5.2",
        "actual_id": "gpt-5.2-2025-12-11",
        "root_ai_type": "azure",
    },
    {
        "public_id": "gpt-5",
        "request_id": "GPT-5",
        "actual_id": "gpt-5-2025-08-07",
        "root_ai_type": "azure",
    },
    {
        "public_id": "gpt-4.1",
        "request_id": "GPT-4.1",
        "actual_id": "gpt-4.1-2025-04-14",
        "root_ai_type": "azure",
    },
    {
        "public_id": "gpt-4.1-mini",
        "request_id": "GPT-4.1-mini",
        "actual_id": "gpt-4.1-mini-2025-04-14",
        "root_ai_type": "azure",
    },
    {
        "public_id": "gpt-o4-mini",
        "request_id": "o4-mini",
        "actual_id": "o4-mini-2025-04-16",
        "root_ai_type": "azure",
    },
    {
        "public_id": "gpt-o3",
        "request_id": "o3",
        "actual_id": "o3-2025-04-16",
        "root_ai_type": "azure",
    },
    {
        "public_id": "deepseek-pro",
        "request_id": "deepseek-pro",
        "actual_id": "deepseek-v4-pro",
        "root_ai_type": "xinference",
    },
    {
        "public_id": "deepseek-chat",
        "request_id": "deepseek-chat",
        "actual_id": "deepseek-v4-flash",
        "root_ai_type": "xinference",
    },
]


def build_model_alias_lookup():
    """构建模型别名查找表。

    将对外公开名称、上游请求名称和上游实际模型名称统一映射到同一份
    模型规格上，便于后续按任意别名解析。

    Returns:
        dict[str, dict]: 以小写别名为键、模型规格字典为值的查找表。
    """
    alias_lookup = {}
    for spec in MODEL_SPECS:
        for alias in {spec["public_id"], spec["request_id"], spec["actual_id"]}:
            alias_lookup[alias.lower()] = spec
    return alias_lookup


MODEL_ALIAS_LOOKUP = build_model_alias_lookup()


def resolve_model(model_name):
    """解析模型名称到上游请求参数。

    Args:
        model_name (Any): 调用方传入的模型名，可能是 public id、request id
            或 actual id。

    Returns:
        tuple[Any, str]: 第一个元素为实际发给上游的 `aiType`，第二个元素为
        `rootAiType`。
    """
    if not isinstance(model_name, str):
        return model_name, infer_root_ai_type(model_name)

    spec = MODEL_ALIAS_LOOKUP.get(model_name.lower())
    if spec is None:
        return model_name, infer_root_ai_type(model_name)
    return spec["request_id"], spec["root_ai_type"]


def get_request_access_token():
    """从 OpenAI 常用认证头中提取请求级 GenAI token。"""
    authorization = request.headers.get("Authorization", "")
    if authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
        if token:
            return token

    for header_name in ("X-Access-Token", "api-key", "X-API-Key"):
        token = request.headers.get(header_name)
        if token:
            return token.strip()

    return None


def build_genai_headers(access_token=None):
    """构建上游请求头，请求级 token 优先于启动参数 token。"""
    headers = GENAI_HEADERS.copy()
    if access_token:
        headers["X-Access-Token"] = access_token
    logger.debug("Using upstream access token override: %s", bool(access_token))
    return headers


def build_genai_upload_headers(access_token=None):
    """构建图片上传请求头。"""
    headers = {
        "Accept": "*/*",
        "Origin": "https://genai.shanghaitech.edu.cn",
        "Referer": "https://genai.shanghaitech.edu.cn/",
        "User-Agent": GENAI_HEADERS["User-Agent"],
        # 上传接口要求独立 token 头；同时附带 X-Access-Token 保持兼容。
        "token": args.upload_token,
        "X-Access-Token": access_token or args.token,
    }
    logger.debug("Upload headers prepared (token set=%s, request token override=%s)", bool(args.upload_token), bool(access_token))
    return headers


def infer_root_ai_type(model_name):
    """为未知模型推断上游路由类型。

    Args:
        model_name (Any): 调用方传入的模型名。

    Returns:
        str: 推断得到的 `rootAiType`，当前仅返回 `azure` 或 `xinference`。
    """
    if not isinstance(model_name, str):
        return "xinference"

    normalized = model_name.lower()
    # OpenAI / Azure 系列模型目前统一走 azure 路由。
    azure_markers = (
        "gpt-",
        "gpt",
        "o3",
        "o4-mini",
    )
    return "azure" if normalized.startswith(azure_markers) else "xinference"


def is_gpt_model(model_name):
    """判断模型是否为 GPT/Azure 系列（图片能力仅对其开放）。"""
    _, root_ai_type = resolve_model(model_name)
    return root_ai_type == "azure"


def guess_filename_from_url(image_url):
    """从 URL 推断文件名。"""
    path = urlparse(image_url).path
    filename = os.path.basename(path) or "image"
    if "." not in filename:
        filename += ".jpg"
    return filename


def read_image_from_data_url(data_url):
    """解析 data URL，返回 (bytes, mime_type, filename)。"""
    header, encoded = data_url.split(",", 1)
    mime_type = "image/jpeg"
    if header.startswith("data:"):
        mime_type = header[5:].split(";")[0] or mime_type
    extension = mimetypes.guess_extension(mime_type) or ".jpg"
    image_bytes = base64.b64decode(encoded)
    return image_bytes, mime_type, f"image{extension}"


def fetch_image_bytes(image_url):
    """下载远端图片，返回 (bytes, mime_type, filename)。"""
    response = requests.get(image_url, timeout=60)
    response.raise_for_status()
    mime_type = response.headers.get("Content-Type", "image/jpeg").split(";")[0].strip() or "image/jpeg"
    filename = guess_filename_from_url(image_url)
    return response.content, mime_type, filename


def upload_image_to_genai(image_bytes, filename, mime_type, access_token=None):
    """上传图片到 GenAI 图片服务，返回上游需要的 URL 与尺寸信息。"""
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    cached_payload = IMAGE_UPLOAD_CACHE.get(image_hash)
    if cached_payload:
        logger.debug("Image cache hit: sha256=%s url=%s", image_hash, cached_payload.get("imageUrl"))
        return cached_payload

    logger.debug("Image cache miss: sha256=%s", image_hash)
    files = {
        "file": (filename, image_bytes, mime_type),
    }
    data = {
        "biz": "temp",
        "uploadType": "local",
    }
    logger.debug("Uploading image to GenAI: filename=%s mime=%s bytes=%s", filename, mime_type, len(image_bytes))
    response = requests.post(
        GENAI_UPLOAD_URL,
        headers=build_genai_upload_headers(access_token),
        files=files,
        data=data,
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    logger.debug("Upload response: %s", payload)
    if not payload.get("success") or not isinstance(payload.get("result"), dict):
        raise RuntimeError(f"Image upload failed: {payload}")

    result = payload["result"]
    relative_url = result.get("url")
    if not relative_url:
        raise RuntimeError("Image upload failed: missing result.url")
    image_url = relative_url
    if not image_url.startswith("http://") and not image_url.startswith("https://"):
        image_url = f"{GENAI_IMAGE_STATIC_URL}{relative_url}"

    payload = {
        "imageUrl": image_url,
        "width": result.get("width"),
        "height": result.get("height"),
    }
    IMAGE_UPLOAD_CACHE[image_hash] = payload
    logger.debug("Image cached: sha256=%s url=%s", image_hash, payload.get("imageUrl"))
    return payload


def parse_image_input_from_message(message):
    """从单条 OpenAI user message 中提取图片输入（URL 或 data URL）。"""
    if not isinstance(message, dict) or message.get("role") != "user":
        return None

    content = message.get("content")
    if not isinstance(content, list):
        return None

    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type not in {"image_url", "input_image"}:
            continue

        if isinstance(part.get("image_url"), dict):
            url_value = part["image_url"].get("url")
            if url_value:
                return url_value
        if isinstance(part.get("image_url"), str):
            return part.get("image_url")
        if isinstance(part.get("url"), str):
            return part.get("url")

    return None


def prepare_image_payload(messages, model, access_token=None):
    """从请求消息中准备上游所需图片参数（仅 GPT 模型可用）。"""
    image_input = None
    for message in reversed(messages):
        image_input = parse_image_input_from_message(message)
        if image_input:
            break

    if not image_input:
        logger.debug("No image input found in messages")
        return None

    if not is_gpt_model(model):
        logger.debug("Rejecting image input for non-GPT model: %s", model)
        raise ValueError("Image input is only available for GPT models")

    if image_input.startswith("data:"):
        image_bytes, mime_type, filename = read_image_from_data_url(image_input)
    else:
        image_bytes, mime_type, filename = fetch_image_bytes(image_input)

    image_payload = upload_image_to_genai(image_bytes, filename, mime_type, access_token)
    logger.debug("Prepared image payload: %s", image_payload)
    return image_payload


def convert_messages_to_genai_format(messages):
    """从消息列表中提取 GenAI 所需的 `chatInfo`。

    当前上游实际请求中 `chatInfo` 只使用最后一条用户消息内容，因此这里
    仅做最小提取。

    Args:
        messages (list[dict]): OpenAI 风格的消息列表。

    Returns:
        str: 最后一条用户消息的文本内容；若不存在则返回空字符串。
    """
    # 上游会单独接收一份 chatInfo，这里取最后一条 user 消息与网页行为对齐。
    chat_info = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            chat_info = msg.get("content", "")
            break
    
    return chat_info


def normalize_content_for_genai(content):
    """将 OpenAI 消息 content 归一化为上游可读文本。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if text:
                    text_parts.append(text)
            elif isinstance(part, str):
                text_parts.append(part)
        return "\n".join(text_parts)
    return str(content)


def normalize_messages_for_genai(messages):
    """把 OpenAI tool messages 降级为普通文本，避免上游无法理解原生工具结构。"""
    normalized_messages = []
    for message in messages:
        if not isinstance(message, dict):
            continue

        role = message.get("role", "user")
        content = normalize_content_for_genai(message.get("content"))

        if role == "tool":
            tool_name = message.get("name") or message.get("tool_call_id") or "tool"
            normalized_messages.append({
                "role": "user",
                "content": f"工具 {tool_name} 返回结果：\n{content}",
            })
            continue

        tool_calls = message.get("tool_calls")
        if role == "assistant" and tool_calls and not content:
            normalized_messages.append({
                "role": "assistant",
                "content": "已请求调用工具：\n" + json.dumps(tool_calls, ensure_ascii=False),
            })
            continue

        normalized_messages.append({
            "role": role,
            "content": content,
        })

    return normalized_messages


def should_enable_tools(tools, tool_choice):
    """判断当前请求是否需要启用本地工具调用兼容层。"""
    return bool(tools) and tool_choice != "none"


def get_request_tools(req_data):
    """读取新版 tools 或旧版 functions 入参，统一为 OpenAI tools 结构。"""
    tools = req_data.get("tools")
    if tools:
        return tools

    functions = req_data.get("functions")
    if not functions:
        return []

    return [
        {
            "type": "function",
            "function": function,
        }
        for function in functions
        if isinstance(function, dict)
    ]


def get_request_tool_choice(req_data):
    """读取新版 tool_choice 或旧版 function_call 入参。"""
    if "tool_choice" in req_data:
        return req_data.get("tool_choice")

    function_call = req_data.get("function_call")
    if isinstance(function_call, dict) and function_call.get("name"):
        return {"name": function_call["name"]}
    return function_call


def normalize_tool_choice(tool_choice):
    """将 OpenAI 的 tool_choice 归一化为提示词中便于描述的约束。"""
    if tool_choice is None:
        return "auto"
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict):
        if tool_choice.get("name"):
            return {"name": tool_choice["name"]}
        function_name = tool_choice.get("function", {}).get("name")
        if function_name:
            return {"name": function_name}
    return "auto"


def build_tool_calling_messages(messages, tools, tool_choice):
    """通过提示词工程让无原生工具调用能力的上游返回可解析的工具调用 JSON。"""
    normalized_choice = normalize_tool_choice(tool_choice)
    tool_prompt = [
        "你可以调用调用方提供的工具，但上游 API 没有原生 tool calling 能力。",
        "当你决定调用工具时，必须只输出一个 JSON 对象，不要输出 Markdown、解释或额外文本。",
        "JSON 格式必须为：{\"tool_calls\":[{\"name\":\"工具名\",\"arguments\":{}}]}。",
        "arguments 必须是符合工具 JSON Schema 的对象。",
        "如果不需要调用工具，则正常回答用户，不要输出上述 JSON。",
        f"tool_choice: {json.dumps(normalized_choice, ensure_ascii=False)}",
        "可用工具：",
        json.dumps(tools, ensure_ascii=False),
    ]

    if normalized_choice == "required":
        tool_prompt.append("本次请求必须调用至少一个工具。")
    elif isinstance(normalized_choice, dict):
        tool_prompt.append(f"本次请求必须调用工具 {normalized_choice['name']}。")

    return [
        {"role": "system", "content": "\n".join(tool_prompt)},
        *messages,
    ]


def strip_json_code_fence(text):
    """去掉模型偶尔包裹的 JSON Markdown 代码块。"""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def extract_json_object(text):
    """从文本中提取第一个完整 JSON 对象。"""
    stripped = strip_json_code_fence(text)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(stripped[start:index + 1])
                except json.JSONDecodeError:
                    return None
    return None


def normalize_tool_call_arguments(arguments):
    """OpenAI 要求 function.arguments 是 JSON 字符串。"""
    if isinstance(arguments, str):
        return arguments
    if arguments is None:
        return "{}"
    return json.dumps(arguments, ensure_ascii=False)


def parse_tool_calls_from_content(content):
    """解析提示词约定的工具调用 JSON，并转换为 OpenAI tool_calls 结构。"""
    if not content:
        return []

    parsed = extract_json_object(content)
    if not isinstance(parsed, dict):
        return []

    raw_calls = parsed.get("tool_calls")
    if raw_calls is None and parsed.get("name"):
        raw_calls = [parsed]
    if not isinstance(raw_calls, list):
        return []

    tool_calls = []
    for raw_call in raw_calls:
        if not isinstance(raw_call, dict):
            continue

        function = raw_call.get("function") if isinstance(raw_call.get("function"), dict) else raw_call
        name = function.get("name")
        if not name:
            continue

        arguments = function.get("arguments", {})
        tool_calls.append({
            "id": raw_call.get("id") or f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": normalize_tool_call_arguments(arguments),
            },
        })

    return tool_calls


def build_chat_completion_payload(model, content, reasoning_content=None, tool_calls=None):
    """构建非流式 Chat Completions 响应。"""
    message = {
        "role": "assistant",
        "content": None if tool_calls else content,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    elif reasoning_content is not None:
        message["reasoning_content"] = reasoning_content

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(datetime.now().timestamp()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": len(content or ""),
            "total_tokens": len(content or "")
        }
    }

def extract_delta_from_genai(response_data):
    """从 GenAI 增量响应中提取正文和思维链字段。

    Args:
        response_data (dict): 单条 GenAI SSE 数据解析后的 JSON 对象。

    Returns:
        dict[str, str | None]: 包含 `reasoning` 与 `content` 两个字段；若缺失则
        返回 `None`。
    """
    try:
        if "choices" in response_data and len(response_data["choices"]) > 0:
            delta = response_data["choices"][0].get("delta", {})
            return {
                "reasoning": delta.get("reasoning"),
                "content": delta.get("content"),
            }
    except (KeyError, IndexError, TypeError):
        pass
    return {"reasoning": None, "content": None}


def stream_genai_events(messages, model, max_tokens, access_token=None, image_payload=None):
    """调用 GenAI 流式接口并产出统一事件流。

    该函数是整个协议转换的底层入口，负责：
    1. 解析模型别名
    2. 调用上游 GenAI SSE 接口
    3. 将上游原始事件规范化为内部事件类型

    Args:
        messages (list[dict]): 发送给上游的消息列表。
        model (str): 调用方指定的模型名。
        max_tokens (int | None): 最大输出 token 数。
        access_token (str | None): 请求级 GenAI token，未提供时使用启动参数。

    Yields:
        dict: 统一事件对象，`type` 可能为 `delta`、`done`、`meta` 或 `error`。
    """
    upstream_model, root_ai_type = resolve_model(model)

    # 这里保持与网页端接近的请求体结构，避免上游校验差异。
    genai_data = {
        "chatInfo": "",
        "messages": messages,
        "type": "3",
        "stream": True,
        "aiType": upstream_model,
        "aiSecType": "1",
        "promptTokens": 0,
        "rootAiType": root_ai_type,
        "maxToken": max_tokens or 30000
    }
    if image_payload:
        genai_data.update(image_payload)

    logger.debug(
        "Upstream request prepared: model=%s rootAiType=%s stream=%s maxToken=%s has_image=%s message_count=%s",
        upstream_model,
        root_ai_type,
        genai_data.get("stream"),
        genai_data.get("maxToken"),
        bool(image_payload),
        len(messages) if isinstance(messages, list) else 0,
    )

    try:
        response = requests.post(
            GENAI_URL,
            headers=build_genai_headers(access_token),
            json=genai_data,
            stream=True,
            timeout=60
        )

        if response.status_code != 200:
            logger.error("GenAI upstream HTTP error: %s", response.status_code)
            yield {
                "type": "error",
                "error": f"GenAI API error: {response.status_code}",
            }
            return

        finished = False
        for line in response.iter_lines():
            if finished:
                break

            if line:
                try:
                    line_str = line.decode('utf-8') if isinstance(line, bytes) else line

                    # 兼容标准 SSE 的 `data:` 前缀。
                    if line_str.startswith('data:'):
                        line_str = line_str[5:].strip()

                    if line_str:
                        genai_json = json.loads(line_str)
                        logger.debug("Upstream SSE chunk keys: %s", list(genai_json.keys()))

                        # 上游偶尔会返回补充元数据，先保留为内部 meta 事件。
                        if genai_json.get("other"):
                            yield {
                                "type": "meta",
                                "other": genai_json.get("other"),
                            }

                        # 只要上游给出 finish_reason，就视为本轮流式输出结束。
                        if "choices" in genai_json and len(genai_json["choices"]) > 0:
                            choice = genai_json["choices"][0]
                            if choice.get("finish_reason") is not None:
                                finished = True

                        if finished:
                            yield {
                                "type": "done",
                                "upstream_model": genai_json.get("model"),
                            }
                            break

                        delta = extract_delta_from_genai(genai_json)
                        reasoning = delta.get("reasoning")
                        content = delta.get("content")
                        # 内部统一拆成 reasoning 和 content，便于上层复用。
                        if reasoning is not None or content is not None:
                            yield {
                                "type": "delta",
                                "upstream_model": genai_json.get("model"),
                                "reasoning": reasoning,
                                "content": content,
                            }

                except json.JSONDecodeError:
                    pass

        yield {
            "type": "done",
            "upstream_model": None,
        }

    except Exception as e:
        logger.exception("stream_genai_events failed")
        # 流式链路统一转成 error 事件，交由上层协议各自包装。
        yield {
            "type": "error",
            "error": str(e),
        }


def stream_chat_completions_response(messages, model, max_tokens, access_token=None, image_payload=None):
    """将内部事件流转换为 Chat Completions SSE。

    Args:
        messages (list[dict]): OpenAI 风格消息列表。
        model (str): 调用方传入的模型名。
        max_tokens (int | None): 最大输出 token 数。
        access_token (str | None): 请求级 GenAI token，未提供时使用启动参数。

    Yields:
        str: 符合 OpenAI Chat Completions SSE 格式的文本片段。
    """
    response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(datetime.now().timestamp())

    for event in stream_genai_events(messages, model, max_tokens, access_token, image_payload):
        if event["type"] == "error":
            yield f"data: {json.dumps({'error': event['error']})}\n\n"
            return

        if event["type"] == "delta":
            delta_payload = {}
            # 对外沿用 DeepSeek 常见字段名 reasoning_content。
            if event.get("reasoning") is not None:
                delta_payload["reasoning_content"] = event["reasoning"]
            if event.get("content") is not None:
                delta_payload["content"] = event["content"]

            if delta_payload:
                openai_response = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": delta_payload,
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(openai_response)}\n\n"

        if event["type"] == "done":
            final_response = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ]
            }
            yield f"data: {json.dumps(final_response)}\n\n"
            yield "data: [DONE]\n\n"
            return


def stream_tool_calls_response(model, content, tool_calls):
    """将完整解析出的工具调用转换为 Chat Completions SSE。"""
    response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(datetime.now().timestamp())

    if tool_calls:
        delta_tool_calls = []
        for index, tool_call in enumerate(tool_calls):
            delta_tool_calls.append({
                "index": index,
                "id": tool_call["id"],
                "type": "function",
                "function": tool_call["function"],
            })

        tool_call_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": delta_tool_calls,
                    },
                    "finish_reason": None,
                }
            ]
        }
        yield f"data: {json.dumps(tool_call_chunk)}\n\n"
        finish_reason = "tool_calls"
    else:
        content_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": content,
                    },
                    "finish_reason": None,
                }
            ]
        }
        yield f"data: {json.dumps(content_chunk)}\n\n"
        finish_reason = "stop"

    final_response = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
            }
        ]
    }
    yield f"data: {json.dumps(final_response)}\n\n"
    yield "data: [DONE]\n\n"


def collect_genai_response(messages, model, max_tokens, access_token=None, image_payload=None):
    """收集完整响应并聚合为非流式结果。

    Args:
        messages (list[dict]): OpenAI 风格消息列表。
        model (str): 调用方传入的模型名。
        max_tokens (int | None): 最大输出 token 数。
        access_token (str | None): 请求级 GenAI token，未提供时使用启动参数。

    Returns:
        dict[str, str | None]: 聚合后的正文、思维链和上游模型名。

    Raises:
        RuntimeError: 当上游事件流返回错误事件时抛出。
    """
    content_parts = []
    reasoning_parts = []
    upstream_model = None

    for event in stream_genai_events(messages, model, max_tokens, access_token, image_payload):
        if event["type"] == "error":
            raise RuntimeError(event["error"])
        if event["type"] == "delta":
            upstream_model = event.get("upstream_model") or upstream_model
            if event.get("reasoning"):
                reasoning_parts.append(event["reasoning"])
            if event.get("content"):
                content_parts.append(event["content"])
        if event["type"] == "done":
            break

    return {
        "content": "".join(content_parts),
        "reasoning_content": "".join(reasoning_parts),
        "upstream_model": upstream_model,
    }


def build_response_input_messages(input_value):
    """将 Responses API 输入归一化为消息列表。

    当前仅处理文本输入，兼容字符串输入以及包含文本片段的数组输入。

    Args:
        input_value (str | list | Any): `/v1/responses` 的 `input` 字段。

    Returns:
        list[dict]: 可直接发给上游的消息列表。
    """
    if isinstance(input_value, str):
        return [{"role": "user", "content": input_value}]

    if isinstance(input_value, list):
        messages = []
        for item in input_value:
            if not isinstance(item, dict):
                continue

            role = item.get("role", "user")
            content = item.get("content")

            if isinstance(content, str):
                messages.append({"role": role, "content": content})
                continue

            if isinstance(content, list):
                # 仅提取文本片段，忽略当前版本尚未支持的其他 item 类型。
                text_parts = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    part_type = part.get("type")
                    if part_type in {"input_text", "text", "output_text"}:
                        text = part.get("text")
                        if text:
                            text_parts.append(text)
                if text_parts:
                    messages.append({"role": role, "content": "\n".join(text_parts)})

        return messages

    return []


def stream_responses_api(messages, model, max_tokens, access_token=None):
    """将内部事件流转换为最小 Responses API SSE。

    Args:
        messages (list[dict]): 发送给上游的消息列表。
        model (str): 调用方传入的模型名。
        max_tokens (int | None): 最大输出 token 数。
        access_token (str | None): 请求级 GenAI token，未提供时使用启动参数。

    Yields:
        str: 符合最小 Responses API SSE 格式的文本片段。
    """
    response_id = f"resp_{uuid.uuid4().hex}"
    created = int(datetime.now().timestamp())
    reasoning_id = f"rs_{uuid.uuid4().hex[:12]}"
    output_index = 0

    created_event = {
        "type": "response.created",
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": created,
            "status": "in_progress",
            "model": model,
        }
    }
    yield f"data: {json.dumps(created_event)}\n\n"

    for event in stream_genai_events(messages, model, max_tokens, access_token):
        if event["type"] == "error":
            error_event = {
                "type": "response.failed",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": created,
                    "status": "failed",
                    "model": model,
                },
                "error": {
                    "message": event["error"],
                }
            }
            yield f"data: {json.dumps(error_event)}\n\n"
            yield "data: [DONE]\n\n"
            return

        if event["type"] == "delta":
            # Responses 接口将 reasoning 和正文拆成不同事件类型。
            if event.get("reasoning") is not None:
                reasoning_event = {
                    "type": "response.reasoning.delta",
                    "response_id": response_id,
                    "output_index": output_index,
                    "item_id": reasoning_id,
                    "delta": event["reasoning"],
                }
                yield f"data: {json.dumps(reasoning_event)}\n\n"

            if event.get("content") is not None:
                content_event = {
                    "type": "response.output_text.delta",
                    "response_id": response_id,
                    "output_index": output_index,
                    "delta": event["content"],
                }
                yield f"data: {json.dumps(content_event)}\n\n"

        if event["type"] == "done":
            completed_event = {
                "type": "response.completed",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": created,
                    "status": "completed",
                    "model": model,
                }
            }
            yield f"data: {json.dumps(completed_event)}\n\n"
            yield "data: [DONE]\n\n"
            return

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """处理 OpenAI Chat Completions 兼容请求。

    Returns:
        Response: Flask JSON 响应或 SSE 流式响应。
    """
    try:
        req_data = request.get_json()
        logger.debug("/v1/chat/completions request received: stream=%s model=%s", (req_data or {}).get('stream'), (req_data or {}).get('model'))
        
        # Chat Completions 至少需要消息数组。
        if not req_data or 'messages' not in req_data:
            return jsonify({'error': 'Missing messages field'}), 400
        
        messages = req_data.get('messages', [])
        model = req_data.get('model', 'gpt-3.5-turbo')
        stream = req_data.get('stream', False)
        max_tokens = req_data.get('max_tokens', req_data.get('max_completion_tokens', 30000))
        tools = get_request_tools(req_data)
        tool_choice = get_request_tool_choice(req_data)
        access_token = get_request_access_token()
        image_payload = prepare_image_payload(messages, model, access_token)
        
        # 转换消息格式
        chat_info = convert_messages_to_genai_format(messages)
        
        if not chat_info:
            return jsonify({'error': 'No user message found'}), 400

        tools_enabled = should_enable_tools(tools, tool_choice)
        upstream_messages = normalize_messages_for_genai(messages)
        if tools_enabled:
            upstream_messages = build_tool_calling_messages(upstream_messages, tools, tool_choice)

        if stream:
            if tools_enabled:
                collected = collect_genai_response(upstream_messages, model, max_tokens, access_token, image_payload)
                tool_calls = parse_tool_calls_from_content(collected["content"])
                return Response(
                    stream_with_context(stream_tool_calls_response(model, collected["content"], tool_calls)),
                    mimetype='text/event-stream',
                    headers={
                        'Cache-Control': 'no-cache',
                        'Connection': 'keep-alive',
                        'Content-Type': 'text/event-stream',
                    }
                )

            return Response(
                stream_with_context(stream_chat_completions_response(upstream_messages, model, max_tokens, access_token, image_payload)),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'Content-Type': 'text/event-stream',
                }
            )

        # 非流式模式先完整收集，再一次性组装 OpenAI 响应体。
        collected = collect_genai_response(upstream_messages, model, max_tokens, access_token, image_payload)
        tool_calls = parse_tool_calls_from_content(collected["content"]) if tools_enabled else []
        response = build_chat_completion_payload(
            model,
            collected["content"],
            collected["reasoning_content"],
            tool_calls,
        )
        return jsonify(response)
    
    except Exception as e:
        logger.exception("chat_completions failed")
        return jsonify({'error': str(e)}), 500


@app.route('/v1/responses', methods=['POST'])
def responses():
    """处理最小 OpenAI Responses 兼容请求。

    Returns:
        Response: Flask JSON 响应或 SSE 流式响应。
    """
    try:
        req_data = request.get_json()
        logger.debug("/v1/responses request received: stream=%s model=%s", (req_data or {}).get('stream'), (req_data or {}).get('model'))
        if not req_data or 'input' not in req_data:
            return jsonify({'error': 'Missing input field'}), 400

        model = req_data.get('model', 'gpt-4.1')
        stream = req_data.get('stream', False)
        max_output_tokens = req_data.get('max_output_tokens', req_data.get('max_tokens', 30000))
        messages = build_response_input_messages(req_data.get('input'))
        access_token = get_request_access_token()

        if not messages:
            return jsonify({'error': 'No input message found'}), 400

        if stream:
            return Response(
                stream_with_context(stream_responses_api(messages, model, max_output_tokens, access_token)),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'Content-Type': 'text/event-stream',
                }
            )

        # 非流式返回时，将 reasoning 和 message 组装到 output 数组中。
        collected = collect_genai_response(messages, model, max_output_tokens, access_token)
        response_id = f"resp_{uuid.uuid4().hex}"
        output = []
        if collected["reasoning_content"]:
            output.append({
                "id": f"rs_{uuid.uuid4().hex[:12]}",
                "type": "reasoning",
                "summary": [
                    {
                        "type": "summary_text",
                        "text": collected["reasoning_content"],
                    }
                ]
            })
        output.append({
            "id": f"msg_{uuid.uuid4().hex[:12]}",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": collected["content"],
                }
            ]
        })

        return jsonify({
            "id": response_id,
            "object": "response",
            "created_at": int(datetime.now().timestamp()),
            "status": "completed",
            "model": model,
            "output": output,
            "output_text": collected["content"],
        })

    except Exception as e:
        logger.exception("responses failed")
        return jsonify({'error': str(e)}), 500

@app.route('/v1/models', methods=['GET'])
def list_models():
    """返回当前对外暴露的模型列表。

    Returns:
        Response: OpenAI `/v1/models` 兼容 JSON 响应。
    """
    models = []
    for spec in MODEL_SPECS:
        models.append({
            "id": spec["public_id"],
            "object": "model",
            "owned_by": "genai",
            "permission": []
        })
    
    return jsonify({"object": "list", "data": models})

@app.route('/health', methods=['GET'])
def health_check():
    """返回服务健康状态。

    Returns:
        tuple[Response, int]: 健康检查 JSON 响应与状态码。
    """
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=args.port, debug=False)
