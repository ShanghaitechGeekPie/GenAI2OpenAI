import argparse
import hashlib
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime

import requests
from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS
from rich.console import Console
from rich.logging import RichHandler

sys.path.append(os.path.join(os.path.dirname(__file__), "utility", "auto_login"))
from cas_login import LoginError, login_genai
from genai_proxy.models import (
    MODEL_SPECS,
    build_remote_model_list,
    resolve_model,
    is_gpt_model,
)
from genai_proxy.upstream import (
    build_chat_completion_payload,
    extract_delta_from_genai,
    build_response_input_messages,
    iter_sse_events,
    ToolCallAccumulator,
)
from genai_proxy.streaming import iter_with_heartbeat
from genai_proxy.images import (
    guess_filename_from_url,
    read_image_from_data_url,
    fetch_image_bytes,
    parse_image_input_from_message,
)
from genai_proxy.tools import (
    convert_messages_to_genai_format,
    normalize_messages_for_genai,
    should_enable_tools,
    get_request_tools,
    get_request_tool_choice,
    build_tool_calling_messages,
    parse_tool_calls_from_content,
)

app = Flask(__name__)
CORS(app)

# 解析命令行参数
parser = argparse.ArgumentParser(description='GenAI Flask API Server')
parser.add_argument('--token', type=str, default=None,
                    help='GenAI API Access Token')
parser.add_argument('--account', type=str, default=None,
                    help='ShanghaiTech account in the format student_id@password, used to auto-login and get token')
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

BASE_DIR = os.path.dirname(__file__)
TOKEN_CACHE_PATH = os.path.join(BASE_DIR, ".genai_token_cache")


def load_cached_token():
    """从项目本目录读取缓存 token。"""
    if not os.path.exists(TOKEN_CACHE_PATH):
        return None

    try:
        with open(TOKEN_CACHE_PATH, "r", encoding="utf-8") as token_file:
            token = token_file.read().strip()
    except OSError:
        logger.exception("Failed to read token cache")
        return None

    return token or None


def save_cached_token(token):
    """将自动登录获得的 token 写入项目本目录缓存。"""
    try:
        with open(TOKEN_CACHE_PATH, "w", encoding="utf-8") as token_file:
            token_file.write(token.strip())
    except OSError:
        logger.exception("Failed to write token cache")


def build_startup_genai_headers(token):
    """构建启动阶段校验 token 所需的最小上游请求头。"""
    return {
        "Accept": "*/*, text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Content-Type": "application/json",
        "Origin": "https://genai.shanghaitech.edu.cn",
        "Referer": "https://genai.shanghaitech.edu.cn/dialogue",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
        "X-Access-Token": token,
    }


def validate_cached_token(token):
    """用 deepseek-v3 发起最小对话，返回非空内容则认为 token 有效。"""
    if not token:
        return False

    payload = {
        "chatInfo": "你好",
        "messages": [{"role": "user", "content": "你好"}],
        "type": "3",
        "stream": True,
        "aiType": "deepseek-v3:671b",
        "aiSecType": "1",
        "promptTokens": 0,
        "rootAiType": "xinference",
        "maxToken": 16,
    }

    try:
        response = requests.post(
            "https://genai.shanghaitech.edu.cn/htk/chat/start/chat",
            headers=build_startup_genai_headers(token),
            json=payload,
            stream=True,
            timeout=30,
        )
        if response.status_code != 200:
            logger.info("Cached token validation failed with HTTP %s", response.status_code)
            return False

        for line in response.iter_lines():
            if not line:
                continue

            line_str = line.decode("utf-8") if isinstance(line, bytes) else line
            if line_str.startswith("data:"):
                line_str = line_str[5:].strip()
            if not line_str:
                continue

            try:
                chunk = json.loads(line_str)
            except json.JSONDecodeError:
                continue

            choices = chunk.get("choices") or []
            if not choices:
                continue

            delta = choices[0].get("delta") or {}
            if delta.get("content") or delta.get("reasoning"):
                return True
    except Exception:
        logger.exception("Cached token validation failed")

    return False


def auto_login_with_account(account):
    try:
        account_student_id, account_password = account.split("@", 1)
        logger.info("Attempting auto login with account: %s", account_student_id)
        token = login_genai(account_student_id, account_password)
        logger.info("Auto login succeeded for account: %s", account_student_id)
        save_cached_token(token)
        return token
    except ValueError:
        logger.error("Invalid --account format, expected student_id@password")
        raise SystemExit("--account must be in the format student_id@password")
    except LoginError as exc:
        logger.exception("Auto login failed")
        raise SystemExit(f"Auto login failed: {exc}")


if not args.token:
    cached_token = load_cached_token()
    if cached_token:
        logger.info("Found cached token, validating with deepseek-v3")
        if validate_cached_token(cached_token):
            args.token = cached_token
            logger.info("Cached token is valid")
        else:
            logger.info("Cached token is invalid or expired")

if not args.token and args.account:
    args.token = auto_login_with_account(args.account)

# 进程内图片去重缓存：image_sha256 -> {imageUrl, width, height}
IMAGE_UPLOAD_CACHE = {}

# GenAI API 配置
GENAI_URL = "https://genai.shanghaitech.edu.cn/htk/chat/start/chat"
GENAI_MODELS_URL = "https://genai.shanghaitech.edu.cn/htk/ai/aiModel/list"
GENAI_UPLOAD_URL = "https://genaipic.shanghaitech.edu.cn//sys/common/upload"
GENAI_IMAGE_STATIC_URL = "https://genaipic.shanghaitech.edu.cn//sys/common/static/"
GENAI_STREAM_TIMEOUT = (10, 180)
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
    "X-Access-Token": args.token or "",
    "sec-ch-ua": '"Chromium";v="142", "Google Chrome";v="142", "Not_A Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
}

def get_request_access_token():
    """从 OpenAI 常用认证头中提取请求级 GenAI token。"""
    if args.account:
        logger.debug("Ignoring request access token because --account is enabled")
        return None

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


def fetch_remote_models(access_token=None):
    """拉取 GenAI 平台当前可用模型列表。"""
    response = requests.get(
        GENAI_MODELS_URL,
        headers=build_genai_headers(access_token),
        params={
            "_t": int(datetime.now().timestamp() * 1000),
            "pageNo": 1,
            "pageSize": 999,
            "showStatusList": "2,3",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success"):
        raise RuntimeError(f"Failed to fetch remote models: {payload}")
    records = (payload.get("result") or {}).get("records")
    if not isinstance(records, list):
        raise RuntimeError("Invalid remote model list: missing records array")
    return records


def log_new_remote_models(access_token=None):
    """启动时检查远端模型列表，提示本地未登记的新模型。"""
    if not access_token:
        logger.debug("Skipping remote model discovery because no startup token is available")
        return

    try:
        remote_records = fetch_remote_models(access_token)
    except Exception:
        logger.exception("Failed to fetch remote model list at startup")
        return

    local_aliases = {
        alias.lower()
        for spec in MODEL_SPECS
        for alias in (spec["public_id"], spec["request_id"], spec["actual_id"])
        if isinstance(alias, str)
    }

    discovered = []
    for record in remote_records:
        ai_type = record.get("aiType")
        simple_name = record.get("simpleName")
        ai_name = record.get("aiName")
        candidates = [value for value in (ai_type, simple_name, ai_name) if isinstance(value, str) and value]
        if any(candidate.lower() in local_aliases for candidate in candidates):
            continue
        discovered.append({
            "aiType": ai_type,
            "simpleName": simple_name,
            "aiName": ai_name,
            "rootAiType": record.get("rootAiType"),
        })

    if not discovered:
        logger.info("Remote model discovery: no new models compared with local MODEL_SPECS")
        return

    logger.warning("Remote model discovery found %d new model(s) not in local MODEL_SPECS:", len(discovered))
    for model in discovered:
        logger.warning(
            "  - aiType=%s simpleName=%s aiName=%s rootAiType=%s",
            model.get("aiType"),
            model.get("simpleName"),
            model.get("aiName"),
            model.get("rootAiType"),
        )


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


def _stream_genai_attempt(messages, model, max_tokens, access_token=None, image_payload=None, request_id=None, attempt=1):
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

    response = None
    started_at = time.monotonic()
    stats = {"events": 0, "content_chars": 0, "reasoning_chars": 0,
             "native_fragments": 0, "delta_fields": {}, "message_fields": {}, "finish_reason": None}

    def trace_delta(payload):
        stats["events"] += 1
        choice = (payload.get("choices") or [{}])[0]
        for source, key in (("delta", "delta_fields"), ("message", "message_fields")):
            fields = choice.get(source)
            if isinstance(fields, dict):
                for name in fields:
                    stats[key][name] = stats[key].get(name, 0) + 1
        stats["finish_reason"] = choice.get("finish_reason") or stats["finish_reason"]
        delta = extract_delta_from_genai(payload)
        for field, key in (("content", "content_chars"), ("reasoning", "reasoning_chars")):
            if isinstance(delta.get(field), str):
                stats[key] += len(delta[field])
        stats["native_fragments"] += len(delta.get("tool_calls") or [])
        return delta

    try:
        response = requests.post(
            GENAI_URL,
            headers=build_genai_headers(access_token),
            json=genai_data,
            stream=True,
            timeout=GENAI_STREAM_TIMEOUT
        )

        if response.status_code != 200:
            logger.error("GenAI upstream HTTP error: %s", response.status_code)
            yield {
                "type": "error",
                "error": f"GenAI API error: {response.status_code}",
            }
            return

        for event in iter_sse_events(response.iter_lines(chunk_size=1), trace_delta):
            if event["type"] == "delta":
                logger.debug("Upstream SSE delta received")
            if event["type"] == "done":
                stats["completion_marker"] = True
                stats["finish_reason"] = event.get("finish_reason")
            yield event

    except Exception as e:
        logger.exception("stream_genai_events failed")
        # 流式链路统一转成 error 事件，交由上层协议各自包装。
        yield {
            "type": "error",
            "error": str(e),
        }
    finally:
        if response is not None:
            response.close()
        logger.info(
            "Upstream summary: request_id=%s attempt=%s model=%s elapsed=%.2fs stats=%s",
            request_id, attempt, model, time.monotonic() - started_at, json.dumps(stats),
        )


def stream_genai_events(messages, model, max_tokens, access_token=None, image_payload=None):
    """仅对正常结束的空生成重试一次；发布正文或调用后不再重试。"""
    request_id = uuid.uuid4().hex[:12]
    for attempt in (1, 2):
        pending = []
        published = False
        retry = False
        events = _stream_genai_attempt(
            messages, model, max_tokens, access_token, image_payload, request_id, attempt,
        )
        try:
            for event in events:
                if event["type"] == "error":
                    yield event
                    return
                if event["type"] == "delta":
                    has_output = bool((event.get("content") or "").strip() or event.get("tool_calls"))
                    if not published and not has_output:
                        pending.append(event)
                        continue
                    if not published:
                        yield from pending
                        pending.clear()
                        published = True
                    yield event
                elif event["type"] == "done":
                    if not published and event.get("finish_reason", "stop") == "stop":
                        if attempt == 1:
                            logger.warning("Retrying empty upstream generation: request_id=%s", request_id)
                            retry = True
                            break
                        yield {"type": "error", "error": "GenAI upstream returned an empty response after one retry"}
                        return
                    yield from pending
                    yield event
                    return
                else:
                    yield event
        finally:
            close = getattr(events, "close", None)
            if close:
                close()
        if not retry:
            yield {"type": "error", "error": "GenAI upstream stream ended without a completion marker"}
            return


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
        if event["type"] == "delta" and event.get("tool_calls"):
            event = {"type": "error", "error": "Upstream returned native tool calls without enabled tools"}
        if event["type"] == "error":
            yield f"data: {json.dumps({'error': event['error']})}\n\n"
            yield "data: [DONE]\n\n"
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
                        "finish_reason": event.get("finish_reason", "stop")
                    }
                ]
            }
            yield f"data: {json.dumps(final_response)}\n\n"
            yield "data: [DONE]\n\n"
            return


def stream_tool_calls_response(model, content, tool_calls, finish_reason="stop"):
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


def stream_buffered_tool_response(messages, model, max_tokens, access_token=None, image_payload=None, tools=None, tool_choice=None):
    """聚合工具 JSON 时发送心跳，避免客户端在静默等待时断开。"""
    started_at = time.monotonic()
    content_parts = []
    native_calls = ToolCallAccumulator()
    finish_reason = None
    events = iter_with_heartbeat(
        stream_genai_events(messages, model, max_tokens, access_token, image_payload)
    )
    try:
        for event in events:
            if event is None:
                yield ": keep-alive\n\n"
            elif event["type"] == "error":
                raise RuntimeError(event["error"])
            elif event["type"] == "delta":
                if event.get("content"):
                    content_parts.append(event["content"])
                native_calls.add(event.get("tool_calls"))
            elif event["type"] == "done":
                finish_reason = event.get("finish_reason", "stop")
                break
        content = "".join(content_parts)
        raw_calls = native_calls.result()
        if not content.strip() and not raw_calls:
            raise RuntimeError("GenAI upstream returned an empty response; please retry")
        if finish_reason not in ("stop", "tool_calls", "function_call"):
            raise RuntimeError(f"GenAI tool response was not completed: {finish_reason}")
        tool_calls = parse_tool_calls_from_content(
            json.dumps({"tool_calls": raw_calls}) if raw_calls else content, tools, tool_choice,
        )
        if finish_reason in ("tool_calls", "function_call") and not tool_calls:
            raise RuntimeError("GenAI upstream ended with tool_calls but returned no valid call")
        logger.info(
            "Tool response collected: model=%s elapsed=%.1fs content_chars=%s tool_call_count=%s",
            model, time.monotonic() - started_at, len(content), len(tool_calls),
        )
        yield from stream_tool_calls_response(model, content, tool_calls)
    except Exception as exc:
        logger.warning(
            "Tool stream failed: model=%s content_chars=%s finish_reason=%s error=%s",
            model, sum(map(len, content_parts)), finish_reason, exc,
        )
        yield f"data: {json.dumps({'error': {'message': str(exc), 'type': 'upstream_error'}})}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        events.close()


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
    native_calls = ToolCallAccumulator()
    upstream_model = None
    finish_reason = "stop"

    for event in stream_genai_events(messages, model, max_tokens, access_token, image_payload):
        if event["type"] == "error":
            raise RuntimeError(event["error"])
        if event["type"] == "delta":
            native_calls.add(event.get("tool_calls"))
            upstream_model = event.get("upstream_model") or upstream_model
            if event.get("reasoning"):
                reasoning_parts.append(event["reasoning"])
            if event.get("content"):
                content_parts.append(event["content"])
        if event["type"] == "done":
            finish_reason = event.get("finish_reason", "stop")
            break

    return {
        "content": "".join(content_parts),
        "reasoning_content": "".join(reasoning_parts),
        "upstream_model": upstream_model,
        "finish_reason": finish_reason,
        "tool_calls": native_calls.result(),
    }


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
        if event["type"] == "delta" and event.get("tool_calls"):
            event = {"type": "error", "error": "Native tool calls require Chat Completions with enabled tools"}
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
            incomplete = event.get("finish_reason", "stop") != "stop"
            completed_event = {
                "type": "response.incomplete" if incomplete else "response.completed",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": created,
                    "status": "incomplete" if incomplete else "completed",
                    "model": model,
                    "incomplete_details": {"reason": event["finish_reason"]} if incomplete else None,
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
                logger.info(
                    "Collecting tool response: model=%s message_count=%s tool_count=%s",
                    model,
                    len(upstream_messages),
                    len(tools),
                )
                return Response(
                    stream_with_context(stream_buffered_tool_response(upstream_messages, model, max_tokens, access_token, image_payload, tools, tool_choice)),
                    mimetype='text/event-stream',
                    headers={
                        'Cache-Control': 'no-cache',
                        'Connection': 'keep-alive',
                        'Content-Type': 'text/event-stream',
                        'X-Accel-Buffering': 'no',
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
        if collected["tool_calls"] and not tools_enabled:
            raise RuntimeError("Upstream returned native tool calls without enabled tools")
        if tools_enabled and collected["finish_reason"] not in ("stop", "tool_calls", "function_call"):
            raise RuntimeError(f"GenAI tool response was not completed: {collected['finish_reason']}")
        tool_content = json.dumps({"tool_calls": collected["tool_calls"]}) if collected["tool_calls"] else collected["content"]
        tool_calls = parse_tool_calls_from_content(tool_content, tools, tool_choice) if tools_enabled else []
        if tools_enabled and collected["finish_reason"] in ("tool_calls", "function_call") and not tool_calls:
            raise RuntimeError("GenAI upstream ended with tool_calls but returned no valid call")
        response = build_chat_completion_payload(
            model,
            collected["content"],
            collected["reasoning_content"],
            tool_calls,
            collected["finish_reason"],
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
        if collected["tool_calls"]:
            raise RuntimeError("Native tool calls require Chat Completions with enabled tools")
        incomplete = collected["finish_reason"] != "stop"
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
            "status": "incomplete" if incomplete else "completed",
            "incomplete_details": {"reason": collected["finish_reason"]} if incomplete else None,
            "model": model,
            "output": output,
            "output_text": collected["content"],
        })

    except Exception as e:
        logger.exception("responses failed")
        return jsonify({'error': str(e)}), 500

@app.route('/v1/models', methods=['GET'])
def list_models():
    """使用当前请求凭据拉取上游模型列表。

    Returns:
        Response: OpenAI `/v1/models` 兼容 JSON 响应。
    """
    try:
        records = fetch_remote_models(get_request_access_token())
        return jsonify(build_remote_model_list(records))
    except Exception as exc:
        logger.exception("Failed to fetch remote model list")
        return jsonify({"error": str(exc)}), 502

@app.route('/health', methods=['GET'])
def health_check():
    """返回服务健康状态。

    Returns:
        tuple[Response, int]: 健康检查 JSON 响应与状态码。
    """
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    log_new_remote_models(args.token)
    app.run(host='0.0.0.0', port=args.port, debug=False)
