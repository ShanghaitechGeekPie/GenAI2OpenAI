"""Pure upstream protocol shaping helpers."""

import json
import uuid
from datetime import datetime


def build_chat_completion_payload(model, content, reasoning_content=None, tool_calls=None, finish_reason="stop"):
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
                "finish_reason": "tool_calls" if tool_calls else finish_reason
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
            choice = response_data["choices"][0]
            delta = choice.get("delta") or choice.get("message") or {}
            calls = delta.get("tool_calls")
            if not calls and delta.get("function_call"):
                calls = [{"index": 0, "type": "function", "function": delta["function_call"]}]
            return {
                "reasoning": delta.get("reasoning") or delta.get("reasoning_content"),
                "content": delta.get("content"),
                "tool_calls": calls,
            }
    except (KeyError, IndexError, TypeError):
        pass
    return {"reasoning": None, "content": None}


def iter_sse_events(lines, delta_parser=extract_delta_from_genai):
    """将上游 SSE 行转换为内部增量、元数据和结束事件。"""
    finished = False
    for line in lines:
        if not line:
            continue

        line_str = line.decode("utf-8") if isinstance(line, bytes) else line
        if line_str.startswith(":") or line_str.startswith("event:"):
            continue
        if line_str.startswith("data:"):
            line_str = line_str[5:].strip()
        if not line_str:
            continue
        if line_str == "[DONE]":
            finished = True
            yield {"type": "done", "upstream_model": None, "finish_reason": "stop"}
            break

        try:
            payload = json.loads(line_str)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue

        # 上游可能以 HTTP 200 返回业务错误，不能当作空回复结束。
        if payload.get("errMsg") or payload.get("success") is False or payload.get("error"):
            yield {
                "type": "error",
                "error": payload.get("errMsg") or payload.get("message") or str(payload.get("error") or "GenAI API error"),
            }
            return

        if payload.get("other"):
            yield {"type": "meta", "other": payload["other"]}

        delta = delta_parser(payload)
        if delta.get("reasoning") is not None or delta.get("content") is not None or delta.get("tool_calls"):
            yield {
                "type": "delta",
                "upstream_model": payload.get("model"),
                "reasoning": delta.get("reasoning"),
                "content": delta.get("content"),
                "tool_calls": delta.get("tool_calls"),
            }

        choices = payload.get("choices") or []
        if choices and choices[0].get("finish_reason") is not None:
            finished = True
            yield {
                "type": "done", "upstream_model": payload.get("model"),
                "finish_reason": choices[0]["finish_reason"],
            }
            break

    if not finished:
        yield {"type": "error", "error": "GenAI upstream stream ended without a completion marker"}


class ToolCallAccumulator:
    """按 index 聚合原生工具增量，参数在完整收集后统一验证。"""

    def __init__(self):
        self.calls = {}

    def add(self, fragments):
        for position, fragment in enumerate(fragments or []):
            index = fragment.get("index", position)
            call = self.calls.setdefault(index, {
                "id": None, "type": "function",
                "function": {"name": "", "arguments": ""},
            })
            if fragment.get("id"):
                call["id"] = fragment["id"]
            function = fragment.get("function") or {}
            for key in ("name", "arguments"):
                value = function.get(key)
                if value is not None:
                    if key == "arguments" and isinstance(value, dict):
                        value = json.dumps(value, ensure_ascii=False)
                    if not isinstance(value, str):
                        raise ValueError(f"Invalid native tool {key}")
                    call["function"][key] += value

    def result(self):
        return [self.calls[index] for index in sorted(self.calls)]


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
