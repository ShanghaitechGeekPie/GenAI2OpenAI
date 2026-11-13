"""Message normalization and prompt-based tool-call compatibility."""

import json
import logging
import re
import uuid


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
        if role == "assistant" and tool_calls:
            normalized_messages.append({
                "role": "assistant",
                "content": (content + "\n" if content else "") + "已请求调用工具：\n" + json.dumps(tool_calls, ensure_ascii=False),
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
        "当你决定调用工具时，优先输出一个 JSON 对象，不要输出 Markdown、解释或额外文本。",
        "JSON 格式必须为：{\"tool_calls\":[{\"name\":\"工具名\",\"arguments\":{}}]}。",
        "兼容格式：也允许输出 <tool_call>{\"name\":\"工具名\",\"arguments\":{}}</tool_call>；若并行调用可连续输出多个 <tool_call>...</tool_call>。",
        "arguments 必须是符合工具 JSON Schema 的对象。",
        "如果不需要调用工具，则正常回答用户，不要输出上述 JSON。",
        "如果需要通过工具查看文件或执行操作，本轮必须输出工具调用，不能只描述接下来要做什么。",
        f"tool_choice: {json.dumps(normalized_choice, ensure_ascii=False)}",
        "可用工具：",
        json.dumps(tools, ensure_ascii=False),
    ]

    if normalized_choice == "required":
        tool_prompt.append("本次请求必须调用至少一个工具。")
    elif isinstance(normalized_choice, dict):
        tool_prompt.append(f"本次请求必须调用工具 {normalized_choice['name']}。")

    # 客户端可能已有很长的 system prompt；将上游输出格式约定放在完整
    # 上下文之后，避免模型只输出行动预告就结束本轮，而不输出工具 JSON。
    return [*messages, {"role": "system", "content": "\n".join(tool_prompt)}]


def strip_json_code_fence(text):
    """去掉模型偶尔包裹的 JSON Markdown 代码块。"""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def is_tool_call_object(value):
    """普通 JSON 的 name 字段不代表工具调用。"""
    if not isinstance(value, dict):
        return False
    function = value.get("function")
    return (
        isinstance(function, dict) and "name" in function
        or "name" in value and "arguments" in value
    )


def looks_like_tool_container(text):
    """只检查看似 JSON 的容器，不用正文里的字段名猜测工具协议。"""
    if not re.match(r'^\s*[\[{]\s*(?:[\[{\"]|$)', text):
        return False
    return bool(re.search(
        r'"tool_calls"\s*:|"function"\s*:\s*\{|"name"\s*:\s*"[^"\n]+"\s*,\s*"arguments"\s*:',
        text,
    ))


def repair_tool_wrapper(text):
    """只补齐完整调用外缺少的数组/包装对象闭合，不补参数或字符串。"""
    stack = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "[{":
            stack.append(char)
        elif char in "]}":
            if not stack or stack.pop() != ("[" if char == "]" else "{"):
                return None
    if in_string or stack not in (["["], ["{"], ["{", "["]):
        return None
    suffix = "".join("]" if char == "[" else "}" for char in reversed(stack))
    try:
        parsed = json.loads(text + suffix)
    except json.JSONDecodeError:
        return None
    calls = parsed.get("tool_calls") if isinstance(parsed, dict) else parsed
    if not isinstance(calls, list) or not calls or not all(is_tool_call_object(call) for call in calls):
        return None
    # 调用对象和参数必须在原输出中闭合；只允许其外部包装缺少闭合。
    if isinstance(parsed, dict) and set(parsed) != {"tool_calls"}:
        return None
    logging.getLogger("genai-proxy").warning(
        "Repaired missing outer tool wrapper: closing_chars=%s", len(suffix),
    )
    return parsed


def extract_json_object(text):
    """从文本中提取完整容器；仅修复完整工具调用的外层闭合。"""
    stripped = strip_json_code_fence(text)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", stripped):
        candidate = stripped[match.start():]
        try:
            value, _ = decoder.raw_decode(stripped, match.start())
            return value
        except json.JSONDecodeError:
            if looks_like_tool_container(candidate):
                repaired = repair_tool_wrapper(candidate)
                if repaired is not None:
                    return repaired
                raise ValueError("Invalid or incomplete tool call JSON")
    return None


def normalize_tool_call_arguments(arguments):
    """OpenAI 要求 function.arguments 是 JSON 字符串。"""
    if isinstance(arguments, str):
        return arguments
    if arguments is None:
        return "{}"
    return json.dumps(arguments, ensure_ascii=False)


def parse_tool_calls_from_content(content, tools=None, tool_choice=None):
    """解析工具对象、数组和 XML；只返回已声明且参数为 JSON 对象的调用。"""
    if not content:
        return []

    if "<tool_call>" in content:
        blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", content, flags=re.DOTALL)
        if len(blocks) != content.count("<tool_call>"):
            raise ValueError("Incomplete tool_call block")
        raw_calls = []
        for block in blocks:
            parsed = extract_json_object(block)
            if not isinstance(parsed, dict):
                raise ValueError("Invalid tool_call JSON")
            raw_calls.append(parsed)
    else:
        parsed = extract_json_object(content)
        if isinstance(parsed, list):
            # 普通答案也可能是 JSON 数组，只有调用形状才当作工具协议。
            if not any(is_tool_call_object(item) for item in parsed):
                raw_calls = []
            else:
                raw_calls = parsed
        elif isinstance(parsed, dict):
            raw_calls = parsed.get("tool_calls")
            if raw_calls is None and is_tool_call_object(parsed):
                raw_calls = [parsed]
            if raw_calls is None:
                raw_calls = []
        else:
            raw_calls = []

    if not isinstance(raw_calls, list):
        raise ValueError("tool_calls must be an array")
    allowed_names = None if tools is None else {
        tool.get("function", {}).get("name") for tool in tools if isinstance(tool, dict)
    }
    tool_calls = []
    for raw_call in raw_calls:
        if not isinstance(raw_call, dict):
            raise ValueError("Each tool call must be an object")
        function = raw_call.get("function") if isinstance(raw_call.get("function"), dict) else raw_call
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Tool call is missing a function name")
        if allowed_names is not None and name not in allowed_names:
            raise ValueError(f"Unknown tool: {name}")
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON arguments for tool: {name}") from exc
        if not isinstance(arguments, dict):
            raise ValueError(f"Tool arguments must be an object: {name}")
        call_id = raw_call.get("id")
        tool_calls.append({
            "id": call_id if isinstance(call_id, str) and call_id else f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": name, "arguments": normalize_tool_call_arguments(arguments)},
        })

    choice = normalize_tool_choice(tool_choice)
    if choice == "required" and not tool_calls:
        raise ValueError("Upstream did not return the required tool call")
    if isinstance(choice, dict) and (not tool_calls or any(call["function"]["name"] != choice["name"] for call in tool_calls)):
        raise ValueError(f"Upstream did not call the selected tool: {choice['name']}")
    return tool_calls
