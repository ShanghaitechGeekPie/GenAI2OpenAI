# GenAI 项目

## 项目简介

GenAI 是一个基于 Flask 的聊天机器人接口服务，兼容 OpenAI 的聊天完成接口，利用上海科技大学的 GenAI API 进行智能对话。项目通过封装 GenAI API，支持思维链、流式响应和普通响应，从而方便客户端集成与调用。该项目适合开发具有中文支持及本地化需求的智能聊天机器人应用。

## 安装与运行

### 环境要求

- Python 3.11 及以上版本
- 依赖包见 `pyproject.toml`，推荐使用uv管理环境。

### 启动服务

```bash
uv run main.py --token <token> [--port 5000]
```

token 的获取方式见下文，端口默认5000。服务将在本地 `0.0.0.0:5000` 端口启动。

## 功能和用法

- 兼容 OpenAI API，支持 `POST /v1/chat/completions`、`POST /v1/responses`接口，实现智能聊天功能。
- 支持流式（stream）及非流式响应，方便高效地获取 AI 回复。
- `POST /v1/chat/completions` 支持基于提示词工程和 JSON 解析的 OpenAI `tools`/`tool_choice` 兼容工具调用，也兼容旧版 `functions`/`function_call` 入参。
- 提供 `/v1/models` 接口列出可用模型，如 `deepseek-v4-pro`、`gpt-5.5`、`glm-5.1` 等。
- 内置 `/health` 健康检查接口，用于服务状态监测。

### 支持模型

| 模型 id | 可用性 | 思维链 | 实测上下文长度 |
|---|---|---|---|
| deepseek-r1 | ✅ | ✅ | ~100k-128k tokens |
| deepseek-v3 | ✅ | ❌ | ≥200k tokens |
| glm-5.1 | ✅ | ❌ | ≥200k tokens |
| minimax-m1 | ✅ | ✅ | ≥200k tokens |
| qwen3.5-397b-a17b | ✅ | ✅ | <100k tokens |
| gpt-5.5 | ✅ | 隐藏 | 未测试（额度限制） |
| gpt-5.4 | ✅ | 隐藏 | 未测试（额度限制） |
| gpt-5.2 | ✅ | 隐藏 | 未测试（额度限制） |
| gpt-5 | ✅ | 隐藏 | 未测试（额度限制） |
| gpt-4.1 | ✅ | 隐藏 | 未测试（额度限制） |
| gpt-4.1-mini | ✅ | 隐藏 | 未测试（额度限制） |
| gpt-o4-mini | ✅ | 隐藏 | 未测试（额度限制） |
| gpt-o3 | ✅ | 隐藏 | 未测试（额度限制） |
| deepseek-v4-pro | ❌ | 未知 | - |
| deepseek-v4-flash | ❌ | 未知 | - |

兼容层同时兼容历史请求名和底层模型名，详见[模型列表](docs/模型列表.md)。
以上信息最后更新于 `2026-04-29`。

### 测试模型上下文长度

项目内置 `context_length_tester` skill，可用于测试模型的实际上下文处理能力：

```bash
# 大海捞针测试（推荐）
uv run tools/skills/context_length_tester/context_length_tester.py --model deepseek-v3

# 快速探测 API 上限
uv run tools/skills/context_length_tester/context_length_tester.py --model deepseek-v3 --mode probe
```

测试方法采用**大海捞针法**（Needle in a Haystack）：在长文本中间插入关键信息，验证模型能否准确检索。这比简单的二分查找更能反映模型的真实上下文处理能力。

**注意**：Azure GPT 模型有严格的额度限制，无法进行上下文长度测试。建议参考各模型的官方文档了解其标称上下文长度。

### 工具调用兼容

上游 GenAI API 没有原生 tool calling 能力。本项目在 Chat Completions 接口中通过系统提示词要求模型输出工具调用 JSON，并在本地解析为 OpenAI 兼容的 `tool_calls`：

```json
{
  "model": "gpt-5.5",
  "messages": [
    {"role": "user", "content": "上海今天适合带伞吗？"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "查询城市天气",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string"}
          },
          "required": ["city"]
        }
      }
    }
  ],
  "tool_choice": "auto"
}
```

如果模型决定调用工具，非流式响应会返回 `finish_reason: "tool_calls"` 和 `message.tool_calls`。流式请求也会返回兼容的 `tool_calls` chunk，但为了可靠解析 JSON，带工具的流式请求会先在服务端收集完整上游输出后再发送结果。

## Token获取

1. 首先前往[GenAI对话平台](https://genai.shanghaitech.edu.cn/dialogue)
2. 打开浏览器开发者工具，随便发送一条消息，捕获名为`chat`的请求
3. 复制请求标头中的`x-access-token`字段，即为`<token>`

服务启动时可通过 `--token <token>` 设置默认 GenAI token。客户端也可以通过传统的 API key 传递token，此时请求级 key 会覆盖启动参数中的默认 token，并作为上游 GenAI 的 `X-Access-Token` 使用。

支持的请求头：

- `Authorization: Bearer <token>`（推荐，兼容 OpenAI SDK）
- `X-Access-Token: <token>`
- `api-key: <token>`
- `X-API-Key: <token>`

示例：

```bash
curl http://127.0.0.1:5000/v1/chat/completions \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v3","messages":[{"role":"user","content":"你好"}]}'
```

![图片说明](images/chrome.png)

## 开发与贡献指南

- 欢迎 fork 并提交 PR，改进功能或修复 bug。
- 请遵守项目代码风格，代码中请添加必要注释。
- 贡献代码时建议附带测试，确保功能完整性。
- 遇到问题可通过 issue 反馈。

## 联系方式与许可

- 联系邮箱：arnoliu@shanghaitech.edu.cn
- 本项目采用 MIT 许可证，详见 LICENSE 文件。
