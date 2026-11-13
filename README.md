# GenAI2OpenAI

## 写在前面

写这个项目的时候, GenAI平台还是比较好的, 当时我也没申请API. 虽然GenAI平台实际上烂完了,但胜在免费,自己用用还是可以的. 后来申请了API, 只能说难兄难弟,没比公开的好用多少. 现在我已经不缺token了,所以维护这个repo的动力很低很低. 我建议大家玩一玩genai就够了, 最多是一些高通量的任务用一下, 除此之外别折腾了. 嫌贵可以去买那些中转服务, 一个平台可以用所有模型的那种,其实是挺好用的. 最近我看到大家对于这个项目还是挺热情的,所以参考其他人的工作补全了很多很多的功能. 当然距离一个“能用”的API还有距离, 当然这个距离我也无能为力了. 一想当GenAI刚出的时候我还是非常有信心的, Yu老师亲口说这个平台很吊, 但是我只能说烂完了(除了免费). 希望大家能利用AI改善自己的生活. 最后如果不出意外这个项目基本不会再更新了,如果有本科生小朋友愿意接手的话可以联系我(issue里直接提也可以). ciallo (∠·ω )⌒★


## 项目简介

GenAI 是一个基于 Flask 的聊天机器人接口服务，兼容 OpenAI 的聊天完成接口，利用上海科技大学的 GenAI API 进行智能对话。项目通过封装 GenAI API，支持思维链、流式响应和普通响应，从而方便客户端集成与调用。该项目适合开发具有中文支持及本地化需求的智能聊天机器人应用。

## 项目结构

```text
main.py                         Flask 入口、路由和上游流式编排
genai_proxy/models.py           模型列表、别名解析和上游路由类型
genai_proxy/tools.py            消息归一化和提示词式 Tool Calling
genai_proxy/images.py           图片 URL、Data URL 和消息图片解析
genai_proxy/upstream.py         OpenAI 响应结构和输入转换
utility/auto_login/             上海科技大学 CAS 自动登录
tools/                          模型测速和上下文长度测试工具
docs/                           模型记录和问题登记
```

`main.py` 保留 HTTP 路由和请求编排；可复用的模型、消息、图片和协议转换逻辑放在 `genai_proxy/` 中，便于单独检查和测试。

**OpenAI Compatible 功能对比**

| 能力项                                    | OpenAI 官方接口 | 本项目实现情况 | 说明                                                       |
| ----------------------------------------- | --------------- | -------------- | ---------------------------------------------------------- |
| `POST /v1/chat/completions`               | ✅ 原生支持     | ✅ 已兼容      | 入参/出参保持 OpenAI 风格，转发至 GenAI 上游               |
| `POST /v1/responses`                      | ✅ 原生支持     | ✅ 最小兼容    | 支持基础 `input`、流式与非流式输出                         |
| 流式输出（SSE）                           | ✅              | ✅             | 支持 Chat Completions 与 Responses 两条链路                |
| 非流式输出                                | ✅              | ✅             | 统一聚合上游增量后返回标准 JSON                            |
| 推理内容字段（reasoning）                 | 部分模型支持    | ✅ 兼容输出    | 通过 `reasoning_content` / `response.reasoning.delta` 暴露 |
| Tool Calling（`tools/tool_choice`）       | ✅ 原生         | ✅ 提示词兼容  | 上游无原生工具调用，本项目做 JSON 约定与本地解析           |
| 旧版函数调用（`functions/function_call`） | 已逐步废弃      | ✅ 兼容        | 自动转换为 `tools/tool_choice` 语义                        |
| 图片输入（Vision）                        | ✅              | ✅（GPT 模型） | 服务端自动上传图片并注入 `imageUrl/width/height`           |
| 模型列表接口（`GET /v1/models`）          | ✅              | ✅             | 实时拉取上游列表，返回上游 `aiType` 作为模型 ID             |
| 认证头兼容（Bearer/API Key）              | ✅              | ✅             | 支持 `Authorization`、`X-Access-Token`、`api-key` 等       |

### Agent Tool 生态测试

| 客户端 / Agent | 兼容性 |
|---|---|
| Chatbox | ✅ 完美支持 | 
| Kilo Code | ❌ 不支持(模型限制) |

## 安装与运行

### 环境要求

- Python 3.11 及以上版本
- 依赖包见 `pyproject.toml`，推荐使用 uv 管理环境。

### 启动服务

```bash
uv run main.py [--token <token>] [--account <student_id@password>] [--upload-token <upload_token>] [--log-level INFO] [--port 5000]
```

端口默认 5000。服务将在本地 `0.0.0.0:5000` 端口启动。

可选参数：

- `--token` ：若不在启动时提供，可由客户端在每次请求中通过 `Authorization: Bearer <token>` 或其他兼容 API key 请求头传递。
- `--account`：上海科技大学统一身份认证账号，格式为 `学号@密码`。当未提供 `--token` 时，服务启动时会自动登录并获取 GenAI token。
- `--upload-token`：图片上传接口 `token` 请求头值（默认内置项目当前可用值）。
- `--log-level`：控制台日志级别，支持 `DEBUG / INFO / WARNING / ERROR / CRITICAL`，默认 `INFO`。

## 功能和用法

- 兼容 OpenAI API，支持 `POST /v1/chat/completions`、`POST /v1/responses`接口，实现智能聊天功能。
- 支持流式（stream）及非流式响应，方便高效地获取 AI 回复。
- `POST /v1/chat/completions` 支持基于提示词工程和 JSON 解析的 OpenAI `tools`/`tool_choice` 兼容工具调用，也兼容旧版 `functions`/`function_call` 入参。
- `POST /v1/chat/completions` 支持图片输入（服务端自动上传到 GenAI 图片服务后再发起对话），当前**仅 GPT 系列模型可用**。
- `/v1/models` 每次请求都会拉取上游模型列表，客户端可据此选择模型；列表不会混入已下线的静态模型。
- 内置 `/health` 健康检查接口，用于服务状态监测。

### 支持模型

当前模型请以 `GET /v1/models` 为准。该接口沿用对话接口的认证方式：未启用 `--account` 时，请求级 token 优先，未提供则使用启动 token；`--account` 模式使用服务端 token。返回的 `id` 是上游 `aiType`，可直接填入对话请求的 `model`。上游拉取失败返回 HTTP 502，不回退到旧列表。列表包含上游发布的不同类型模型，不代表本项目已实现所有模型对应的功能。

以下表格保留历史测速和上下文测试数据，不代表当前在线模型：

| 模型 id           | 可用性 | 思维链 | 实测上下文长度     | first_token_delay | 输出速度       |
| ----------------- | ------ | ------ | ------------------ | -------------- | -------------- |
| deepseek-r1       | ✅     | ✅     | ~100k-128k tokens  | 0.880s         | 73.04 tokens/s | 
| deepseek-v3       | ✅     | ❌     | ≥200k tokens       | 0.917s         | 83.85 tokens/s | 
| glm-5.1           | ✅     | ❌     | ≥200k tokens       | 0.874s         | 98.15 tokens/s | 
| minimax-m1        | ✅     | ✅     | ≥200k tokens       | 0.827s         | 129.94 tokens/s |
| qwen3.5-397b-a17b | ✅     | ✅     | <100k tokens       | 0.851s         | 2.47 tokens/s  | 
| gpt-5.5           | ✅     | 隐藏   | 未测试（额度限制） | 5.639s         | 128.93 tokens/s |
| gpt-5.4           | ✅     | 隐藏   | 未测试（额度限制） | 4.205s         | 107.74 tokens/s |
| gpt-5.2           | ✅     | 隐藏   | 未测试（额度限制） | 2.940s         | 142.57 tokens/s |
| gpt-5             | ✅     | 隐藏   | 未测试（额度限制） | 41.525s        | 87.74 tokens/s | 
| gpt-4.1           | ✅     | 隐藏   | 未测试（额度限制） | 2.534s         | 133.66 tokens/s |
| gpt-4.1-mini      | ✅     | 隐藏   | 未测试（额度限制） | 2.385s         | 81.96 tokens/s | 
| gpt-o4-mini       | ✅     | 隐藏   | 未测试（额度限制） | 11.030s        | 175.94 tokens/s |
| gpt-o3            | ✅     | 隐藏   | 未测试（额度限制） | 11.612s        | 254.31 tokens/s |
| deepseek-pro      | ✅     | 未知   | 未测试             | 1.012s         | 62.13 tokens/s | 
| deepseek-chat     | ✅     | 未知   | 未测试             | 0.983s         | 19.62 tokens/s | 

兼容层同时兼容历史请求名和底层模型名，详见[模型列表](docs/模型列表.md)。
以上信息最后更新于 `2026-05-08`。上游模型和额度会变化，表中的速度与上下文数据只代表记录时的实测结果。

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
  "messages": [{ "role": "user", "content": "上海今天适合带伞吗？" }],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "查询城市天气",
        "parameters": {
          "type": "object",
          "properties": {
            "city": { "type": "string" }
          },
          "required": ["city"]
        }
      }
    }
  ],
  "tool_choice": "auto"
}
```

如果模型决定调用工具，非流式响应会返回 `finish_reason: "tool_calls"` 和 `message.tool_calls`。流式请求也会返回兼容的 `tool_calls` chunk，但为了可靠解析 JSON，带工具的流式请求会先在服务端收集完整上游输出后再发送结果。连接会立即建立，等待期间每 5 秒发送 SSE 心跳；上游返回空内容时会发送明确的错误。心跳不会显示为模型正文，实际工具调用仍需等待生成完成。

服务端会先收集上游 SSE，再解析完整 JSON。上游最后一个 SSE 事件可能同时包含最后一段文本和 `finish_reason`；服务端会先处理最后一段内容，再结束本轮响应，避免工具 JSON 的末尾被截掉。

兼容性补充：

- 解析优先级为 **JSON 优先**；
- 同时兼容 XML 标签形式的工具调用块：`<tool_call>{"name":"...","arguments":{...}}</tool_call>`；
- 支持标准 `tool_calls` 对象、顶层调用数组、单个嵌套 `function` 对象和尾随文字；多个 `<tool_call>...</tool_call>` 块会按顺序解析。
- 工具名称必须来自本次请求，参数必须是合法 JSON 对象。上游正常结束且每个调用及参数完整时，允许补齐缺少的外层数组或包装对象闭合；参数、字符串或调用对象残缺时仍明确报错。正文中的 `name` 字段和普通 JSON 数据不会仅因字段名被当作工具调用。
- 普通文本响应保留上游 `length` 等结束原因。异常 EOF 不会作为正常完成；工具响应被长度限制中断时不会执行其调用。
- 上游正常结束但没有正文或工具调用时，最多重新请求一次生成；两次均为空则明确报错。重试沿用原历史，不执行客户端工具。已有正文/调用、网络错误、长度限制或异常 EOF 不会触发此重试。
- Chat Completions 工具请求也支持上游原生 `tool_calls` / `function_call` 增量，以及 `choice.message` 形式的完整结果，统一聚合后校验。`reasoning_content` 与 `reasoning` 均会识别；仅有思考而无最终输出时也受上述一次重试限制。

### 图片输入（仅 GPT 模型）

`/v1/chat/completions` 支持 OpenAI 常见多模态消息格式：

- `type: "image_url"` + `image_url.url`（可传公网图片 URL）
- `type: "input_image"` + `image_url.url` / `url`
- 支持 `data:image/...;base64,...` 的 data URL

服务端行为：

1. 从最后一条包含图片的 user message 提取图片输入。
2. 自动调用 GenAI 图片上传接口 `https://genaipic.shanghaitech.edu.cn//sys/common/upload`。
3. 将返回的 `imageUrl`、`width`、`height` 透传到上游对话请求。

限制：

- 图片能力仅对 GPT/Azure 路由模型开放（如 `gpt-5.5`、`gpt-4.1`）。
- 若对非 GPT 模型传图，请求会返回错误：`Image input is only available for GPT models`。

示例：

```bash
curl http://127.0.0.1:5000/v1/chat/completions \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.5",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "这张图里有什么？"},
          {
            "type": "image_url",
            "image_url": {
              "url": "https://example.com/demo.jpg"
            }
          }
        ]
      }
    ]
  }'
```

## Token 获取

1. 首先前往[GenAI 对话平台](https://genai.shanghaitech.edu.cn/dialogue)
2. 打开浏览器开发者工具，随便发送一条消息，捕获名为`chat`的请求
3. 复制请求标头中的`x-access-token`字段，即为`<token>`

服务启动时可通过 `--token <token>` 设置默认 GenAI token；也可通过 `--account <学号@密码>` 在启动时自动登录获取 token。客户端也可以通过传统的 API key 传递 token，此时请求级 key 会覆盖启动参数中的默认 token，并作为上游 GenAI 的 `X-Access-Token` 使用。

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

对于图片功能, 需要捕获`upload` API, 提取请求 header 中的 `token` ,然后通过 `--upload-token` 传入.

## 常见问题

### 工具调用结果被截断

最近一次测试结果与验证范围见[工具调用测试记录](docs/工具调用测试.md)。

对于 DSH 这类带较长系统提示和多个工具的客户端，兼容层会在完整上下文末尾附加工具输出格式约定。如果只收到“我先查看项目”之类的行动预告就结束，需区分模型未生成工具调用与 JSON 被截断；两者的排查方向不同。

先确认客户端发送的是 `tools` 和 `tool_choice`，并检查服务端日志中的上游 HTTP 状态。带工具的流式请求会在服务端聚合上游内容，因此客户端收到第一个 `tool_calls` chunk 前可能会等待一段时间，期间会持续收到 SSE 心跳。

如果仍然出现截断，可以用不带工具的请求确认普通流式链路，再用 `stream: false` 检查模型是否能返回完整 JSON。上游 token 失效时，服务端会收到 `Token失效，请重新登录`，这不是工具解析错误。

### Token 无效

可以直接传入新的 token：

```bash
uv run main.py --token '<token>'
```

也可以使用自动登录：

```bash
uv run main.py --account '<学号>@<密码>'
```

不要把真实 token 或密码写进仓库、Issue、日志和示例文档。

## 开发与贡献指南

- 欢迎 fork 并提交 PR，改进功能或修复 bug。
- 请遵守项目代码风格，代码中请添加必要注释。
- 贡献代码时建议附带测试，确保功能完整性。
- 遇到问题可通过 issue 反馈。

## 联系方式与许可

- 联系邮箱：arnoliu@shanghaitech.edu.cn
- 本项目采用 MIT 许可证，详见 LICENSE 文件。
