# GenAI2OpenAI

GenAI2OpenAI is a Flask proxy for ShanghaiTech's GenAI service. It provides OpenAI-style Chat Completions, basic Responses, streaming, reasoning fields, tool-call conversion and GPT image input.

## Maintenance

The original maintainer built this proxy around GenAI's free service and now maintains it infrequently. Upstream response quality and reliability determine the experience. Community contributions expanded the feature set; students interested in taking over maintenance can open an issue or contact the maintainer.

## Structure

```text
main.py                         Flask routes and upstream orchestration
genai_proxy/models.py           Live catalog, aliases and routing types
genai_proxy/tools.py            Message normalization and prompted tool calls
genai_proxy/images.py           Image URLs, data URLs and image messages
genai_proxy/upstream.py         OpenAI response and input conversion
utility/auto_login/             ShanghaiTech CAS login
tools/                          Speed and context-length checks
docs/                           Model records and issue register
```

HTTP orchestration lives in `main.py`; reusable model, message, image and protocol logic lives in `genai_proxy/`.

## Supported routes

| Capability | Proxy behavior |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI-style requests and responses forwarded to GenAI |
| `POST /v1/responses` | Basic `input`, streaming and non-streaming output |
| SSE and non-streaming JSON | Supported on both chat routes |
| Reasoning | `reasoning_content` / `response.reasoning.delta` |
| `tools` / `tool_choice` | Prompted JSON conversion, plus recognition of upstream native fields |
| Legacy `functions` / `function_call` | Converted to tool semantics |
| Image input | GPT/Azure routes; automatic upload with `imageUrl`, `width`, `height` |
| `GET /v1/models` | Live upstream catalog using `aiType` as the model ID |
| Authentication | Bearer token and compatible API-key headers |
| `GET /health` | Health check |

Historical client reports list Chatbox as supported and Kilo Code as unsupported because of model limitations. See the [validation record](docs/工具调用测试.md) for protocol checks and pending client acceptance.

## Run

Use Python 3.11 or newer. Dependencies are in `pyproject.toml`; `uv` manages the environment.

```bash
uv run main.py [--token <token>] [--account <student_id@password>] [--upload-token <upload_token>] [--log-level INFO] [--port 5000]
```

The service binds to `0.0.0.0:5000` by default.

| Option | Behavior |
| --- | --- |
| `--token` | Default upstream token; clients can supply a request token |
| `--account` | CAS credentials as `student_id@password`; logs in at startup when no token is supplied |
| `--upload-token` | Image-upload `token` header; the current implementation includes a built-in default |
| `--log-level` | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`; default `INFO` |
| `--port` | Listen port; default `5000` |

Review the [security issue register](docs/安全问题登记.md) before exposing the service to other users. Store real tokens and passwords outside repositories, issues, logs and documentation.

## Models

Use `GET /v1/models` to select a currently published upstream ID. In normal mode, a request token takes precedence over the startup token. Account mode uses the server-managed token. Upstream catalog failures return HTTP 502. The catalog includes upstream model types beyond the routes implemented by this proxy.

The following measurements were recorded on 2026-05-08. Availability, speed and context measurements describe that run; current service and quota conditions come from the upstream provider.

| Model | Recorded available | Reasoning | Measured context | First-token delay | Output speed |
| --- | --- | --- | --- | --- | --- |
| deepseek-r1       | ✅     | ✅     | ~100k-128k tokens  | 0.880s         | 73.04 tokens/s | 
| deepseek-v3       | ✅     | ❌     | ≥200k tokens       | 0.917s         | 83.85 tokens/s | 
| glm-5.1           | ✅     | ❌     | ≥200k tokens       | 0.874s         | 98.15 tokens/s | 
| minimax-m1        | ✅     | ✅     | ≥200k tokens       | 0.827s         | 129.94 tokens/s |
| qwen3.5-397b-a17b | ✅     | ✅     | <100k tokens       | 0.851s         | 2.47 tokens/s  | 
| gpt-5.5           | ✅     | Hidden   | Untested (quota) | 5.639s         | 128.93 tokens/s |
| gpt-5.4           | ✅     | Hidden   | Untested (quota) | 4.205s         | 107.74 tokens/s |
| gpt-5.2           | ✅     | Hidden   | Untested (quota) | 2.940s         | 142.57 tokens/s |
| gpt-5             | ✅     | Hidden   | Untested (quota) | 41.525s        | 87.74 tokens/s | 
| gpt-4.1           | ✅     | Hidden   | Untested (quota) | 2.534s         | 133.66 tokens/s |
| gpt-4.1-mini      | ✅     | Hidden   | Untested (quota) | 2.385s         | 81.96 tokens/s | 
| gpt-o4-mini       | ✅     | Hidden   | Untested (quota) | 11.030s        | 175.94 tokens/s |
| gpt-o3            | ✅     | Hidden   | Untested (quota) | 11.612s        | 254.31 tokens/s |
| deepseek-pro      | ✅     | Unknown   | Untested             | 1.012s         | 62.13 tokens/s | 
| deepseek-chat     | ✅     | Unknown   | Untested             | 0.983s         | 19.62 tokens/s | 
Historical request aliases and underlying model names are listed in the [model records](docs/模型列表.md).

### Measure context length

```bash
# Needle-in-a-haystack retrieval
uv run tools/skills/context_length_tester/context_length_tester.py --model deepseek-v3

# Probe the API input limit
uv run tools/skills/context_length_tester/context_length_tester.py --model deepseek-v3 --mode probe
```

The retrieval check inserts a key fact halfway through a long input. Azure GPT context checks remained incomplete because of quota restrictions; use current provider documentation for nominal limits.

## Tool calls

The proxy appends a tool-format instruction to the complete Chat Completions context and parses generated JSON into `tool_calls`.

```json
{
  "model": "gpt-5.5",
  "messages": [{"role": "user", "content": "Should I bring an umbrella in Shanghai today?"}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get weather for a city",
      "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
      }
    }
  }],
  "tool_choice": "auto"
}
```

Non-streaming calls return `message.tool_calls` and `finish_reason: "tool_calls"`. Tool-enabled streams establish their SSE connection immediately and send a heartbeat every five seconds while aggregating upstream output. Parsed calls arrive after generation completes. Empty output produces an explicit error. The parser consumes the final delta before its finish reason, including events containing both fields.

The conversion rules are:

- Prefer JSON and also accept `<tool_call>{"name":"...","arguments":{...}}</tool_call>` blocks.
- Accept standard `tool_calls` objects, top-level arrays, individual nested `function` objects and trailing text. Preserve multiple XML blocks in order.
- Require tool names declared by the request and arguments that parse as JSON objects. After normal upstream completion, repair only missing outer array/wrapper delimiters around complete calls. Incomplete arguments, strings and call objects fail. Ordinary JSON `name` fields remain text.
- Preserve text finish reasons such as `length`. Abnormal EOF fails, and length-truncated tool calls cannot execute.
- Retry generation once after normal completion with neither text nor calls. Reuse history and existing tool results. Published output, network/business errors, length limits and abnormal EOF end the retry path. Generation retries execute no client tools.
- Aggregate native `tool_calls` / `function_call` deltas and complete `choice.message` results, then apply the same validation. Recognize both `reasoning_content` and `reasoning`; reasoning-only output follows the same single-retry limit.

## Image input

GPT/Azure routes such as `gpt-5.5` and `gpt-4.1` accept public URLs and `data:image/...;base64,...` URLs through:

- `type: "image_url"` with `image_url.url`.
- `type: "input_image"` with `image_url.url` or `url`.

The server extracts images from the last user message containing images, uploads them to `https://genaipic.shanghaitech.edu.cn//sys/common/upload`, and forwards the returned `imageUrl`, `width` and `height` upstream. Other model routes return `Image input is only available for GPT models`.

```bash
curl http://127.0.0.1:5000/v1/chat/completions \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.5",
    "messages": [{"role": "user", "content": [
      {"type": "text", "text": "What is in this image?"},
      {"type": "image_url", "image_url": {"url": "https://example.com/demo.jpg"}}
    ]}]
  }'
```

## Obtain a token

1. Open the [GenAI dialogue platform](https://genai.shanghaitech.edu.cn/dialogue).
2. Open browser developer tools, send a message and inspect the `chat` request.
3. Copy its `x-access-token` request header for local use.

Supply the token with `--token` or a request header. Account mode can obtain a token with `--account '<student_id>@<password>'`. Outside account mode, a request token overrides the startup token and becomes upstream `X-Access-Token`.

Supported request headers:

- `Authorization: Bearer <token>`; compatible with OpenAI SDK clients.
- `X-Access-Token: <token>`.
- `api-key: <token>`.
- `X-API-Key: <token>`.

```bash
curl http://127.0.0.1:5000/v1/chat/completions \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v3","messages":[{"role":"user","content":"Hello"}]}'
```

![Browser request inspection](images/chrome.png)

For image uploads, inspect the `upload` request and pass its `token` header through `--upload-token`.

## Troubleshooting

### Tool output ends early

Check the [validation record](docs/工具调用测试.md), confirm the client sends `tools` and `tool_choice`, and inspect the upstream HTTP status and finish reason. A preamble followed by `stop` means the model produced no call; malformed call JSON requires a parser/finish-state investigation. Tool streams send heartbeats while waiting for complete upstream output.

Compare an ordinary streaming request with a non-streaming tool request (`stream: false`) to isolate transport and generation behavior. Upstream `Token失效，请重新登录` means the token needs renewal.

### Renew a token

```bash
uv run main.py --token '<token>'
# Or obtain one through CAS:
uv run main.py --account '<student_id>@<password>'
```

## Contribute

Fork the repository and open a PR for a focused fix or feature. Follow the existing code style, explain behavior changes and add checks that exercise the changed behavior. Report reproducible issues through GitHub Issues.

Contact: arnoliu@shanghaitech.edu.cn. Licensed under MIT; see `LICENSE`.
