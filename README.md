# GenAI 项目

## 项目简介

GenAI 是一个基于 Flask 的聊天机器人接口服务，兼容 OpenAI 的聊天完成接口，利用上海科技大学的 GenAI API 进行智能对话。项目通过封装 GenAI API，支持流式响应和普通响应，从而方便客户端集成与调用。该项目适合开发具有中文支持及本地化需求的智能聊天机器人应用。

## 安装与运行

### 环境要求

- Python 3.11 及以上版本
- 依赖包见 `pyproject.toml`，推荐使用uv管理环境。

### 启动服务

```bash
uv run main.py --token <token> --imgtoken <imgtoken> [--port 5000]
```

<token>, <imgtoken>的获取方式见下文，端口默认5000。服务将在本地 `0.0.0.0:5000` 端口启动。

## 功能和用法

- 兼容 OpenAI Chat Completion API 请求格式，支持 POST `/v1/chat/completions` 接口，实现智能聊天功能。
- 支持流式（stream）及非流式响应，方便高效地获取 AI 回复。
- 提供 `/v1/models` 接口列出可用模型，如 `gpt-3.5-turbo`、`gpt-4`、`deepseek-v3` 等。
- 内置 `/health` 健康检查接口，用于服务状态监测。

## Token获取

1. 首先前往[GenAI对话平台](https://genai.shanghaitech.edu.cn/dialogue)
2. 打开浏览器开发者工具，随便发送一条消息，捕获名为`chat`的请求
3. 复制请求标头中的`x-access-token`字段，即为`<token>`
4. 再进行一次图片上传，捕获名为`upload`的请求
5. 复制请求标头中的`Token`字段，即为`<imgtoken>`

![图片说明](images/chrome.png)

## 开发与贡献指南

- 欢迎 fork 并提交 PR，改进功能或修复 bug。
- 请遵守项目代码风格，代码中请添加必要注释。
- 贡献代码时建议附带测试，确保功能完整性。
- 遇到问题可通过 issue 反馈。

## 联系方式与许可

- 联系邮箱：arnoliu@shanghaitech.edu.cn
- 本项目采用 MIT 许可证，详见 LICENSE 文件。

## TODO

- 身份验证或者签名机制，避免滥用

