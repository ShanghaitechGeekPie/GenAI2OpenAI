# 自动登录工具

该目录提供一个独立的上海科技大学 GenAI CAS 自动登录脚本，用于通过 `学号@密码` 获取 GenAI JWT token。

## 文件说明

- `cas_login.py`：CAS 登录流程实现
- `login_cli.py`：命令行入口

## 用法

```bash
uv run python utility/auto_login/login_cli.py --credential '2024233160@jinziyuan4644651@'
```

成功时返回：

```json
{ "success": true, "token": "eyJ..." }
```
