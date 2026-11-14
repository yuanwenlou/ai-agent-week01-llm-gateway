# 第一周作业 · LLM 统一模型调用服务

## 普通文本调用

客户端向 `POST /v1/chat/completions` 发送 `model` 和文本 `messages`。网关根据 `model` 选择适配器，由适配器转换消息、发送上游请求，再把回复放进 `choices[0].message.content`。

| 网关模型名 | 上游模型 | 协议 |
| --- | --- | --- |
| `pro` | `deepseek-v4-pro` | OpenAI Responses |
| `flash` | `deepseek-flash` | Anthropic Messages |

`deepseek-flash` 是当前的 Flash 模型 ID。凭证通过网关进程的 `DEEPSEEK_API_KEY` 环境变量读取。Responses 适配器用 Bearer 鉴权，Messages 适配器用 `x-api-key`，并将系统消息合并到上游的 `system` 字段。

当前接受 `system`、`user` 和 `assistant` 文本消息。普通调用默认关闭思考模式，输出预算为 1024 token。未知模型返回 `404 model_not_found`，无效请求返回 `422 invalid_request`，缺少上游密钥时返回 `503 upstream_not_configured`。上游失败、超时或不完整结果也通过 `error` 对象返回。

## 本地启动

在本目录使用 Python 3.10 或更新版本，执行以下 PowerShell 命令：

```powershell
python -m pip install -r requirements.txt
$env:DEEPSEEK_API_KEY = '<your DeepSeek API key>'
python -m uvicorn gateway:app --host 127.0.0.1 --port 8000
```

在另一个 PowerShell 窗口执行同一请求，分别使用两个模型名：

```powershell
foreach ($courseModel in 'pro', 'flash') {
    $courseBody = @{model = $courseModel; messages = @(@{role = 'user'; content = 'Say hello'})} | ConvertTo-Json -Depth 4 -Compress
    $courseBody | curl.exe -sS http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' --data-binary '@-'
}
```

## 流式文本调用

在同一接口传入 `stream: true`。两个模型都以 SSE 返回，文本从 `choices[0].delta.content` 读取。成功时收到结束块和 `data: [DONE]`。响应头发出后若上游失败，最后一帧含 `error`，随后流关闭。

```powershell
foreach ($courseModel in 'pro', 'flash') {
    $courseBody = @{model = $courseModel; stream = $true; messages = @(@{role = 'user'; content = 'Explain SSE in three short sentences.'})} | ConvertTo-Json -Depth 4 -Compress
    $courseBody | curl.exe -NsS -i http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' --data-binary '@-'
}
```

Uvicorn 日志中的 `first_token_ms` 从开始读取上游流计到首个非空文本增量，单位为毫秒。用响应头 `X-Request-ID` 对应日志条目。未收到文本的请求没有首 Token 延迟。

## 结构化输出和 Prompt 版本

`response_format` 支持 `{"type":"json_object"}` 和 OpenAI Chat 格式的 `json_schema`。Pro 将格式映射到 Responses 的 `text.format`。Flash 通过系统消息要求输出 JSON。网关在两条路由上都解析实际回复，按需校验 JSON Schema。校验通过时从原有的 `choices[0].message.content` 读取 JSON 字符串。流式请求与 `response_format` 组合时返回 `422 unsupported_combination`。

`prompts.json` 保存模板和版本。编辑文件后重启服务加载。模板变量写成 `${language}`，请求中用 `gateway_prompt` 指定 id、版本及字符串变量。版本不存在或必需变量缺失时，网关在上游调用前返回错误。

```powershell
$courseBody = @'
{
  "model": "pro",
  "messages": [{"role": "user", "content": "Return an answer in a JSON object."}],
  "gateway_prompt": {"id": "reviewer", "version": 2, "variables": {"language": "English"}},
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "answer",
      "schema": {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": false
      }
    }
  }
}
'@
$courseBody | curl.exe -sS -i http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' --data-binary '@-'
```

把上面的模型换成 `flash` 可验证同一个 Schema 契约。

## 用量查询 重试和限流

普通和流式请求都可通过响应头 `X-Request-ID` 找到 SQLite 记录。流式请求完成后再查询。记录模型别名、模板版本、状态、错误码、重试次数、总延迟、TTFT 和可取得的 Token 分类。普通调用的 TTFT 是 `null`，上游没有返回的 Token 分类也为 `null`。Messages 的总输入包含缓存读取和创建量，Responses 的缓存命中量是总输入的子集。费用按 2026-11-14 的 DeepSeek 峰值单价估算，约定字段缺失时不估算。重试前失败但上游未报告的用量无法计入。 [DeepSeek 定价](https://api-docs.deepseek.com/quick_start/pricing/)

```powershell
$courseRequestId = Read-Host 'X-Request-ID'
curl.exe -sS "http://127.0.0.1:8000/v1/usage/$courseRequestId"
```

- `GATEWAY_USAGE_DB` 可指定 SQLite 路径，默认 `gateway-usage.sqlite3`。父目录应已存在。
- `GATEWAY_PRO_RPM` 和 `GATEWAY_FLASH_RPM` 分别控制进程内滚动 60 秒的请求数，默认各 60。网关限流返回 `429 rate_limit_exceeded`。上游最终仍限流时使用 `upstream_rate_limited`。在流头已发出的情况下，从 SSE 错误帧读取错误码。
- 上游临时 HTTP 故障、限流和网络超时按 0.25、0.5、1 秒退避。首次调用之外最多再试 3 次。已经发出文本的流保留已发文本，出错后以错误帧结束。一次网关请求只消耗一次本地模型配额。
- 用量查询随网关绑定到本机。部署多个 worker 时，各进程分别计数。

## 验证

```powershell
python -B -m pytest -q -p no:cacheprovider
```

测试通过 FastAPI 入口发送请求，用 `httpx.MockTransport` 验证两个模型的真实模型 ID、协议路径、请求体和鉴权头。也覆盖未知模型、非法请求、缺少凭证以及上游错误。运行模拟测试无需真实密钥。

流式测试在 ASGI 发送边界观察第一段文本，模拟上游收到该信号才继续输出。它还检查首 Token 计时、正常结束和上游中断。

模拟验证覆盖两种路由、SSE、JSON 与 Schema、Prompt 版本、SQLite 用量、退避次数及按模型独立限流。验证真实服务时先设置 `DEEPSEEK_API_KEY` 并启动网关，然后在本目录运行：

```powershell
python -B verify_live.py
```

脚本对两个模型各发一次普通请求、一次带模板的 JSON 请求和一次流式请求，并读取每次的用量记录。真实调用会计入供应商用量。
