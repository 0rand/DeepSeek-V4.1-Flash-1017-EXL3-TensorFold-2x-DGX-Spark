# Connect your app

The server speaks the OpenAI API. Any app or agent that lets you set a custom OpenAI endpoint can use it.

## The three settings

| Setting | Value |
|---|---|
| Base URL | `http://127.0.0.1:8888/v1` |
| Model | `DeepSeek-V4.1-Flash-1017` |
| API key | Any text, if you did not set `API_KEY_FILE`. Your key, if you did. |

The model reads up to **1,048,576 tokens** in one conversation. Tell your app this number if it asks for a context size.

## Try it

```bash
curl http://127.0.0.1:8888/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "DeepSeek-V4.1-Flash-1017", "messages": [{"role": "user", "content": "Say hello in French."}]}'
```

The same call in Python, with the official `openai` package:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8888/v1", api_key="none")
reply = client.chat.completions.create(
    model="DeepSeek-V4.1-Flash-1017",
    messages=[{"role": "user", "content": "Say hello in French."}],
)
print(reply.choices[0].message.content)
```

## Thinking

Thinking is **off** unless a request turns it on. Use one of these fields:

| Field | Values | Example |
|---|---|---|
| `reasoning_effort` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` | `"reasoning_effort": "high"` |
| `chat_template_kwargs.enable_thinking` | `true` or `false` | `"chat_template_kwargs": {"enable_thinking": true}` |
| `chat_template_kwargs.thinking` | `true` or `false` | `"chat_template_kwargs": {"thinking": true}` |

- Thinking on with no effort set means `high`.
- The thinking text comes back in `reasoning_content`, apart from the answer in `content`.

## Tools

Send `tools` as in the OpenAI API. `tool_choice` can be:
- `"auto"` (default): the model decides;
- `"required"`: the model must call a tool;
- `{"type": "function", "function": {"name": "..."}}`: the model must call that tool.

## Replies in an exact format

| You want | Send |
|---|---|
| JSON that matches a schema | `"response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": {...}, "strict": true}}` |
| Any valid JSON | `"response_format": {"type": "json_object"}` |
| Text that matches a pattern | `"guided_regex": "\\d{4}-\\d{2}-\\d{2}"` |
| One of a list of answers | `"guided_choice": ["yes", "no"]` |

These work with thinking on: the model thinks freely, then the answer follows the format.

## Sampling

The server default is `temperature` 1.0 and `top_p` 1.0, as DeepSeek recommends for V4.1. A value in the request wins.

## Use it from another computer

The server listens only on the head itself by default. To serve your local network:

1. Make a key file with one key in it:

```bash
openssl rand -hex 24 > ~/tfv41-api-key
```

2. In `.env`, set `BIND_HOST=0.0.0.0` and `API_KEY_FILE=` the full path of that file.
3. Restart:

```bash
./stop.sh && ./start.sh
```

4. In your app, use `http://<head address>:8888/v1` and the key.

Do not open port 8888 to the internet.

## Other endpoints

| Endpoint | What |
|---|---|
| `/v1/models` | the model name (and `SERVED_ALIAS`, if set) |
| `/health` | 200 when the server answers; only a real reply proves that both computers work |
| `/v1/completions` | plain text completion |
| `/v1/responses`, `/v1/messages` | OpenAI Responses and Anthropic Messages formats (the engine offers them; this recipe did not test them) |
| `/tokenize`, `/detokenize` | text to tokens and back |
