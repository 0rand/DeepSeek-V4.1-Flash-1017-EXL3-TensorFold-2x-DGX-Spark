"""Small stdlib parser double. It is not a TensorFold compatibility test."""
import json
import re

_DSML_BLOCK_RE = re.compile(r"<｜DSML｜ calls>(.*?)</｜DSML｜ calls>", re.DOTALL)
_DSML_INVOKE_RE = re.compile(r'<｜DSML｜ invoke name="([^"]+)">(.*?)</｜DSML｜ invoke>', re.DOTALL)


def tool_spec_name(tool):
    return (tool.get("function") or tool).get("name", "")


def parse_tool_calls(text, tools, max_calls=None):
    known = {tool_spec_name(tool).lower(): tool_spec_name(tool) for tool in tools}
    calls = []
    payload = None
    if text.startswith("<｜DSML｜ calls>"):
        inv = _DSML_INVOKE_RE.search(text)
        if inv and not inv.group(2):
            payload = {"name": inv.group(1), "arguments": {}}
    elif text.lower().startswith("<tool_call>") and text.lower().endswith("</tool_call>"):
        body = text[len("<tool_call>"):-len("</tool_call>")].strip()
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            xml = re.fullmatch(r"<function=([^>]+)>(.*?)</function>", body, re.DOTALL)
            if xml:
                args = dict(re.findall(r"<parameter=([^>]+)>(.*?)</parameter>", xml.group(2), re.DOTALL))
                payload = {"name": xml.group(1), "arguments": args}
            else:
                glm = re.fullmatch(r"([\w.:-]+)((?:<arg_key>.*?</arg_key><arg_value>.*?</arg_value>)*)", body, re.DOTALL)
                if glm:
                    payload = {"name": glm.group(1), "arguments": dict(re.findall(
                        r"<arg_key>(.*?)</arg_key><arg_value>(.*?)</arg_value>", glm.group(2), re.DOTALL))}
    if isinstance(payload, dict):
        source = payload.get("function") or payload
        name = source.get("name", "")
        if name.lower() in known:
            if max_calls is None or max_calls > 0:
                args = source.get("arguments", {})
                calls.append({"id": "fixture-call", "type": "function", "function": {
                    "name": known[name.lower()], "arguments": args if isinstance(args, str) else json.dumps(args, separators=(',', ':'))}})
            return "", calls or None
    return text, None
