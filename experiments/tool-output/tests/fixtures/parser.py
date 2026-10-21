"""Reduced stdlib copy of the reply-text functions of TensorFold for the offline tests (not a compatibility test).

The functions follow `src/tensorfold/cuda/reply_text.py` and `src/tensorfold/server/tools.py` at revision
07aa02585f64 (Apache-2.0; see ../../NOTICE.md). Parameter values stay strings here: the real parser decodes them by the
tool schema. The block rules, the names and the DSML rewrite are the real ones, because the guard reads them.
"""
import json
import re
import uuid

_CALL_OPEN, _CALL_CLOSE = "<tool_call>", "</tool_call>"
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.IGNORECASE | re.DOTALL)
_TOOL_FUNCTION_BLOCK_RE = re.compile(r"^\s*<function=([^>\s]+)>\s*(.*?)\s*</function>\s*$", re.IGNORECASE | re.DOTALL)
_TOOL_PARAMETER_BLOCK_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.IGNORECASE | re.DOTALL)
_GLM_ARG_RE = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)
_GLM_NAME_RE = re.compile(r"[\w.:-]+")

_DSML = "｜DSML｜"
_DSML_OPEN, _DSML_CLOSE = f"<{_DSML} calls>", f"</{_DSML} calls>"
_DSML_BLOCK_RE = re.compile(re.escape(_DSML_OPEN) + r"\s*(.*?)\s*" + re.escape(_DSML_CLOSE), re.DOTALL)
_DSML_INVOKE_RE = re.compile(rf'<{_DSML} invoke name="([^"]+)">\s*(.*?)\s*</{_DSML} invoke>', re.DOTALL)
_DSML_PARAM_RE = re.compile(rf'<{_DSML} parameter name="([^"]+)"(?: string="(true|false)")?>(.*?)</{_DSML} parameter>',
                            re.DOTALL)


def tool_spec_name(tool):
    function = tool.get("function") if isinstance(tool, dict) else None
    if isinstance(function, dict):
        return str(function.get("name") or "").strip()
    return str(tool.get("name") or "").strip() if isinstance(tool, dict) else ""


def _tool_name(tool):
    fn = tool.get("function") if isinstance(tool, dict) else None
    return str((fn or tool).get("name") or "").strip() if isinstance(tool, dict) else ""


def dsml_to_tool_calls(text):
    if _DSML_OPEN not in text:
        return text

    def block(m):
        out = []
        for inv in _DSML_INVOKE_RE.finditer(m.group(1)):
            args = {}
            for p in _DSML_PARAM_RE.finditer(inv.group(2)):
                key, is_str, value = p.group(1), p.group(2), p.group(3)
                if is_str == "false":
                    try:
                        args[key] = json.loads(value)
                        continue
                    except json.JSONDecodeError:
                        pass
                args[key] = value
            out.append(_CALL_OPEN + json.dumps({"name": inv.group(1), "arguments": args}, ensure_ascii=False)
                       + _CALL_CLOSE)
        return "".join(out)

    return _DSML_BLOCK_RE.sub(block, text)


def parse_glm_tool_call_block(block, tools, *, complete=False):
    name, found, rest = block.partition("<arg_key>")
    name = name.strip()
    if _GLM_NAME_RE.fullmatch(name) is None:
        return None
    rest = found + rest
    if complete and _GLM_ARG_RE.sub("", rest).strip():
        return None
    return name, {key.strip(): value for key, value in _GLM_ARG_RE.findall(rest)}


def parse_tool_calls(text, tools, *, max_calls=None):
    if not tools:
        return text, None
    text = dsml_to_tool_calls(text)
    known = {_tool_name(t).lower(): _tool_name(t) for t in tools}
    calls, residue, cursor = [], [], 0
    for match in _TOOL_CALL_BLOCK_RE.finditer(text):
        residue.append(text[cursor:match.start()])
        cursor = match.end()
        if max_calls is not None and len(calls) >= max_calls:
            continue
        block = match.group(1).strip()
        name, args = None, {}
        try:
            payload = json.loads(block)
            if isinstance(payload, dict):
                fn = payload.get("function") if isinstance(payload.get("function"), dict) else payload
                name = fn.get("name")
                args = fn.get("arguments", fn.get("parameters", {}))
                if isinstance(args, str):
                    args = json.loads(args) if args.strip() else {}
        except (json.JSONDecodeError, AttributeError):
            m = _TOOL_FUNCTION_BLOCK_RE.match(block)
            if m:
                if max_calls is not None and _TOOL_PARAMETER_BLOCK_RE.sub("", m.group(2)).strip():
                    continue
                name = m.group(1).strip()
                args = {p.group(1).strip(): p.group(2) for p in _TOOL_PARAMETER_BLOCK_RE.finditer(m.group(2))}
            else:
                glm = parse_glm_tool_call_block(block, tools, complete=max_calls is not None)
                if glm is not None:
                    name, args = glm
        if not name or str(name).lower() not in known:
            if max_calls is None:
                residue.append(match.group(0))
            continue
        if max_calls is not None:
            try:
                if not isinstance(args, dict):
                    continue
                json.dumps(args, allow_nan=False)
            except (ValueError, TypeError):
                continue
        calls.append({"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                      "function": {"name": known[str(name).lower()],
                                   "arguments": json.dumps(args, ensure_ascii=False, separators=(",", ":"))}})
    residue.append(text[cursor:])
    return "".join(residue).strip(), calls or None
