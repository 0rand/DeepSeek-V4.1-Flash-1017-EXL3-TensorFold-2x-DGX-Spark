"""Reject complete model tool calls that were not offered for a request."""

from __future__ import annotations

import json
import re
from typing import Any


class ModelOutputError(Exception):
    """A safe public error for a model attempting an unavailable tool."""

    code = "unavailable_tool_call"

    def __init__(self, tool_name: str | None = None):
        message = "The model attempted to call a tool not offered by this request."
        super().__init__(message)
        self.error = {"type": "model_output_error", "code": self.code, "message": message}
        self.tool_name = tool_name


class ToolOutputGuard:
    """Validate complete call envelopes before the CUDA parser can execute any call."""

    _call_re = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.IGNORECASE | re.DOTALL)
    _xml_re = re.compile(r"\s*<function=([^>\s]+)>\s*(.*?)\s*</function>\s*", re.IGNORECASE | re.DOTALL)
    _parameters_re = re.compile(r"<parameter=[^>\s]+>.*?</parameter>", re.IGNORECASE | re.DOTALL)
    _glm_re = re.compile(r"\s*([\w.:-]+)(.*)", re.DOTALL)
    _glm_args_re = re.compile(r"<arg_key>.*?</arg_key>\s*<arg_value>.*?</arg_value>", re.DOTALL)
    _name_re = re.compile(r"[\w.:-]+\Z")

    def __init__(self, tools: list[dict[str, Any]] | None):
        self.tools = tools or []

    def parse(self, text: str, max_calls: int | None = None) -> tuple[str, list[dict[str, Any]] | None]:
        """Return CUDA's parsed result, or fail atomically on an unavailable complete call."""

        from tensorfold.cuda import reply_text
        from tensorfold.server.tools import tool_spec_name

        known = {tool_spec_name(tool).lower() for tool in self.tools if tool_spec_name(tool)}
        literal = self._literal_spans(text)
        candidates: list[tuple[int, int, list[str]]] = []
        for block in reply_text._DSML_BLOCK_RE.finditer(text):
            names = [match.group(1).strip() for match in reply_text._DSML_INVOKE_RE.finditer(block.group(1))]
            names = [name for name in names if self._name_re.fullmatch(name)]
            candidates.append((block.start(), block.end(), names))
        for block in self._call_re.finditer(text):
            name = self._call_name(block.group(1))
            candidates.append((block.start(), block.end(), [name] if name else []))
        # A closing tag inside a JSON string is data, not the envelope's end.
        for opening in re.finditer(r"<tool_call>\s*", text, re.IGNORECASE):
            try:
                _, size = json.JSONDecoder().raw_decode(text, opening.end())
            except json.JSONDecodeError:
                continue
            closing = re.match(r"\s*</tool_call>", text[size:], re.IGNORECASE)
            if closing:
                name = self._call_name(text[opening.end():size])
                candidates.append((opening.start(), size + closing.end(), [name] if name else []))

        end = -1
        envelopes = []
        for start, stop, names in sorted(candidates, key=lambda item: (item[0], -item[1])):
            if start < end:
                continue
            end = stop
            is_literal = self._literal(start, stop, text, literal)
            envelopes.append((start, stop, is_literal))
            if is_literal:
                continue
            for name in names:
                if name.lower() not in known:
                    raise ModelOutputError(name)

        # Parse real top-level envelopes without masking or rewriting argument values.
        content, calls, cursor = [], [], 0
        for start, stop, is_literal in envelopes:
            content.append(text[cursor:start])
            block = text[start:stop]
            if is_literal:
                content.append(block)
            else:
                remaining = None if max_calls is None else max(0, max_calls - len(calls))
                residue, parsed = reply_text.parse_tool_calls(block, self.tools, max_calls=remaining)
                content.append(residue)
                calls.extend(parsed or [])
            cursor = stop
        content.append(text[cursor:])
        result = "".join(content)
        return (result if any(item[2] for item in envelopes) else result.strip()), calls or None

    def _call_name(self, block: str) -> str | None:
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            function = payload.get("function")
            source = function if isinstance(function, dict) else payload
            name = source.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
        xml = self._xml_re.fullmatch(block)
        if xml and not self._parameters_re.sub("", xml.group(2)).strip():
            return xml.group(1).strip()
        glm = self._glm_re.fullmatch(block)
        if glm and not self._glm_args_re.sub("", glm.group(2)).strip():
            return glm.group(1).strip()
        return None

    @staticmethod
    def _literal_spans(text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        fence: tuple[str, int, int] | None = None
        offset = 0
        for line in text.splitlines(keepends=True):
            marker = re.match(r"[ \t]*(`{3,}|~{3,})", line)
            if marker:
                token = marker.group(1)
                if fence is None:
                    fence = (token[0], len(token), offset)
                elif token[0] == fence[0] and len(token) >= fence[1] and not line[marker.end():].strip():
                    spans.append((fence[2], offset + len(line)))
                    fence = None
            offset += len(line)
        if fence is not None:
            spans.append((fence[2], len(text)))
        cursor = 0
        for match in re.finditer(r"`+", text):
            if match.start() < cursor or any(start <= match.start() < stop for start, stop in spans):
                continue
            close = re.search(r"(?<!`)`{" + str(len(match.group())) + r"}(?!`)", text[match.end():])
            if close:
                cursor = match.end() + close.end()
                spans.append((match.start(), cursor))
        return spans

    @staticmethod
    def _literal(start: int, stop: int, text: str, spans: list[tuple[int, int]]) -> bool:
        if any(left <= start < right for left, right in spans):
            return True
        before, after = text[:start].rstrip(), text[stop:].lstrip()
        if before and after and before[-1] in "\"'" and after[0] == before[-1]:
            return True
        line_start = text.rfind("\n", 0, start) + 1
        return re.fullmatch(r"[ \t]*(?:>[ \t]*)+", text[line_start:start]) is not None
