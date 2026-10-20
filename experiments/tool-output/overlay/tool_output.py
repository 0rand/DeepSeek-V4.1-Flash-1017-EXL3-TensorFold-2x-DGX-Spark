"""Reject complete model tool calls that were not offered for a request."""

from __future__ import annotations

import json
import hashlib
import re
import uuid
from typing import Any


class ModelOutputError(Exception):
    """A safe public error for a model attempting an unavailable tool."""

    code = "unavailable_tool_call"

    def __init__(self, tool_name: str | None = None, *, diagnostic: dict | None = None):
        self.rejection_id = uuid.uuid4().hex[:16]
        reason = (diagnostic or {}).get("reason", "unavailable_tool")
        message = ("The model attempted to call a tool not offered by this request. "
                   f"Reason: {reason}; source: tensorfold.tool_output_guard; diagnostic ID: {self.rejection_id}.")
        super().__init__(message)
        self.error = {"type": "model_output_error", "code": self.code, "message": message,
                      "source": "tensorfold.tool_output_guard", "reason": reason, "rejection_id": self.rejection_id}
        self.tool_name = tool_name
        self.diagnostic = {**(diagnostic or {}), "rejection_id": self.rejection_id,
                           "source": self.error["source"], "code": self.code,
                           "tool_name": (tool_name or "")[:96]}

    def log_line(self) -> str:
        return f"[tensorfold] model output rejected: {self.code} " + json.dumps(self.diagnostic, ensure_ascii=True)


class ToolOutputGuard:
    """Validate complete call envelopes before the CUDA parser can execute any call."""

    _call_re = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.IGNORECASE | re.DOTALL)
    _xml_re = re.compile(r"\s*<function=([^>\s]+)>\s*(.*?)\s*</function>\s*", re.IGNORECASE | re.DOTALL)
    _parameters_re = re.compile(r"<parameter=[^>\s]+>.*?</parameter>", re.IGNORECASE | re.DOTALL)
    _glm_re = re.compile(r"\s*([\w.:-]+)(.*)", re.DOTALL)
    _glm_args_re = re.compile(r"<arg_key>.*?</arg_key>\s*<arg_value>.*?</arg_value>", re.DOTALL)
    _name_re = re.compile(r"[\w.:-]+\Z")

    def __init__(self, tools: list[dict[str, Any]] | None, *, request: dict | None = None):
        self._stream_cut: int | None = None
        self.tools = tools or []
        self.declared_tools = (request.get("tools") or []) if request is not None else self.tools
        self.tools_field = ("omitted" if request is not None and "tools" not in request else
                            "nonempty" if self.declared_tools else "empty")
        choice = request.get("tool_choice", "auto") if request is not None else "auto"
        if isinstance(choice, dict):
            function = choice.get("function")
            name = function.get("name") if isinstance(function, dict) else None
            self.tool_choice = {"type": "function", "name": str(name or "")[:96]}
        else:
            self.tool_choice = str(choice)[:96]

    _openers = ("<tool_call>", "<｜DSML｜ calls>")

    def stream(self, text: str) -> str:
        """Release cumulative prose until a candidate, then validate at the end.

        Only a possible opening tag suffix waits for more characters. Once an
        executable opener occurs, the rest waits for final atomic validation.
        No call delta can escape before a later unoffered call is checked.
        """
        if self._stream_cut is not None:
            return text[:self._stream_cut]
        for opening in re.finditer(r"<tool_call>|<｜DSML｜ calls>", text, re.IGNORECASE):
            if not self._literal_prefix(text[:opening.start()]):
                self._stream_cut = opening.start()
                return text[:self._stream_cut]
        lower = text.lower()
        held = max((size for tag in self._openers for size in range(1, len(tag))
                    if lower.endswith(tag[:size].lower())), default=0)
        return text[:-held] if held else text

    @staticmethod
    def _literal_prefix(prefix: str, suffix: str | None = None) -> bool:
        """Read Markdown before a call, never inside a real argument body.

        Unclosed inline backticks are conservative literals. An envelope counts as quoted only when the same
        quote mark stands right before it (spaces and tabs allowed, no newline) and right after it. That check
        needs ``suffix``, the text after the envelope. While streaming there is no suffix, so a quote never
        makes the rest literal there. This is a tool-markup guard, not a Markdown renderer.
        """
        fence = None
        inline = None
        for line in prefix.splitlines(keepends=True):
            marker = re.match(r"[ \t]*(`{3,}|~{3,})", line)
            if fence is not None:
                if (marker and marker.group(1)[0] == fence[0] and len(marker.group(1)) >= fence[1]
                        and not line[marker.end():].strip()):
                    fence = None
                continue
            if marker and inline is None:
                fence = (marker.group(1)[0], len(marker.group(1)))
                continue
            for token in re.finditer(r"`+", line):
                width = len(token.group())
                if inline is None:
                    inline = width
                elif inline == width:
                    inline = None
        last_line = prefix.rsplit("\n", 1)[-1]
        quoted = False
        if suffix is not None:
            before = last_line.rstrip(" \t")
            quoted = bool(before) and before[-1] in ('"', "'") and suffix.lstrip(" \t").startswith(before[-1])
        return (fence is not None or inline is not None
                or bool(re.fullmatch(r"[ \t]*(?:>[ \t]*)+", last_line))
                or quoted)

    def parse(self, text: str, max_calls: int | None = None) -> tuple[str, list[dict[str, Any]] | None]:
        """Return CUDA's parsed result, or fail atomically on an unavailable complete call."""

        from tensorfold.cuda import reply_text
        from tensorfold.server.tools import tool_spec_name

        known = {tool_spec_name(tool).lower() for tool in self.tools if tool_spec_name(tool)}
        candidates: list[tuple[int, int, list[str], str]] = []
        for block in reply_text._DSML_BLOCK_RE.finditer(text):
            names = [match.group(1).strip() for match in reply_text._DSML_INVOKE_RE.finditer(block.group(1))]
            names = [name for name in names if self._name_re.fullmatch(name)]
            candidates.append((block.start(), block.end(), names, "dsml"))
        for block in self._call_re.finditer(text):
            name = self._call_name(block.group(1))
            body = block.group(1).lstrip()
            kind = "json" if body.startswith("{") else "qwen_xml" if body.lower().startswith("<function=") else "glm"
            candidates.append((block.start(), block.end(), [name] if name else [], kind))
        # A closing tag inside a JSON string is data, not the envelope's end.
        for opening in re.finditer(r"<tool_call>\s*", text, re.IGNORECASE):
            try:
                _, size = json.JSONDecoder().raw_decode(text, opening.end())
            except json.JSONDecodeError:
                continue
            closing = re.match(r"\s*</tool_call>", text[size:], re.IGNORECASE)
            if closing:
                name = self._call_name(text[opening.end():size])
                candidates.append((opening.start(), size + closing.end(), [name] if name else [], "json"))

        end = -1
        envelopes = []
        masked = text
        for start, stop, names, kind in sorted(candidates, key=lambda item: (item[0], -item[1])):
            if start < end:
                continue
            end = stop
            is_literal = self._literal_prefix(masked[:start], masked[stop:])
            envelopes.append((start, stop, is_literal))
            if is_literal:
                continue
            # Argument fences are data, not Markdown for subsequent calls.
            masked = masked[:start] + ' ' * (stop - start) + masked[stop:]
            for name in names:
                if name.lower() not in known:
                    effective = sorted({tool_spec_name(tool) for tool in self.tools if tool_spec_name(tool)})
                    declared = sorted({tool_spec_name(tool) for tool in self.declared_tools if tool_spec_name(tool)})
                    raise ModelOutputError(name, diagnostic={
                        "reason": "disallowed_by_tool_choice" if name.lower() in {value.lower() for value in declared}
                                  else "unavailable_tool",
                        "wrapper": kind, "span_start": start, "span_end": stop,
                        "answer_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "effective_tools": [value[:96] for value in effective[:64]],
                        "effective_tool_count": len(effective),
                        "declared_tools": [value[:96] for value in declared[:64]],
                        "declared_tool_count": len(declared),
                        "tools_field": self.tools_field, "tool_choice": self.tool_choice,
                    })

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
        return result, calls or None

    def _call_name(self, block: str) -> str | None:
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            function = payload.get("function")
            for source in (function, payload):         # an empty nested object falls back to the flat form
                name = source.get("name") if isinstance(source, dict) else None
                if isinstance(name, str) and name.strip():
                    return name.strip()
        xml = self._xml_re.fullmatch(block)
        if xml and not self._parameters_re.sub("", xml.group(2)).strip():
            return xml.group(1).strip()
        glm = self._glm_re.fullmatch(block)
        if glm and not self._glm_args_re.sub("", glm.group(2)).strip():
            return glm.group(1).strip()
        return None
