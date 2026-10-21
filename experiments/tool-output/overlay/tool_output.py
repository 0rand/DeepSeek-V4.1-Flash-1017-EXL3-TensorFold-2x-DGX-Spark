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
    _opener_re = re.compile(r"<tool_call>|<｜DSML｜ calls>", re.IGNORECASE)

    def __init__(self, tools: list[dict[str, Any]] | None, *, request: dict | None = None):
        self._stream_cut: int | None = None
        self._stream_scanned = 0      # no opener starts before this offset of the cumulative answer
        self._stream_seen = ""        # the answer of the last stream() call
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
        No call delta can escape before a later unoffered call is checked. An opener that follows a code fence,
        inline code or a quote line also waits: they can end inside the call, and only parse() sees the whole call.
        """
        if not text.startswith(self._stream_seen):
            # The text does not extend the text of the last call (a reset or a rewrite): trust nothing, scan again.
            self._stream_cut, self._stream_scanned = None, 0
        self._stream_seen = text
        if self._stream_cut is not None:
            return text[:self._stream_cut]
        # Normally the answer only grows, so each call scans the new text plus the longest opener prefix again.
        longest = max(len(tag) for tag in self._openers)
        first = min(self._stream_scanned, len(text))
        for opening in self._opener_re.finditer(text, first):
            # Open code or a quote line can end inside the call (a blank line ends a quote), so only parse() may
            # call an opener literal, when it sees the whole call.
            self._stream_cut = opening.start()
            return text[:self._stream_cut]
        tail = text[-(longest - 1):].lower()
        held = max((size for tag in self._openers for size in range(1, len(tag))
                    if tail.endswith(tag[:size].lower())), default=0)
        self._stream_scanned = max(first, len(text) - (longest - 1))
        return text[:-held] if held else text

    @staticmethod
    def _fenced_blocks(masked: str) -> list[tuple[int, int | None]]:
        """The code fences of ``masked`` as (offset where the opener line ends, offset of the closer line or None).

        This is the only Markdown the guard reads. An opener is a line of three or more backticks or tildes with at
        most three spaces of indent; for backticks the info string must hold no backtick. A closer is a line of at
        least as many of the same character, with at most three spaces of indent and nothing after it. The text may
        use CRLF line ends. Anything else (quotes, quote lines, inline code, indented code) is not read.
        """
        blocks: list[tuple[int, int | None]] = []
        fence: tuple[str, int, int] | None = None
        offset = 0
        for line in masked.split("\n"):
            base, offset = offset, offset + len(line) + 1
            line = line.rstrip("\r")
            if fence is None:
                match = re.fullmatch(r" {0,3}(`{3,}|~{3,})([^\n]*)", line)
                if match and not (match.group(1)[0] == "`" and "`" in match.group(2)):
                    fence = (match.group(1)[0], len(match.group(1)), offset)
            elif re.fullmatch(r" {0,3}" + re.escape(fence[0]) + "{" + str(fence[1]) + r",}[ \t]*", line):
                blocks.append((fence[2], base))
                fence = None
        if fence is not None:
            blocks.append((fence[2], None))
        return blocks

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

        def check(start: int, stop: int, names: list[str], kind: str) -> None:
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

        end = -1
        chosen, crossing = [], []
        for start, stop, names, kind in sorted(candidates, key=lambda item: (item[0], -item[1])):
            if start < end:
                if stop > end:                       # it starts inside a span and ends after it: not data, so check it
                    crossing.append((start, stop, names, kind))
                continue
            end = stop
            chosen.append((start, stop, names, kind))
        for start, stop, names, kind in crossing:
            check(start, stop, names, kind)
        # Text of a call is data, not Markdown: replace it by a filler (newlines stay) before the code fences are read.
        # The filler is not white space, so a line with a call after a fence mark is still not a closing line.
        masked = text
        for start, stop, _, _ in sorted(chosen + crossing, reverse=True):   # a crossing call is data, too
            head, newline, rest = masked[start:stop].partition("\n")
            # The first line keeps its backticks: after a fence mark they make an invalid info string.
            masked = masked[:start] + re.sub(r"[^`]", "x", head) + newline + re.sub(r"[^\n]", "x", rest) + masked[stop:]
        fences = self._fenced_blocks(masked)
        envelopes = []
        for start, stop, names, kind in chosen:
            # A call is literal text only inside a closed code fence that began before it and ends after it.
            is_literal = any(opened <= start and closed is not None and closed >= stop for opened, closed in fences)
            envelopes.append((start, stop, is_literal))
            if is_literal:
                continue
            check(start, stop, names, kind)

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
