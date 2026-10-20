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
    def _escaped(text: str, index: int) -> bool:
        """True when an odd number of backslashes stand right before ``text[index]``."""
        run = 0
        while index - run > 0 and text[index - run - 1] == "\\":
            run += 1
        return run % 2 == 1

    @staticmethod
    def _blank_code_spans(line: str) -> str:
        """``line`` with every inline code span (and its backticks) replaced by spaces; an unclosed span runs to the end."""
        out = list(line)
        opened = None
        for token in re.finditer(r"`+", line):
            if opened is None:
                if not ToolOutputGuard._escaped(line, token.start()):
                    opened = token
            elif len(token.group()) == len(opened.group()):
                out[opened.start():token.end()] = " " * (token.end() - opened.start())
                opened = None
        if opened is not None:
            out[opened.start():] = " " * (len(line) - opened.start())
        return "".join(out)

    @staticmethod
    def _in_blockquote(prefix: str) -> bool:
        return bool(re.fullmatch(r"[ \t]*(?:>[ \t]*)+", prefix.rsplit("\n", 1)[-1]))

    @staticmethod
    def _opens_quote(line: str, mark: str) -> bool:
        """True when the quote mark after ``line`` opens a span: an even number of marks come before it.

        A backslash escapes a mark. A single quote between two letters or digits is an apostrophe.
        """
        count = 0
        for index, char in enumerate(line):
            if char != mark or ToolOutputGuard._escaped(line, index):
                continue
            if (mark == "'" and 0 < index < len(line) - 1
                    and line[index - 1].isalnum() and line[index + 1].isalnum()):
                continue
            count += 1
        return count % 2 == 0

    @staticmethod
    def _closes_inside(prefix: str, body: str) -> bool:
        """True when the code fence or the inline code open at the end of ``prefix`` ends inside ``body``.

        Spans are compared by where they began, so a span that ends and another that begins inside ``body``
        still count as a close.
        """
        before = ToolOutputGuard._markdown_state(prefix)
        if before == (None, None):
            return False
        after = ToolOutputGuard._markdown_state(prefix + body)
        return any(old is not None and new != old for old, new in zip(before, after))

    @staticmethod
    def _literal_prefix(prefix: str, suffix: str | None = None, body: str | None = None) -> bool:
        """Read Markdown before a call, never inside a real argument body.

        Unclosed inline backticks are conservative literals. An envelope counts as quoted only when the same
        quote mark stands right before it (spaces and tabs allowed, no newline) and right after it. That check
        needs ``suffix``, the text after the envelope. While streaming there is no suffix, so a quote never
        makes the rest literal there. A code fence or inline code that closes inside ``body`` (the envelope)
        does not make the envelope literal: a Markdown reader sees the code end there. This is a tool-markup
        guard, not a Markdown renderer.
        """
        code = ToolOutputGuard._markdown_open(prefix)
        if code and body is not None and ToolOutputGuard._closes_inside(prefix, body):
            code = False
        last_line = prefix.rsplit("\n", 1)[-1]
        quoted = False
        if suffix is not None:
            before = ToolOutputGuard._blank_code_spans(last_line).rstrip(" \t")     # marks in code do not pair
            quoted = (bool(before) and before[-1] in ('"', "'") and suffix.lstrip(" \t").startswith(before[-1])
                      and not ToolOutputGuard._escaped(before, len(before) - 1)
                      and ToolOutputGuard._opens_quote(before[:-1], before[-1]))
        quote_line = (ToolOutputGuard._in_blockquote(prefix)
                      and (body is None or not re.search(r"\n[ \t]*(?:>[ \t]*)*\n", body)))   # a blank line ends a quote
        return code or quote_line or quoted

    @staticmethod
    def _markdown_open(prefix: str) -> bool:
        """True when ``prefix`` ends inside a code fence or inside inline code."""
        return ToolOutputGuard._markdown_state(prefix) != (None, None)

    @staticmethod
    def _closer(prefix: str) -> str:
        """Text that ends the code fence or the inline code open at the end of ``prefix`` ("" when none is open)."""
        fence, inline = ToolOutputGuard._scan_markdown(prefix)
        if fence is not None:
            return "\n" + fence[0] * fence[1] + "\n"
        return " " + "`" * inline[0] if inline else ""            # (a space keeps it apart from a backtick before it)

    @staticmethod
    def _markdown_state(prefix: str) -> tuple[int | None, int | None]:
        """Where the open code fence began and where the open inline code began in ``prefix`` (None: not open)."""
        fence, inline = ToolOutputGuard._scan_markdown(prefix)
        return (fence[2] if fence else None, inline[1] if inline else None)

    @staticmethod
    def _scan_markdown(prefix: str) -> tuple[tuple | None, tuple | None]:
        """The open fence (char, width, offset) and the open inline code (width, offset) at the end of ``prefix``."""
        fence = None
        inline = None
        offset = 0
        for line in prefix.splitlines(keepends=True):
            base, offset = offset, offset + len(line)
            marker = re.match(r" {0,3}(`{3,}|~{3,})", line)      # four spaces make indented code, not a fence
            if fence is not None:
                if (marker and marker.group(1)[0] == fence[0] and len(marker.group(1)) >= fence[1]
                        and not line[marker.end():].strip()):
                    fence = None
                continue
            if marker and inline is None:
                fence = (marker.group(1)[0], len(marker.group(1)), base)
                continue
            if inline is None and re.match(r" {4}|\t", line):
                continue                  # an indented code line: its backticks are text (the safe reading)
            for token in re.finditer(r"`+", line):
                if inline is None and ToolOutputGuard._escaped(line, token.start()):
                    continue       # an escaped backtick is text, but a backslash does nothing inside a code span
                width = len(token.group())
                if inline is None:
                    inline = (width, base + token.start())
                elif inline[0] == width:
                    inline = None
        return fence, inline

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
            body = text[start:stop]
            is_literal = self._literal_prefix(masked[:start], masked[stop:], body)
            envelopes.append((start, stop, is_literal))
            # Argument text is data, not Markdown for later calls. Newlines stay, so line starts stay true.
            # Exception: code that was open before the call and ends inside it also ends for later calls. That one
            # close stays; any delimiter after it in the call is data.
            blank = re.sub(r"[^\n]", " ", body)
            if self._closes_inside(masked[:start], body):
                closer = self._closer(masked[:start])
                if len(closer) <= len(blank):
                    blank = closer + blank[len(closer):]
            masked = masked[:start] + blank + masked[stop:]
            if is_literal:
                continue
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
