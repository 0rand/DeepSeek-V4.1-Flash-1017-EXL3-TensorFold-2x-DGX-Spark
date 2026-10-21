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

    _opener_re = re.compile(r"(?i:<tool_call>)|<｜DSML｜ calls>")

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
        No call delta can escape before a later unoffered call is checked. Every opener waits, wherever it is
        (the guard reads no Markdown): only parse() sees the whole call.
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
            self._stream_cut = opening.start()
            return text[:self._stream_cut]
        tail = text[-(longest - 1):]
        held = 0
        for tag in self._openers:                      # (the call tag in any case, the DSML tag as the parser reads it)
            exact = "DSML" in tag
            for size in range(1, len(tag)):
                if (tail if exact else tail.lower()).endswith(tag[:size] if exact else tag[:size].lower()):
                    held = max(held, size)
        self._stream_scanned = max(first, len(text) - (longest - 1))
        return text[:-held] if held else text

    def parse(self, text: str, max_calls: int | None = None) -> tuple[str, list[dict[str, Any]] | None]:
        """Return CUDA's parsed result, or fail atomically on a call that is not offered.

        The guard reads calls with the engine's own parser rules (``tensorfold.cuda.reply_text``): it rewrites DSML
        blocks and finds call blocks with the engine's own expressions, reads the name of each block as the engine
        does (also its single-call checks when ``max_calls`` is set), and rejects the answer when a name is not
        offered. The result is the engine's own ``parse_tool_calls`` on the same text, so the guard and the parser
        cannot disagree about what is a call.
        """

        from tensorfold.cuda import reply_text
        from tensorfold.server.tools import tool_spec_name

        def reject(name: str, reason: str, start: int, stop: int, kind: str) -> ModelOutputError:
            effective = sorted({tool_spec_name(tool) for tool in self.tools if tool_spec_name(tool)})
            declared = sorted({tool_spec_name(tool) for tool in self.declared_tools if tool_spec_name(tool)})
            return ModelOutputError(name, diagnostic={
                "reason": reason, "wrapper": kind, "span_start": start, "span_end": stop,
                "answer_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "effective_tools": [value[:96] for value in effective[:64]],
                "effective_tool_count": len(effective),
                "declared_tools": [value[:96] for value in declared[:64]],
                "declared_tool_count": len(declared),
                "tools_field": self.tools_field, "tool_choice": self.tool_choice,
            })

        known = {reply_text._tool_name(tool).lower() for tool in self.tools if reply_text._tool_name(tool)}
        try:
            converted = reply_text.dsml_to_tool_calls(text)
        except ValueError:                             # a value that the JSON decoder refuses (for example 5000 digits)
            raise reject("", "undecodable_call", 0, len(text), "dsml") from None
        if reply_text._CALL_CLOSE not in converted.lower():
            # No call block can exist. (The parser would try every opener against the rest of the answer.)
            return (converted.strip(), None) if self.tools else (text, None)
        # The blocks that the DSML rewrite makes, with the span of their DSML block in the answer.
        rewritten: dict[str, tuple[int, int]] = {}
        for dsml in reply_text._DSML_BLOCK_RE.finditer(text):
            for made in reply_text._TOOL_CALL_BLOCK_RE.finditer(reply_text.dsml_to_tool_calls(dsml.group(0))):
                rewritten.setdefault(made.group(0), (dsml.start(), dsml.end()))
        for block in reply_text._TOOL_CALL_BLOCK_RE.finditer(converted):
            start, stop = rewritten.get(block.group(0), (block.start(), block.end()))
            wrapper = "dsml" if block.group(0) in rewritten else "json"
            try:
                names = self._call_names(reply_text, block.group(1).strip(), complete=max_calls is not None)
            except ValueError:                         # a value that the JSON decoder refuses
                raise reject("", "undecodable_call", start, stop, wrapper) from None
            for name, kind in names:
                if name.lower() not in known:
                    declared = {tool_spec_name(tool).lower() for tool in self.declared_tools if tool_spec_name(tool)}
                    raise reject(name, "disallowed_by_tool_choice" if name.lower() in declared else "unavailable_tool",
                                 start, stop, "dsml" if wrapper == "dsml" else kind)
        return reply_text.parse_tool_calls(text, self.tools, max_calls=max_calls)

    def _call_names(self, reply_text: Any, block: str, complete: bool = False) -> list[tuple[str, str]]:
        """The (tool name, wrapper) that the engine's parser may read from one call block (JSON, Qwen XML or GLM).
        ``complete`` is the parser's single-call mode (``max_calls`` set): it refuses Qwen and GLM bodies with extra text."""

        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            pass                                       # not JSON: the parser tries the Qwen and GLM forms
        else:
            if not isinstance(payload, dict):
                return []                              # (valid JSON that is not an object is no call for the parser)
            function = payload.get("function")
            for source in (function, payload):         # (an empty nested object falls back to the flat form)
                name = source.get("name") if isinstance(source, dict) else None
                if name and str(name).strip():
                    # The parser matches the name as written. Offered names are trimmed, so a name with outer white
                    # space can never match one: it is no call.
                    return [(str(name), "json")] if str(name) == str(name).strip() else []
            return []
        xml = reply_text._TOOL_FUNCTION_BLOCK_RE.match(block)
        if xml:
            if complete and reply_text._TOOL_PARAMETER_BLOCK_RE.sub("", xml.group(2)).strip():
                return []
            return [(xml.group(1).strip(), "qwen_xml")]
        glm = reply_text.parse_glm_tool_call_block(block, self.tools, complete=complete)
        return [(str(glm[0]).strip(), "glm")] if glm is not None and str(glm[0]).strip() else []
