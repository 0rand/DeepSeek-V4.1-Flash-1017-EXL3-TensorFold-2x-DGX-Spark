"""Atomic, fail-closed validation for the strict native tool-output lane.

Do not use the permissive legacy parser here: it repairs/truncates invalid calls.
Literal markdown examples are never executable. Prose may stream before the
first possible protocol marker. The remainder and every executable call wait
for whole-answer validation, never a parse of a provisional envelope prefix.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from decimal import Decimal, localcontext
from typing import Any


class ModelOutputError(Exception):
    """A public terminal error, without model-generated argument contents."""
    code = "unavailable_tool_call"  # Existing typed wire contract, including malformed calls.

    def __init__(self, tool_name: str | None = None, *, diagnostic: dict | None = None):
        self.rejection_id = uuid.uuid4().hex[:16]
        reason = (diagnostic or {}).get("reason", "unavailable_tool")
        message = ("The model's executable tool output was rejected. "
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


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _constant(value):
    raise ValueError("nonfinite JSON constant")


_DECODER = json.JSONDecoder(object_pairs_hook=_pairs, parse_constant=_constant)
_MISSING = object()


def _json(text):
    value, end = _DECODER.raw_decode(text.lstrip())
    if text.lstrip()[end:].strip():
        raise ValueError("trailing JSON data")
    # Also catches numeric overflow (1e999) inside nested values.
    json.dumps(value, allow_nan=False)
    return value


def _equal(a, b, budget=None):
    if budget is not None:
        budget.step()
    # JSON numbers compare numerically, booleans must not alias 0/1.
    if type(a) in (int, float) and type(b) in (int, float):
        return a == b
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_equal(v, b[k], budget) for k, v in a.items())
    if isinstance(a, list):
        return len(a) == len(b) and all(_equal(x, y, budget) for x, y in zip(a, b))
    return a == b


class _Mismatch(ValueError):
    """Only an ordinary instance/assertion mismatch may be inverted by a schema."""


class _EvaluationError(ValueError):
    """Invalid/unsupported schema or exhausted evaluation; never a mismatch."""


class _Budget:
    """Per-call shared structural/evaluation and regex work allowances."""
    def __init__(self):
        self.steps = 10000
        self.regex_work = 2000000
        self.patterns = {}

    def step(self, count=1):
        self.steps -= count
        if self.steps < 0:
            raise _EvaluationError('schema evaluation budget exhausted')


def _safe_pattern(pattern, budget):
    """Prove a small regex grammar safe BEFORE giving it to Python re.

    Sequences of literals, dot, character classes and simple escapes only.
    Fixed atom repeats <=128, total fixed width <=256; at most ONE variable
    atom repeat, and then both ^ and $ are mandatory. No groups, alternation,
    lookaround, backreferences, flags, lazy/possessive or compound repeats.
    This is deliberately not a general JSON Schema regex implementation.
    """
    if not isinstance(pattern, str) or len(pattern) > 256:
        raise _EvaluationError('regex pattern limit/type')
    if pattern in budget.patterns:
        return budget.patterns[pattern]
    i, fixed, variable = 0, 0, 0
    start = pattern.startswith('^')
    end = False
    if start:
        i = 1

    def escape(pos):
        if pos + 1 >= len(pattern):
            raise _EvaluationError('incomplete regex escape')
        ch = pattern[pos + 1]
        if ch not in 'dDsSwWtnrfv' and (ch.isalnum() or ch == '_'):
            raise _EvaluationError('unsupported regex escape')
        return pos + 2

    while i < len(pattern):
        ch = pattern[i]
        if ch == '$' and i == len(pattern) - 1:
            end = True
            i += 1
            break
        if ch == '\\':
            i = escape(i)
        elif ch == '[':
            i += 1
            if i < len(pattern) and pattern[i] == '^':
                i += 1
            members = 0
            while i < len(pattern) and pattern[i] != ']':
                if pattern[i] == '[':
                    raise _EvaluationError('nested regex class')
                if pattern[i] == '\\':
                    i = escape(i)
                else:
                    i += 1
                members += 1
            if i >= len(pattern) or not members:
                raise _EvaluationError('invalid regex class')
            i += 1
        elif ch in '()|^$*+?{}]':
            raise _EvaluationError('unsupported regex construct')
        else:
            i += 1  # Literal (including Unicode) or single dot atom.
        low = high = 1
        if i < len(pattern) and pattern[i] in '*+?':
            token = pattern[i]
            low, high = {'*':(0,None), '+':(1,None), '?':(0,1)}[token]
            i += 1
        elif i < len(pattern) and pattern[i] == '{':
            close = pattern.find('}', i + 1)
            if close < 0:
                raise _EvaluationError('invalid regex repeat')
            parts = pattern[i + 1:close].split(',')
            if len(parts) not in (1,2) or not parts[0] or any(c not in '0123456789' for c in parts[0]):
                raise _EvaluationError('unsupported regex repeat')
            low = int(parts[0])
            high = low
            if len(parts) == 2:
                if any(c not in '0123456789' for c in parts[1]):
                    raise _EvaluationError('unsupported regex repeat')
                high = int(parts[1]) if parts[1] else None
            if low > 128 or (high is not None and (high < low or high > 128)):
                raise _EvaluationError('regex repeat limit')
            i = close + 1
        if low != high:
            variable += 1
        else:
            fixed += high
        if variable > 1 or fixed > 256:
            raise _EvaluationError('regex complexity limit')
    if variable and not (start and end):
        raise _EvaluationError('variable regex repeat requires both anchors')
    try:
        compiled = re.compile(pattern)
    except re.error:
        raise _EvaluationError('invalid regex') from None
    result = (compiled, fixed + len(pattern) + 1)
    budget.patterns[pattern] = result
    return result


def _pattern_search(pattern, value, budget):
    compiled, width = _safe_pattern(pattern, budget)
    if len(value) > 8192:
        raise _EvaluationError('regex input limit')
    # Conservative bound on start positions/backtracking times fixed sequence
    # width, shared across ALL properties, array elements and union branches.
    budget.regex_work -= (len(value) + 1) * width
    if budget.regex_work < 0:
        raise _EvaluationError('regex work budget exhausted')
    return compiled.search(value) is not None


def _schema_contract(schema, root=None, depth=0, budget=None, refs=None):
    """Check assertion shapes even in unused properties/union branches."""
    budget = _Budget() if budget is None else budget
    refs = set() if refs is None else refs
    budget.step()
    if depth > 64:
        raise _EvaluationError('schema nesting limit')
    if type(schema) is bool:
        return
    if not isinstance(schema,dict):
        raise _EvaluationError('invalid schema')
    root = schema if root is None else root
    supported = {'$ref','type','enum','const','anyOf','oneOf','allOf','not','if','then','else','properties','required','additionalProperties','patternProperties','minProperties','maxProperties','dependentRequired','items','prefixItems','minItems','maxItems','uniqueItems','contains','minContains','maxContains','minLength','maxLength','pattern','minimum','maximum','exclusiveMinimum','exclusiveMaximum','multipleOf'}
    annotations = {'title','description','default','examples','$schema','$id','$comment','$defs','definitions','deprecated','readOnly','writeOnly','format'}
    if set(schema) - supported - annotations:
        raise _EvaluationError('unsupported schema assertion')
    for key in ('minimum','maximum','exclusiveMinimum','exclusiveMaximum','multipleOf'):
        if key in schema:
            n=schema[key]
            if type(n) not in (int,float) or not math.isfinite(n) or (key=='multipleOf' and n<=0):
                raise _EvaluationError('invalid numeric schema limit')
    for key in ('minProperties','maxProperties','minItems','maxItems','minContains','maxContains','minLength','maxLength'):
        if key in schema and (type(schema[key]) is not int or schema[key]<0):
            raise _EvaluationError('invalid integer schema limit')
    if 'uniqueItems' in schema and type(schema['uniqueItems']) is not bool:
        raise _EvaluationError('invalid uniqueItems')
    if 'type' in schema:
        types=schema['type'] if isinstance(schema['type'],list) else [schema['type']]
        if not types or any(not isinstance(t,str) or t not in {'object','array','string','number','integer','boolean','null'} for t in types) or len(types)!=len(set(types)):
            raise _EvaluationError('invalid schema type')
    for key in ('required',):
        if key in schema and (not isinstance(schema[key],list) or any(not isinstance(x,str) for x in schema[key]) or len(schema[key])!=len(set(schema[key]))):
            raise _EvaluationError('invalid required schema')
    if 'enum' in schema and (not isinstance(schema['enum'],list) or not schema['enum']):
        raise _EvaluationError('invalid schema enum')
    if 'pattern' in schema:
        if not isinstance(schema['pattern'],str):
            raise _EvaluationError('invalid schema pattern')
        _safe_pattern(schema['pattern'], budget)
    for key in ('properties','patternProperties','$defs','definitions'):
        if key in schema:
            if not isinstance(schema[key],dict) or any(not isinstance(x,str) for x in schema[key]):
                raise _EvaluationError('invalid schema properties')
            for name,child in schema[key].items():
                if key=='patternProperties': _safe_pattern(name, budget)
                _schema_contract(child,root,depth+1,budget,refs)
    for key in ('items','additionalProperties','contains','not','if','then','else'):
        if key in schema: _schema_contract(schema[key],root,depth+1,budget,refs)
    for key in ('allOf','anyOf','oneOf','prefixItems'):
        if key in schema:
            if not isinstance(schema[key],list) or not schema[key]:
                raise _EvaluationError('invalid schema branches')
            for child in schema[key]: _schema_contract(child,root,depth+1,budget,refs)
    if 'dependentRequired' in schema:
        deps=schema['dependentRequired']
        if not isinstance(deps,dict) or any(not isinstance(k,str) or not isinstance(v,list) or any(not isinstance(x,str) for x in v) for k,v in deps.items()):
            raise _EvaluationError('invalid dependentRequired')
    if '$ref' in schema:
        ref=schema['$ref']
        if not isinstance(ref,str) or not ref.startswith('#/'):
            raise _EvaluationError('nonlocal schema reference')
        target=root
        try:
            for part in ref[2:].split('/'):
                target=target[part.replace('~1','/').replace('~0','~')]
        except (KeyError, TypeError):
            raise _EvaluationError('unresolved schema reference') from None
        if not isinstance(target,dict) and type(target) is not bool:
            raise _EvaluationError('invalid reference target')
        # A local pointer may target an annotation, not just $defs/properties.
        # Validate those targets even in unselected branches, without unfolding
        # recursive reference cycles during this structural preflight.
        if id(target) not in refs:
            refs.add(id(target))
            _schema_contract(target,root,depth+1,budget,refs)


def _schema(value, schema, root=None, depth=0, budget=None):
    """Dependency-free JSON Schema subset; unsupported assertions fail CLOSED.

    Standard annotations are ignored. Local refs, unions, nested types, object/
    array/string/numeric constraints are supported. No coercion or JSON repair.
    """
    budget = _Budget() if budget is None else budget
    budget.step()
    if depth == 0:
        if len(json.dumps(schema,allow_nan=False)) > 262144 or len(json.dumps(value,allow_nan=False)) > 262144:
            raise _EvaluationError("schema/instance size limit")
        _schema_contract(schema, budget=budget)
    if depth > 64:
        raise _EvaluationError("schema nesting limit")
    if schema is True:
        return
    if schema is False:
        raise _Mismatch("false schema")
    if not isinstance(schema, dict):
        raise _EvaluationError("invalid schema")
    root = schema if root is None else root
    annotations = {'title','description','default','examples','$schema','$id','$comment','$defs','definitions','deprecated','readOnly','writeOnly','format'}
    supported = {'$ref','type','enum','const','anyOf','oneOf','allOf','not','if','then','else','properties','required','additionalProperties','patternProperties','minProperties','maxProperties','dependentRequired','items','prefixItems','minItems','maxItems','uniqueItems','contains','minContains','maxContains','minLength','maxLength','pattern','minimum','maximum','exclusiveMinimum','exclusiveMaximum','multipleOf'}
    if set(schema) - annotations - supported:
        raise _EvaluationError("unsupported schema assertion")
    def check(v, s):
        return _schema(v, s, root, depth + 1, budget)
    def matches(v, s):
        try:
            check(v, s)
            return True
        except _Mismatch:
            return False
    if '$ref' in schema:
        ref = schema['$ref']
        if not isinstance(ref, str) or not ref.startswith('#/'):
            raise _EvaluationError("nonlocal schema reference")
        target = root
        try:
            for part in ref[2:].split('/'):
                target = target[part.replace('~1','/').replace('~0','~')]
        except (KeyError, TypeError):
            raise _EvaluationError("unresolved schema reference") from None
        check(value, target)
    kind = schema.get('type')
    kinds = {'object': lambda v:isinstance(v,dict), 'array':lambda v:isinstance(v,list),
             'string':lambda v:isinstance(v,str), 'boolean':lambda v:type(v) is bool,
             'integer':lambda v:type(v) is int or (type(v) is float and math.isfinite(v) and v.is_integer()),
             'number':lambda v:type(v) in (int,float) and math.isfinite(v), 'null':lambda v:v is None}
    if kind is not None:
        types = kind if isinstance(kind,list) else [kind]
        if not types or any(not isinstance(k,str) or k not in kinds for k in types) or not any(kinds[k](value) for k in types):
            raise _Mismatch("schema type mismatch")
    if 'enum' in schema and not any(_equal(value, x, budget) for x in schema['enum']):
        raise _Mismatch("schema enum mismatch")
    if 'const' in schema and not _equal(value, schema['const'], budget):
        raise _Mismatch("schema const mismatch")
    for key in ('allOf','anyOf','oneOf'):
        if key in schema:
            branches = schema[key]
            if not isinstance(branches,list) or not branches:
                raise _EvaluationError("invalid schema union")
            n = sum(matches(value, x) for x in branches)
            if (key == 'allOf' and n != len(branches)) or (key == 'anyOf' and n == 0) or (key == 'oneOf' and n != 1):
                raise _Mismatch("schema union mismatch")
    if 'not' in schema and matches(value,schema['not']):
        raise _Mismatch("schema not mismatch")
    if 'if' in schema:
        branch = 'then' if matches(value,schema['if']) else 'else'
        if branch in schema:
            check(value,schema[branch])
    if isinstance(value,dict):
        props = schema.get('properties',{})
        patterns = schema.get('patternProperties',{})
        required = schema.get('required',[])
        if not isinstance(props,dict) or not isinstance(patterns,dict) or not isinstance(required,list) or any(not isinstance(k,str) for k in required):
            raise _EvaluationError("invalid object schema")
        if any(k not in value for k in required):
            raise _Mismatch("missing required argument")
        for k,v in value.items():
            covered = False
            if k in props:
                check(v,props[k]); covered = True
            for pattern, spec in patterns.items():
                if _pattern_search(pattern,k,budget):
                    check(v,spec); covered = True
            if not covered:
                check(v,schema.get('additionalProperties',True))
        for k, dependencies in schema.get('dependentRequired',{}).items():
            if k in value and any(dep not in value for dep in dependencies):
                raise _Mismatch("missing dependent argument")
        if len(value) < schema.get('minProperties',0) or len(value) > schema.get('maxProperties',math.inf):
            raise _Mismatch("object size mismatch")
    if isinstance(value,list):
        prefix = schema.get('prefixItems',[])
        for i,v in enumerate(value):
            check(v,prefix[i] if i < len(prefix) else schema.get('items',True))
        if len(value) < schema.get('minItems',0) or len(value) > schema.get('maxItems',math.inf):
            raise _Mismatch("array size mismatch")
        if schema.get('uniqueItems') and any(_equal(x,y,budget) for i,x in enumerate(value) for y in value[:i]):
            raise _Mismatch("array uniqueness mismatch")
        if 'contains' in schema:
            count = sum(matches(v,schema['contains']) for v in value)
            if count < schema.get('minContains',1) or count > schema.get('maxContains',math.inf):
                raise _Mismatch("array contains mismatch")
    if isinstance(value,str):
        if len(value)<schema.get('minLength',0) or len(value)>schema.get('maxLength',math.inf) or ('pattern' in schema and not _pattern_search(schema['pattern'],value,budget)):
            raise _Mismatch("string constraint mismatch")
    if type(value) in (int,float):
        if not math.isfinite(value):
            raise _EvaluationError("nonfinite argument")
        for key, op in [('minimum',lambda n:value>=n),('maximum',lambda n:value<=n),('exclusiveMinimum',lambda n:value>n),('exclusiveMaximum',lambda n:value<n)]:
            if key in schema and (type(schema[key]) not in (int,float) or not math.isfinite(schema[key]) or not op(schema[key])):
                raise _Mismatch("numeric constraint mismatch")
        if 'multipleOf' in schema:
            n = schema['multipleOf']
            if type(n) not in (int,float) or not math.isfinite(n) or n<=0:
                raise _Mismatch("multipleOf mismatch")
            # Decimal JSON numbers must not pass by an epsilon around a bound.
            with localcontext() as context:
                context.prec=1024
                if Decimal(str(value)) % Decimal(str(n)) != 0:
                    raise _Mismatch("multipleOf mismatch")


class ToolOutputGuard:
    """Validate the entire answer before releasing any executable delta."""
    _marker = re.compile(r'<(?:/?(?:[\w.-]+:)?tool_call\b|/?[｜|]DSML[｜|]|\|tool_call\b)', re.I)
    # A conservative superset of incomplete _marker spellings. Namespaces can
    # be arbitrarily long, so a fixed longest-opener holdback is NOT sufficient.
    _pending_marker = re.compile(r'<(?:/?[\w.:-]*|/?[｜|][\w｜|.-]*)\Z', re.I)
    _pending_scan_limit = 256
    _quote_line = re.compile(r'^[ \t]*>[^\n]*(?:\n|$)', re.M)
    _prose_token = re.compile(_quote_line.pattern + r'|' + _marker.pattern + r'|`+|~{3,}', re.I|re.M)
    _inline_boundary = re.compile(r'^[ \t]*(?:\n|$)|' + _quote_line.pattern + r'|`+|~{3,}', re.M)
    _fence_close = re.compile(r'(?m)^[ \t]*(`{3,}|~{3,})([^\n]*)(?:\n|$)')
    _param = re.compile(r'<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>',re.I|re.S)
    _invoke = re.compile(r'<｜DSML｜ invoke name="([^"\s]+)">\s*(.*?)\s*</｜DSML｜ invoke>',re.S)
    _dsparam = re.compile(r'<｜DSML｜ parameter name="([^"\s]+)"(?: string="(true|false)")?>(.*?)</｜DSML｜ parameter>',re.S)
    _invoke_v4 = re.compile(r'<｜DSML｜invoke name="([^"\s]+)">\s*(.*?)\s*</｜DSML｜invoke>',re.S)
    _dsparam_v4 = re.compile(r'<｜DSML｜parameter name="([^"\s]+)"(?: string="(true|false)")?>(.*?)</｜DSML｜parameter>',re.S)

    def __init__(self, tools: list[dict[str, Any]] | None, *, request: dict | None = None):
        self.tools = tools if tools is not None else []
        self.request = request if request is not None else {'tools':self.tools}
        self._stream_seen = ''
        self._stream_scanned = 0
        self._stream_cut = None
        self._stream_published = ''

    def stream(self, text: str) -> str:
        """Project cumulative inert prose, then freeze until authoritative parse().

        Deliberately scan raw markers, even inside literal examples. Inline and
        quoted ownership can change with later output, including across lines.
        No partial call is decoded, validated, assigned an ID or released here.
        """
        if not text.startswith(self._stream_seen):
            self._reject(text, 'stream_prefix_changed')
        self._stream_seen = text
        if self._stream_cut is None:
            marker = self._marker.search(text, self._stream_scanned)
            if marker is not None:
                self._stream_cut = marker.start()
            else:
                start = text.rfind('<', self._stream_scanned)
                pending = start >= 0 and self._pending_marker.fullmatch(text, start) is not None
                self._stream_scanned = start if pending else len(text)
                if pending and len(text) - start > self._pending_scan_limit:
                    # Freeze rather than repeatedly rescan an unbounded namespace.
                    # This is a latency fallback, not a rejection or output limit.
                    self._stream_cut = start
        end = self._stream_cut if self._stream_cut is not None else self._stream_scanned
        shown = text[:end]
        if not shown.startswith(self._stream_published):
            self._reject(text, 'stream_prefix_changed')
        self._stream_published = shown
        return shown

    def check_stream_prefix(self, text: str, content: str) -> None:
        """Refuse a decoder rewrite or a final-parser/projection disagreement."""
        if not text.startswith(self._stream_seen) or not content.startswith(self._stream_published):
            self._reject(text, 'stream_prefix_changed')

    def _reject(self, text, reason):
        raise ModelOutputError(diagnostic={'reason':reason,'answer_sha256':hashlib.sha256(text.encode()).hexdigest()})

    def _call(self, name, args, ident=_MISSING):
        if not isinstance(name,str) or not name or not isinstance(args,dict):
            raise ValueError('malformed_call')
        # Exact, case-sensitive protocol names; never silently normalize IDs or names.
        offered = {}
        for tool in self.tools:
            fn = tool.get('function',tool)
            declared = fn.get('name')
            if not isinstance(declared,str) or not declared or declared in offered:
                raise ValueError('invalid_offered_tool')
            offered[declared] = tool
        if name not in offered:
            raise ValueError('unavailable_tool')
        choice = self.request.get('tool_choice','auto')
        if choice == 'none' or (isinstance(choice,dict) and choice.get('function',{}).get('name') != name):
            raise ValueError('disallowed_by_tool_choice')
        tool = offered[name]
        fn = tool.get('function',tool)
        schema = fn.get('parameters',fn.get('input_schema',{}))
        json.dumps(args,allow_nan=False)
        _schema(args,schema)
        if ident is not _MISSING and (not isinstance(ident,str) or not ident):
            raise ValueError('invalid_call_id')
        return {'id':ident if ident is not _MISSING else 'call_'+uuid.uuid4().hex[:24], 'type':'function',
                'function':{'name':name,'arguments':json.dumps(args,ensure_ascii=False,separators=(',',':'))}}

    def _parameters(self, body, pattern, name, dsml=False):
        fn = next((t.get('function',t) for t in self.tools if t.get('function',t).get('name')==name),{})
        schema = fn.get('parameters',fn.get('input_schema',{}))
        props = schema.get('properties',{}) if isinstance(schema,dict) else {}
        args = {}; cursor = 0
        for m in pattern.finditer(body):
            if body[cursor:m.start()].strip() or m.group(1) in args:
                raise ValueError('malformed_parameters')
            key = m.group(1)
            raw = m.group(3) if dsml else m.group(2)
            # DSML string=false is strict JSON. XML strings are literal; typed
            # values must be complete JSON, not the legacy close/repair decoder.
            typed = props.get(key,{})
            if (dsml and m.group(2)=='false') or (not dsml and isinstance(typed,dict) and typed.get('type') not in (None,'string')):
                args[key] = _json(raw)
            else:
                args[key] = raw
            cursor = m.end()
        if body[cursor:].strip():
            raise ValueError('malformed_parameters')
        return args

    def _block(self, body):
        body = body.strip()
        if body.startswith('{'):
            payload = _json(body)
            nested = 'function' in payload
            allowed = {'id','type','function'} if nested else {'id','type','name','arguments','parameters'}
            if set(payload) - allowed:
                raise ValueError('unexpected_call_fields')
            fn = payload.get('function',payload)
            if (not isinstance(fn,dict) or payload.get('type','function') != 'function'
                    or (nested and set(fn)-{'name','arguments','parameters'})
                    or ('arguments' in fn and 'parameters' in fn)):
                raise ValueError('malformed_call')
            args = fn.get('arguments',fn.get('parameters',{}))
            if isinstance(args,str):
                args = _json(args)
            return self._call(fn.get('name'),args,payload.get('id',_MISSING))
        xml = re.fullmatch(r'<function=([^>\s]+)>\s*(.*?)\s*</function>',body,re.I|re.S)
        if xml:
            name = xml.group(1)
            return self._call(name,self._parameters(xml.group(2),self._param,name))
        glm = re.fullmatch(r'([\w.:-]+)(.*)',body,re.S)
        if glm:
            # Convert GLM parameters without forgiving unparsed residue/duplicates.
            pattern = re.compile(r'<arg_key>([^<]+)</arg_key>\s*<arg_value>(.*?)</arg_value>',re.S)
            name = glm.group(1)
            return self._call(name,self._parameters(glm.group(2),pattern,name))
        raise ValueError('malformed_call')

    def parse(self, text: str, max_calls: int | None = None) -> tuple[str, list[dict[str, Any]] | None]:
        content=[]; calls=[]; cursor=0
        try:
            choice = self.request.get('tool_choice','auto')
            if choice is None:
                choice = 'auto'
            if isinstance(choice,dict):
                if (choice.get('type') != 'function' or not isinstance(choice.get('function'),dict)
                        or not isinstance(choice['function'].get('name'),str) or not choice['function']['name']):
                    raise ValueError('invalid_tool_choice')
            elif not isinstance(choice,str) or choice not in ('auto','none','required'):
                raise ValueError('invalid_tool_choice')
            parallel = self.request.get('parallel_tool_calls',True)
            if parallel is None:
                parallel = True  # ToolCallPolicy treats JSON null as the default.
            if type(parallel) is not bool:
                raise ValueError('invalid_parallel_tool_calls')
            if max_calls is not None and (type(max_calls) is not int or max_calls < 0):
                raise ValueError('invalid_max_calls')
            while True:
                # Resume prose scanning only AFTER the last envelope. Argument
                # delimiters and nested markers are protocol data, never Markdown.
                m = self._next_marker(text,cursor)
                if not m:
                    content.append(text[cursor:]); break
                start = m.start()
                literal_end = None
                line_start = text.rfind('\n',0,start)+1
                line_end = text.find('\n',start)
                if line_end < 0:
                    line_end = len(text)
                prefix = text[line_start:start].rstrip()
                if (prefix and self._opening_quote(start,text,cursor)
                        and text[start:line_end].rstrip().endswith(prefix[-1])
                        and self._marker.search(text,m.end(),line_end) is None):
                    # A line-quoted example can itself be truncated/malformed.
                    # Never extend its literal exemption past the quoted line.
                    literal_end = line_end
                if literal_end is not None or re.match(r'[ \t]*>',text[line_start:start]):
                    # A malformed example must not consume executable output AFTER
                    # its fence/line. Skip only the actual literal span.
                    stop = literal_end if literal_end is not None else text.find('\n',start)
                    if stop < 0:
                        stop = len(text)
                    content.append(text[cursor:stop]); cursor=stop; continue
                opening = re.match(r'<tool_call>\s*',text[start:],re.I)
                if opening:
                    body_start = start+opening.end()
                    if text[body_start:].startswith('{'):
                        # Locate JSON's actual end, not a closing marker inside a string.
                        try:
                            _, end = _DECODER.raw_decode(text,body_start)
                            close = re.match(r'\s*</tool_call>',text[end:],re.I)
                            stop = end+close.end() if close else len(text)
                        except ValueError:
                            close = re.search(r'</tool_call>',text[body_start:],re.I)
                            stop = body_start+close.end() if close else len(text)
                    else:
                        close = re.search(r'</tool_call>',text[body_start:],re.I)
                        stop = body_start+close.end() if close else len(text)
                else:
                    ds_open = '<｜DSML｜tool_calls>' if text.startswith('<｜DSML｜tool_calls>',start) else '<｜DSML｜ calls>'
                    ds_close = '</｜DSML｜tool_calls>' if ds_open == '<｜DSML｜tool_calls>' else '</｜DSML｜ calls>'
                    close = text.find(ds_close,start)
                    stop = close+len(ds_close) if close>=0 else len(text)
                quoted_end = self._quoted_end(start,stop,text,cursor)
                if quoted_end is not None:
                    # A literal malformed JSON example need not even be decodable.
                    # Consume its closing quote too: it cannot open a new literal.
                    stop=quoted_end
                    content.append(text[cursor:stop]); cursor=stop; continue
                content.append(text[cursor:start])
                if opening:
                    if not re.search(r'</tool_call>\Z',text[start:stop],re.I):
                        raise ValueError('truncated_call')
                    calls.append(self._block(re.sub(r'</tool_call>\Z','',text[body_start:stop],flags=re.I)))
                else:
                    block = text[start:stop]
                    if not block.startswith(ds_open) or not block.endswith(ds_close):
                        raise ValueError('malformed_dsml')
                    body = block[len(ds_open):-len(ds_close)]
                    invoke = self._invoke_v4 if ds_open == '<｜DSML｜tool_calls>' else self._invoke
                    parameter = self._dsparam_v4 if ds_open == '<｜DSML｜tool_calls>' else self._dsparam
                    pos=0; found=False
                    for inv in invoke.finditer(body):
                        if body[pos:inv.start()].strip():
                            raise ValueError('malformed_dsml')
                        name=inv.group(1)
                        calls.append(self._call(name,self._parameters(inv.group(2),parameter,name,True)))
                        pos=inv.end(); found=True
                    if not found or body[pos:].strip():
                        raise ValueError('malformed_dsml')
                cursor=stop
            choice=self.request.get('tool_choice','auto')
            if not calls and (choice=='required' or isinstance(choice,dict)):
                raise ValueError('required_tool_missing')
            if max_calls is not None and len(calls)>max_calls:
                raise ValueError('too_many_tool_calls')
            if self.request.get('parallel_tool_calls') is False and len(calls)>1:
                raise ValueError('parallel_tool_calls_disabled')
            ids=[c['id'] for c in calls]
            if len(ids)!=len(set(ids)):
                raise ValueError('duplicate_call_id')
        except (ValueError,TypeError,KeyError,AttributeError,ArithmeticError,RecursionError,re.error) as exc:
            # Do not reflect raw JSON/schema errors containing model data.
            self._reject(text,'invalid_executable_tool_output')
        return ''.join(content), calls or None

    @classmethod
    def _next_marker(cls, text, cursor):
        """Scan prose literals up to, but never through, an executable opener.

        A literal opened in prose owns its contents (including malformed examples).
        Otherwise the caller must consume/validate the whole protocol envelope
        before invoking this scanner again. No state originates in call arguments.
        """
        while (token := cls._prose_token.search(text,cursor)) is not None:
            raw=token.group()
            if raw.lstrip(' \t').startswith('>'):
                # The complete quoted line owns all its delimiters. None can
                # open inline/fenced state around a later unquoted attempt.
                cursor=token.end()
                continue
            if raw.startswith('<'):
                return token
            cursor=token.end()
            line_start=text.rfind('\n',0,token.start())+1
            if len(raw)>=3 and not text[line_start:token.start()].strip():
                # Unclosed fences remain literal to end-of-answer. Closing runs
                # must use the same character, be at least as long, and stand alone.
                # Search the original string: slicing at cursor invents a line
                # start inside the opening fence's info string (e.g. ~~~ ~~~).
                for close in cls._fence_close.finditer(text,cursor):
                    run=close.group(1)
                    if run[0]==raw[0] and len(run)>=len(raw) and not close.group(2).strip():
                        cursor=close.end()
                        break
                else:
                    return None
            elif raw[0]=='`':
                # Compare runs directly, not a model-sized dynamic regex repeat.
                # A block fence takes precedence over inline code. Do not let
                # a stray prose backtick close inside a later fence and expose
                # its literal examples (including an unclosed fence).
                for close in cls._inline_boundary.finditer(text,cursor):
                    run=close.group()
                    if not run.strip() or run.lstrip(' \t').startswith('>'):
                        # Quoted containers and paragraph breaks terminate this
                        # prose lookup; their delimiters cannot close it either.
                        break
                    line_start=text.rfind('\n',0,close.start())+1
                    if len(run)>=3 and not text[line_start:close.start()].strip():
                        break
                    if run[0]=='`' and len(run)==len(raw):
                        cursor=close.end()
                        break
        return None

    @staticmethod
    def _opening_quote(start,text,context_start=0):
        prefix=text[max(context_start,text.rfind('\n',0,start)+1):start].rstrip()
        if not prefix or prefix[-1] not in "\"'":
            return False
        quote=prefix[-1]; count=0; escaped=False
        for i,ch in enumerate(prefix):
            if escaped:
                escaped=False; continue
            if ch=='\\':
                escaped=True; continue
            if ch==quote:
                # Prose contractions are not single-quote delimiters.
                if quote=="'" and i>0 and i+1<len(prefix) and prefix[i-1].isalnum() and prefix[i+1].isalnum():
                    continue
                count+=1
        return count % 2 == 1 and not escaped

    @staticmethod
    def _quoted_end(start,stop,text,context_start):
        before,after=text[context_start:start].rstrip(),text[stop:].lstrip()
        if ToolOutputGuard._opening_quote(start,text,context_start) and before and after and after[0]==before[-1]:
            return len(text)-len(after)+1
        return None
