#!/usr/bin/env python3
"""Install only the opt-in tool-output guard into pristine pinned TensorFold."""
from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path
import subprocess

REVISION = "07aa02585f64cfb04c60a96de6554c0d4708ed46"
APP = Path("src/tensorfold/families/deepseek_v41/cuda/app.py")
SERVER = Path("src/tensorfold/cuda/server.py")
HTTP = Path("src/tensorfold/cuda/http.py")
FILES = {
    APP: "426b9463b5a254d96895a8c38f2f7cb5cc6c949bc8b607326e09af3a28d1de33",
    SERVER: "933dbd468f057a1a9f592171ee6f7bdc823231237b1fcc4962ac671cf727e0a9",
    HTTP: "bf8cb5e428c372358ff1e8676f3f92a6f354f1dc35999adfd8487cf6ff1aa1cb",
}
HERE = Path(__file__).resolve().parent


def replace(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"expected one pristine anchor: {old!r}")
    return text.replace(old, new)


def transform(relative: str | Path, text: str) -> str:
    """Transform pristine text, without reading or changing the source tree."""
    relative = Path(relative)
    if relative == APP:
        # The guard is on in this image. TF_DS_STRICT_TOOL_OUTPUT=0 (STRICT_TOOL_OUTPUT=0 in .env, see start.sh) turns
        # it off. server.py reads the property for each request.
        return replace(text, "class DsApp(App):\n    reads_ignore_eos = True\n",
                       "class DsApp(App):\n    reads_ignore_eos = True\n\n"
                       "    @property\n    def strict_tool_output(self) -> bool:\n"
                       "        import os\n\n"
                       "        return os.environ.get(\"TF_DS_STRICT_TOOL_OUTPUT\", \"1\") != \"0\"\n\n")
    if relative == SERVER:
        text = replace(text, "from tensorfold.cuda.turns import Turns, Yield\n",
                       "from tensorfold.cuda.turns import Turns, Yield\nfrom tensorfold.cuda.tool_output import ToolOutputGuard\n")
        text = replace(text, "        policy = ToolCallPolicy(body)\n",
                       "        policy = ToolCallPolicy(body)\n"
                       "        tool_output_guard = (ToolOutputGuard(tools, request=body)\n"
                       "                             if chat and getattr(self, 'strict_tool_output', False) else None)\n")
        text = replace(text, "        calls_stream = ToolCallStreamer(tools) if tools and not policy.single else None\n",
                       "        calls_stream = (ToolCallStreamer(tools) if tools and not policy.single\n"
                       "                        and tool_output_guard is None else None)\n")
        text = replace(text, "            answer_raw[0] = answer\n",
                       "            answer_raw[0] = answer\n"
                       "            if tool_output_guard is not None:\n"
                       "                return reasoning, tool_output_guard.stream(answer)\n")
        return replace(text,
                       "        content, calls = parse_tool_calls(raw_answer, tools, max_calls=policy.max_calls) if tools else (answer, None)\n"
                       "        content = policy.content(content) if tools else content\n",
                       "        if tool_output_guard is not None:\n"
                       "            content, calls = tool_output_guard.parse(raw_answer, max_calls=policy.max_calls)\n"
                       "        else:\n"
                       "            content, calls = parse_tool_calls(raw_answer, tools, max_calls=policy.max_calls) if tools else (answer, None)\n"
                       "            content = policy.content(content) if tools else content\n")
    if relative == HTTP:
        text = replace(text, "from tensorfold.cuda import health\n",
                       "from tensorfold.cuda import health\nfrom tensorfold.cuda.tool_output import ModelOutputError\n")
        text = replace(text,
                       "                except RequestError as exc:\n                    return self._stream_error(error_body(exc, field))\n",
                       "                except RequestError as exc:\n                    return self._stream_error(error_body(exc, field))\n"
                       "                except ModelOutputError as exc:\n"
                       "                    print(exc.log_line(), flush=True)\n"
                       "                    return self._stream_error(exc.error)\n")
        return replace(text,
                       "                result = app.run(body, chat, lambda delta: True, prepared=prepared, cancelled=cancelled)\n"
                       "            except RequestCancelled:\n",
                       "                result = app.run(body, chat, lambda delta: True, prepared=prepared, cancelled=cancelled)\n"
                       "            except ModelOutputError as exc:\n"
                       "                print(exc.log_line(), flush=True)\n"
                       "                return self._json(422, {'error': exc.error})\n"
                       "            except RequestCancelled:\n")
    return text


def overlays() -> dict[Path, str]:
    return {Path("src/tensorfold/cuda/tool_output.py"): (HERE / "overlay/tool_output.py").read_text()}


def install(root: Path) -> None:
    root = Path(root)
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError("TensorFold revision does not match the pinned revision")
    generated = {}
    for relative, expected in FILES.items():
        raw = (root / relative).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"source hash mismatch: {relative}")
        generated[relative] = transform(relative, raw.decode("utf-8"))
    for relative, text in overlays().items():
        if (root / relative).exists() or (root / relative).is_symlink():
            raise ValueError(f"overlay already exists: {relative}")
        generated[relative] = text
    # Validate every result before the first write. This is also the composition API.
    for relative, text in generated.items():
        ast.parse(text, filename=str(relative))
    for relative, text in generated.items():
        (root / relative).write_text(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    install(parser.parse_args().source)
