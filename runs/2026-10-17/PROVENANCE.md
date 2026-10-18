# Provenance of the 2026-10-17 runs

This file records where each input of the 2026-10-17 runs came from.
All times are UTC. All facts were read from the head node on 2026-10-17.
The dates in file names and tags follow the build or run time on the head node.

## 1. Weights

| Part | Source | Revision |
| --- | --- | --- |
| EXL3 2.9 bpw weights (43 shards) | `LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw` | `afc7aa0aab98ee735c0f54e0a122019af1ac08f1` |
| Engram tables (2 of 48 shards) | `deepseek-ai/DeepSeek-V4.1-Flash-1017` | `1bae0e1088993b9b82de08f59de5be5a34e42170` |

### EXL3 weights

- The download used `hf download` with `--exclude "assets/*"`. The HF Xet path was off (`HF_HUB_DISABLE_XET=1`).
- The repo has 43 weight shards (`model-00001-of-00043.safetensors` to `model-00043-of-00043.safetensors`). The shards are 1.3 to 5.7 GB each.
- `model/` in the served view is about 196 GiB (210,818,714,729 bytes by `du -sbL`, including the two template files). `engram/` is about 189 GiB (203,080,551,181 bytes).

### Engram shards

| File | Size (bytes) | sha256 |
| --- | --- | --- |
| `model-00047-of-00048.safetensors` | 101,535,150,936 | `76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5` |
| `model-00048-of-00048.safetensors` | 101,537,926,640 | `9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd` |

- The view also holds `model.safetensors.index.json` (7,470,294 bytes) and `config.json` (3,311 bytes) of the base repo, at the same revision.
- The shards are about 101 GB each, so the checksums were NOT computed again for this record.
- The hashes above are the Hugging Face LFS object ids. The staging script compared them with `sha256sum -c` after the download.
  The head check ended with `verify OK` at 17:47:03Z. The worker check ended with `worker view OK` at 18:04:07Z.
- `hf download` refuses files larger than 50 GB without `hf_xet`. The staging script downloaded the Engram files with resumable `curl` from the `resolve` URL.

### Checksums of the whole view

`weights.sha256` (same folder as this file) holds 53 lines in `sha256sum -c` format, with paths relative to the view root.
It has the 47 LFS object ids (43 shards, the index and quantization config of the EXL3 repo, the 2 Engram shards)
and 6 hashes of small files, computed on the head node from the served view (`config.json`, `exl3_k_map.json`, `tokenizer.json`, `tokenizer_config.json`, and the two Engram metadata files).

```
cd <view root> && sha256sum -c weights.sha256
```

## 2. How the served model view was built

- Script: `scripts/_raw_stage-v41-1017.sh` (raw copy of `stage-v41-1017.sh` from the head node). Phases: `download`, `engram`, `verify`, `view`, `worker`.
- The view is a directory of **hard links**. It has no symlinks. Each file is a hard link to the Hugging Face cache blob (EXL3 files) or to the downloaded Engram file (link count 2).
- The script builds the view in `<view>.staging`, checks that it has 43 EXL3 shards, and then renames it.
- The worker gets a copy with `rsync`. The script compares the file lists and the sha256 sums on the worker before it renames the copy.
- Staging ran 17:11Z to 18:04Z. The view was built at 17:47Z.
- View path on both nodes: `~/.cache/huggingface/model-views/deepseek-v41-1017-exl3-2.9bpw-afc7aa0a`
- Mounts used by TensorFold: `<view>/model` to `/model` and `<view>/engram` to `/engram`, both read-only.

Contents (2 levels). Size in bytes, then path:

```
           3311  engram/config.json
   101535150936  engram/model-00047-of-00048.safetensors
   101537926640  engram/model-00048-of-00048.safetensors
        7470294  engram/model.safetensors.index.json
           3628  model/config.json
           1409  model/exl3_k_map.json
     1323827293  model/model-00001-of-00043.safetensors
     5322748742  model/model-00002-of-00043.safetensors
     5223069939  model/model-00003-of-00043.safetensors
     5230054628  model/model-00004-of-00043.safetensors
     5223069939  model/model-00005-of-00043.safetensors
     5223069939  model/model-00006-of-00043.safetensors
     5223069939  model/model-00007-of-00043.safetensors
     5223069939  model/model-00008-of-00043.safetensors
     5223069939  model/model-00009-of-00043.safetensors
     5230054628  model/model-00010-of-00043.safetensors
     5223069939  model/model-00011-of-00043.safetensors
     5223074621  model/model-00012-of-00043.safetensors
     5218650941  model/model-00013-of-00043.safetensors
     5218650941  model/model-00014-of-00043.safetensors
     5297522070  model/model-00015-of-00043.safetensors
     5225635649  model/model-00016-of-00043.safetensors
     5218650941  model/model-00017-of-00043.safetensors
     5218650941  model/model-00018-of-00043.safetensors
     5218650941  model/model-00019-of-00043.safetensors
     3519956829  model/model-00020-of-00043.safetensors
     3519956829  model/model-00021-of-00043.safetensors
     3525291432  model/model-00022-of-00043.safetensors
     3519956829  model/model-00023-of-00043.safetensors
     3519956829  model/model-00024-of-00043.safetensors
     5218650941  model/model-00025-of-00043.safetensors
     5222266717  model/model-00026-of-00043.safetensors
     5218650941  model/model-00027-of-00043.safetensors
     5218650941  model/model-00028-of-00043.safetensors
     5218650941  model/model-00029-of-00043.safetensors
     5222266717  model/model-00030-of-00043.safetensors
     5220125501  model/model-00031-of-00043.safetensors
     5223074621  model/model-00032-of-00043.safetensors
     5223074621  model/model-00033-of-00043.safetensors
     5226690397  model/model-00034-of-00043.safetensors
     5223074621  model/model-00035-of-00043.safetensors
     5223074621  model/model-00036-of-00043.safetensors
     5223074621  model/model-00037-of-00043.safetensors
     5226690397  model/model-00038-of-00043.safetensors
     5223074621  model/model-00039-of-00043.safetensors
     5223074621  model/model-00040-of-00043.safetensors
     5719789284  model/model-00041-of-00043.safetensors
     4854649316  model/model-00042-of-00043.safetensors
     3500776125  model/model-00043-of-00043.safetensors
       15504210  model/model.safetensors.index.json
       54653012  model/quantization_config.json
            801  model/tokenizer_config.json
        6367257  model/tokenizer.json
          11780  model/chat_template.jinja   (added after the view build; patched, see template-patch.diff)
          11410  model/chat_template.jinja.bak-pre-tools-nl   (kit original, before the patch)
```

The two template files were NOT in the stage script output. They were added to `model/` later (see section 3).

## 3. Chat template

| File | Size | sha256 |
| --- | --- | --- |
| served `chat_template.jinja` | 11,780 | `defbdfc2a28ea921d9ad4c6be7244fa82c53f6d552507527a944ad4cbdd24a6d` |
| `chat_template.jinja.bak-pre-tools-nl` | 11,410 | `7d1356675e429ec4a55cbcc1a6fbeb40615dbeb257aca25b8c3705a42087f33e` |

- The backup is byte-identical to `files/chat_template.jinja` of the MiaAI kit checkout.
  The kit checkout is at commit `6f7d1590ad49a2b8995188e45d7b9db31e677452` (short form `6f7d1590`). The kit file has no local change.
- Kit upstream: `https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks`. Its image label states the license `AGPL-3.0`.
  The kit root also has a `LICENSE.MIT` file. Which files it covers was NOT checked.
- The kit template is a port of `encoding/encoding.py` of the DeepSeek repo.
- The patch is in `template-patch.diff` (one hunk, 6 lines added, 2 removed). The template file itself is NOT in this repo.
- What the patch does: for a first user message with no system message, it (a) puts a blank line before the tools block
  (`"\n\n" + tools_tmpl`), and (b) renders `response_format` as a `## Response Format:` block, as the DeepSeek encoder does.
- Patch time: 20:39Z (file modification time of both files). No script or command for the patch was found on the head node.
  Shell history was NOT read.
- The view build did not install any template. The original was copied into `model/` before the patch. The copy command was not found.
- The TensorFold engine reads the template from the model directory when it starts. By the file times,
  suite runs `02` and `03` (servers started 20:10Z and 20:31Z) used the unpatched template.
  Runs `04` and later (server started 20:46Z or later) used the patched template. This is an inference from times.
- The vLLM baseline (`01`) ran before the patch. The kit starts vLLM with `--tokenizer-mode deepseek_v41`
  and an optional `--chat-template` from its `CHAT_TEMPLATE` setting. The value of that setting was NOT read (it is in the kit `.env`).

## 4. Engine images

### vLLM baseline (MiaAI kit)

| Item | Value |
| --- | --- |
| Image | `ghcr.io/miaai-lab/deepseek-v4.1-flash-exl3-2x-dgx-sparks@sha256:2f0cf3adc0f989c1d446be274df864eb799630175f604c3b22b71b7205971dce` |
| Image id | `sha256:4cdba4e946da2d19bf5b5a20c6d3a1a4bf421fa4d6db5082f271a986168176cb` |
| Image label | "DeepSeek-V4.1-Flash EXL3 2.9bpw on 2x DGX Spark (GB10/SM121), vLLM + exllamav3 overlay" |
| Kit commit | `6f7d1590ad49a2b8995188e45d7b9db31e677452` |
| Container | `dsv41-1017-exl3-head` (head), created 18:40:33Z, stopped 20:00:49Z |
| Main settings (container env) | TP=2, `MAX_MODEL_LEN=262144`, `MAX_NUM_SEQS=1`, `MAX_NUM_BATCHED_TOKENS=1536`, `GPU_MEM_UTIL=0.88`, `KV_BLOCK_SIZE=64`, `KV_CACHE_MEMORY_BYTES=2684354560`, `QUANTIZATION=exl3`, `SPEC_METHOD=dspark`, `DSPARK_TOKENS=3`, `LANGUAGE_MODEL_ONLY=1`, `PREFIX_CACHE_RETENTION_INTERVAL=4096` |
| Served model name | `DeepSeek-v4.1-Flash-1017-EXL3` on `172.17.0.1:8888` |

The kit checkout on the head node (`runtime/`) has local changes in `start.sh` and `scripts/boot-shape-warmup.sh`.
The changes bind the API to `127.0.0.1` or `172.17.0.1` (new `API_HOST` and `WORKER_API_HOST`), add `--interface 127.0.0.1` to the health `curl`,
pass `VLLM_API_KEY` through, refuse to remove an existing container, disable `./start.sh restart`, and add SSH keep-alive options.
They do not change the model, the weights, or the sampling.
The kit is a pinned `git bundle` checkout: its parent commits are not in the object store.

### TensorFold (Python engine, TP=2)

- Source repo: `https://github.com/Elevatormusic/TensorFold` (remote `origin`). Upstream: `https://github.com/ashhart/TensorFold` (remote `upstream`).
- Image tag rule: the tag is the first 12 hex digits of the TensorFold commit that was built.
- Base image: `nvcr.io/nvidia/pytorch:26.07-py3`, image id `15b8de8baabe`, created 2026-07-25 05:02:48Z.
  `docker images --digests` lists two repo digests: `sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c` and `sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3`.
- Build log values: `tensorfold==0.6.6`, `xgrammar 0.2.8`, torch `2.13.0a0+9186a08b2c.nv26.7.59513937`, Python 3.12 (aarch64).
- Dockerfile: `docker/_raw_Dockerfile` (raw copy of `build-dsml/Dockerfile`). It pins torch, torchvision, triton and numpy to the NGC versions and installs `/opt/tensorfold[grammar]` in editable mode.

Images on the head node (the worker has the same images, except `fd2365042f56`, which is on the head only):

| Tag | Image id | Built (UTC) | Commit subject |
| --- | --- | --- | --- |
| `469171515cd3` | `ff8c9c0683e8` | 19:49:53Z | test(server): the keep-alive default test's child process imports this checkout's src (#23) |
| `4a413d9c732b` | `63624bba121e` | 20:26:27Z | fix(cuda): tool_choice "required" and named functions on DeepSeek-V4.x DSML templates |
| `90c2d59c8179` | `cb2446a7231a` | 20:41:30Z | fix(dsv41): leading system messages and json_object as DeepSeek's encoding renders them |
| `4f45cdcfdf56` | `21f1eceb7c3d` | 21:04:50Z | feat(dsv41): structured output on the DeepSeek-V4.1 engine (json_schema, regex, choice, grammar) |
| `7f80c3d267da` | `78233795511a` | 21:29:43Z | fix(dsv41): mask grammar logits inside inference mode |
| `d5c9d3cdf4fa` | `535d671ab99f` | 22:14:46Z | fix(dsv41): one failed stream must not break the --parallel lane |
| `20dade5b2945` | `17e417753a87` | 22:34:35Z | test(dsv41): a --parallel reply ends at its grammar stop with ignore_eos |
| `fd2365042f56` | `0679692a9e8a` | 22:46:30Z | Merge branch 'v41-grammar' into v41-lane-faults |

All images are about 24.5 GB.

`git -C tf-lane log --oneline -12` (branch `v41-lane-faults`, HEAD `fd2365042f56fcf726cf4071132c8b779ad2eae0`):

```
fd236504 Merge branch 'v41-grammar' into v41-lane-faults
c4b3cc5f Merge branch 'v41-dsml-tool-choice' into v41-grammar
4dfc261f fix(dsv41): a device fault on a one-rank --parallel lane ends the process
003dd662 fix(dsv41): reject structured-output controls that compete with json_object
20dade5b test(dsv41): a --parallel reply ends at its grammar stop with ignore_eos
92788665 Merge branch 'v41-grammar' into v41-lane-faults
d096e0ac fix(dsv41): coordinate a follower-only stream failure; break the lane on a CUDA fault
83777a3e fix(dsv41): end a constrained reply at its grammar stop token when eos is off
d18d1261 Merge branch 'v41-dsml-tool-choice' into v41-grammar
131937bf fix(dsv41): review findings on the DSML gate and json_object
d5c9d3cd fix(dsv41): one failed stream must not break the --parallel lane
7f80c3d2 fix(dsv41): mask grammar logits inside inference mode
```

Branches: `v41-dsml-tool-choice` (DSML tool choice, template fixes), `v41-grammar` (structured output), `v41-lane-faults` (lane-fault handling, merges both).
The base work is on `deepseek-v41-zig`; its top commit in the clone is `3ff3b2ac`.

### Launch settings (TensorFold, from `scripts/_raw_tfv41.sh`)

`--tp 2`, `--context 1048576`, `--vision`, `--parallel 4`, `--mtp-drafts 5`, `--temperature 1.0`, `--top-p 1.0`,
`TF_DS_REPLAY=1`, `TF_DS_REPLAY_FLOOR=1`, `TF_DS_PREFILL_CHUNK=2048`, `TF_DS_WARM_LENGTHS=1,17,33,131,514,1024,2113`,
served name `DeepSeek-v4.1-Flash-1017-TF` with alias `DeepSeek-v4.1-Flash-1017-EXL3`, API on `172.17.0.1:8888`.

## 5. Node facts

GPU, driver, kernel, OS and Docker are the same on both nodes. CPU and memory were read on the head node only.

| Item | Value |
| --- | --- |
| GPU | `NVIDIA GB10` (one per node), driver `580.178.04`, CUDA `13.0` (from `nvidia-smi`) |
| Kernel | `7.0.0-1019-nvidia` |
| OS | Ubuntu 24.04.5 LTS (Noble Numbat) |
| Docker | `Docker version 29.6.2, build dfc4efb` |
| CPU | 20 cores (Cortex-X925 and Cortex-A725), aarch64 |
| Memory | about 121 GiB unified memory per node |
| Head / worker address (QSFP link) | `10.100.80.1` / `10.100.80.2` on `enp1s0f1np1` (200 Gb/s, MTU 9000) |
| NIC names in the launcher env | `NCCL_SOCKET_IFNAME=enp1s0f1np1`; TensorFold: `TF_RDMA_DEVICES=rocep1s0f1,roceP2p1s0f1`, `NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1`, `TF_RDMA_GID_INDEX=3`, `NCCL_IB_GID_INDEX=3`; vLLM kit: `NCCL_IB_HCA=rocep1s0f1`, `GLOO_SOCKET_IFNAME=enp1s0f1np1`, `NCCL_IB_GID_INDEX=3` |
| RDMA devices present | `rocep1s0f0`, `rocep1s0f1`, `roceP2p1s0f0`, `roceP2p1s0f1` |

## 6. Third-party reference files

The DeepSeek reference encoder `encoding.py` is NOT in this repo.
It comes from `deepseek-ai/DeepSeek-V4.1-Flash-1017` at revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, path `encoding/encoding.py`
(37,962 bytes, sha256 `13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2`). The license of that repo is MIT (Copyright (c) 2023 DeepSeek).
`tests/v41_differential.py` needs it. Put it in a directory and pass that directory as `ENCODING_DIR`.

The five conformance fixtures in `encoding-fixtures/` come from `encoding/tests/` of the same repo and revision. They are unchanged. Their license is MIT (Copyright (c) 2023 DeepSeek).
The file sizes match the Hugging Face listing for that revision.

## 7. Not found or not done

- The command that installed and patched `chat_template.jinja` in the view: not found on disk. Shell history was not read.
- No stored output of `tests/v41_conformance.py` or `tests/v41_differential.py` was found on the head node (only the fixtures).
- A fourth TensorFold run, `suite-tf4.log` (image `4f45cdcfdf56`), stopped after 18 tests without a summary.
  The engine broke the lane on the first constrained request (`RuntimeError: Inplace update to inference tensor outside InferenceMode`). It was rolled back at 21:27Z. Commit `7f80c3d2` fixes this. The log is not in this folder.
- The Hermes agent gate (`08_*`) has no run time in its files. The files were copied to the head node at 21:54:25Z.
  Image `7f80c3d267da` was serving from 21:39Z to 22:19Z.
