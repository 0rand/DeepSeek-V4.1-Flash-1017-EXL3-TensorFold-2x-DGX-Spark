# Display-only KV variant

Use [the opt-in guide](../../docs/DISPLAY-KV.md) for setup, cutover and restore.
This experiment is not the ordinary recipe. Full hardware validation is pending.

- `apply.py` checks TensorFold revision `07aa02585f64cfb04c60a96de6554c0d4708ed46`
  and three pristine source hashes. It transforms only model, engine and multi.
  It validates all generated Python before it writes any file.
- `overlay/display_kv.py` is the only installed overlay. It maps display memory
  for packed KV planes. Ordinary ring buffers and token IDs stay outside it.
- `prepare.py` creates a private bundle. It does not change the active `.env`
  or run a service command. It forces spill off and prefill chunk 512.
- `probe.py` checks allocation and mapping. It is not a serving acceptance test.
- `tests/` uses standard-library fixtures. Source integration tests run only
  when `TF_DISPLAY_SOURCE` names an explicit checkout.

The top-level launcher manages the derived image and runtime DRM preparation.
It checks both ranks before a cutover and probes both nodes before launch.
Never stop GDM. Never use `docker exec` in a serving rank. Stop both ranks and
all GPU or DRM clients before a DRM change. No new performance claim is made.

The split excludes tool guards, allocator policy, cache metrics and the SSD
spill port. Display-only spill is disabled, not an optional tuning setting.
See [NOTICE.md](NOTICE.md) for display mapping attribution and license notes.
