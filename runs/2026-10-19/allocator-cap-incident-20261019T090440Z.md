# Allocator Cap Failure, 2026-10-19 09:04:40 UTC

Read-only capture from both rank containers and earlyoom journals immediately
after the reported failure. Serving image: `tensorfold-dsv41:display-kv-a20775162a93`.

Rank 0 exited with code 75; Docker `OOMKilled=false`. Rank 1 exited with code 1
after losing rank 0. Neither node's earlyoom journal had an entry in the queried
20-minute window.

Rank 0's last guard snapshot:

```text
host_available_bytes: 1612627968
reserved_bytes:       108921880576
allocated_bytes:      108586158080
ceiling_bytes:        108923895808
floor_bytes:          1610612736
released_bytes:       23068672
```

The ceiling allowed only 2,015,232 bytes (1.921875 MiB) of further allocator
growth. The next prefill failed in `model.forward`, expanding embeddings via
`emb[:, None, :].expand(-1, c.hc, -1).contiguous()`:

```text
Tried to allocate 20.00 MiB.
GPU total capacity 121.69 GiB; 3.33 GiB free.
Process memory in use 103.22 GiB; 101.44 GiB allowed.
PyTorch allocated 101.19 GiB; reserved but unallocated 261.17 MiB.
```

The failed request had 103281 prompt tokens, continuing a kept prompt from
98304 tokens. Two preceding requests completed successfully, including a
4155-token response.

This confirms an enforced allocator-cap rejection. It does not prove the host
had 3.33 GiB available: the earlier host `MemAvailable` snapshot was about
1.502 GiB, and host/CUDA measurements were taken at different times. The gap
requires further measurement, not an assumption that either reading is wrong.

Next diagnostic: disable only cap enforcement while retaining cache cleanup,
budget telemetry and earlyoom. No settings or service state were changed as
part of this capture.
