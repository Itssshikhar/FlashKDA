# Flash-Flash KDA

Flash-Flash KDA is an H100-optimized fork of
[MoonshotAI/FlashKDA](https://github.com/MoonshotAI/FlashKDA), providing
high-performance Kimi Delta Attention kernels built on CUTLASS.

## What's new

- **2026-07-24** — Read the [Flash-Flash KDA deep dive](docs/20260724-flash-flash-kda-deep-dive.md) and see the [H100 benchmark](BENCHMARK_H100.md).
- **2026-04-22** — Read the original [FlashKDA v1 deep dive](docs/20260420-flashkda-v1-deep-dive.md).

## Requirements

- SM90 and above
- CUDA 12.9 and above
- PyTorch 2.4 and above

## Installation

```bash
git clone --recurse-submodules \
  https://github.com/Itssshikhar/Flash-Flash-KDA.git
cd Flash-Flash-KDA
pip install -v --no-build-isolation .
```

By default, the build detects the current CUDA device and compiles for that
architecture. For wheel or CI builds, compile all supported architectures:

```bash
FLASH_KDA_CUDA_ARCHS=all pip install -v --no-build-isolation .
```

Supported values are `auto` (default), `all`, or a comma-separated list such
as `90a,100a`.

## Using FlashKDA as an FLA backend

Once installed, FlashKDA is auto-dispatched from
`flash-linear-attention`'s `chunk_kda`. See
[fla-org/flash-linear-attention#852](https://github.com/fla-org/flash-linear-attention/pull/852)
for integration details.

1. Install `flash-linear-attention >= 0.5.0`:

   ```bash
   pip install -U flash-linear-attention
   ```

2. Call `chunk_kda` under `torch.inference_mode()`:

   ```python
   import torch
   from fla.ops.kda import chunk_kda

   with torch.inference_mode():
       out, final_state = chunk_kda(
           q=q, k=k, v=v, g=g, beta=beta,
           scale=scale,
           initial_state=h0,
           output_final_state=True,
           use_gate_in_kernel=True,
           use_qk_l2norm_in_kernel=True,
           use_beta_sigmoid_in_kernel=True,
           safe_gate=True,
           A_log=A_log, dt_bias=dt_bias,
           lower_bound=lower_bound,
           transpose_state_layout=True,
           cu_seqlens=cu_seqlens,
       )
   ```

Set `FLA_FLASH_KDA=0` to fall back to the Triton path. Enable Python logging
at `INFO` level to inspect backend dispatch decisions.

## Performance

- [H100 benchmark](BENCHMARK_H100.md) — Flash-Flash KDA versus original FlashKDA
- [H20 benchmark](BENCHMARK_H20.md) — original FlashKDA versus FLA
- [GB200 benchmark](BENCHMARK_GB200.md) — original FlashKDA versus FLA

## Tests

```bash
bash tests/test.sh
```

- `tests/check_optimized_fwd.py` — exact output and final-state checks
- `tests/test_fwd.py` — Torch-reference and FLA comparisons

## Kernel API

### `flash_kda.fwd`

```python
flash_kda.fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
              initial_state=None, final_state=None, cu_seqlens=None,
              uniform_seq_len=None)
```

| Parameter | Dtype | Shape | Description |
|---|---|---|---|
| `q` | bf16 | `[B, T, H, K]` | Query |
| `k` | bf16 | `[B, T, H, K]` | Key |
| `v` | bf16 | `[B, T, H, V]` | Value |
| `g` | bf16 | `[B, T, H, K]` | Gate before activation |
| `beta` | bf16 | `[B, T, H]` | Beta logits; sigmoid is applied internally |
| `scale` | float | scalar | Scaling factor |
| `out` | bf16 | `[B, T, H, V]` | Output tensor |
| `A_log` | fp32 | `[H]` | Log-gate parameter |
| `dt_bias` | fp32 | `[H, K]` | Gate bias |
| `lower_bound` | float | scalar | Gate lower bound in `[-5.0, 0]` |
| `initial_state` | bf16/fp32/None | `[B, H, V, K]` or `[N, H, V, K]` | Optional initial state |
| `final_state` | bf16/fp32/None | `[B, H, V, K]` or `[N, H, V, K]` | Optional final-state output |
| `cu_seqlens` | int64/None | `[N+1]` | Packed-sequence offsets |
| `uniform_seq_len` | int/None | scalar | Equal-length hint for packed sequences |

- Currently requires `K = V = 128`.
- `initial_state` and `final_state` accept `None`, bf16, or fp32. When both
  are provided, their dtypes must match.
- With `cu_seqlens`, `B` must be 1 and state tensors use
  `[N, H, V, K]`.
- Pass `uniform_seq_len` only when every packed sequence has that length.

## Development

To configure clangd for the CUDA/C++ sources:

```bash
bash setup_clangd.sh
```

## Citation

```bibtex
@misc{flashkda2026,
  title        = {FlashKDA: Flash Kimi Delta Attention},
  author       = {Yutian Chen and Zhiyuan Li and Yucheng Wang and Ming Wei},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/MoonshotAI/FlashKDA}},
}
```
