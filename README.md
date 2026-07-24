# FlashKDA++

FlashKDA++ is an H100-focused optimization of
[MoonshotAI/FlashKDA](https://github.com/MoonshotAI/FlashKDA), a CUTLASS
implementation of Flash Kimi Delta Attention. It keeps the `flash_kda`
package and forward API compatible while reducing complete-call latency
relative to the original FlashKDA source.

The project name is **FlashKDA++**; the GitHub slug is
`FlashKDA-PlusPlus` because GitHub does not permit `+` in repository names.

## What changed

- Replaced the fragmented K1-to-K2 TensorMap workspace transfers with
  contiguous Hopper bulk-image copies.
- Changed K1 to a 128-thread cooperative schedule to improve CTA residency.
- Kept the K2 recurrent state in registers and removed avoidable work from
  the chunk cycle.
- Improved variable-length scheduling and added a trusted uniform-packed
  fast path.
- Strengthened host-side validation and retained an exact-equality
  correctness gate for tails, state modes, and packed sequences.

The largest finding was that K1's workspace tail was limited by TMA request
fragmentation, not HBM bandwidth. Changing the workspace representation
removed that backpressure; the later scheduling changes built on top of it.

## H100 results

The directly comparable metric is the full `flash_kda.fwd` call with an
external FP32 initial/final state, not an isolated kernel phase. The table
below uses `T=8192`, `D=128`, an H100 80 GB, 30 warmups, 200 timed calls per
repeat, five repeats, and the arithmetic mean of all 1,000 samples.

| H96 workload | Original FlashKDA | FlashKDA++ | Speedup | Latency reduction |
|---|---:|---:|---:|---:|
| Fixed `[8192]` | 1.5153 ms | 1.0628 ms | **1.43x** | 29.9% |
| Uneven packed `[1300, 547, 2048, 963, 271, 3063]` | 1.5644 ms | 0.9331 ms | **1.68x** | 40.4% |
| Uniform packed `[1024] x 8` | 1.3819 ms | 0.9270 ms | **1.49x** | 32.9% |

These current stripped-source measurements reproduce a **1.43-1.68x**
improvement over stock FlashKDA. They were recorded with the same protocol
but in separate H100 allocations. A paired baseline/latest/baseline campaign
on the preceding fully stacked snapshot measured **1.42-1.72x** over six
H64/H96 cases, with a **1.55x geometric mean** and at most 0.39% baseline
drift. The stripped source was 1.3-1.6% slower in its H96 replay; that small
cross-allocation difference is not attributed to a specific code change.

See [H100 optimization results](docs/H100_RESULTS.md) for provenance,
correctness criteria, limitations, and the complete paired table.

### How this relates to the upstream “up to 2.2x” claim

The claims use different denominators:

- Upstream measures `FLA chunk_kda / stock FlashKDA` on H20. The checked-in
  [H20 report](BENCHMARK_H20.md) ranges from 1.85x to 2.31x against that FLA
  path.
- This repository measures `stock FlashKDA / FlashKDA++` on H100 and obtains
  1.43x to 1.68x for the current optimized source.

The hardware, baseline implementation, and source revisions differ, so the
two ratios should not be multiplied into a synthetic end-to-end claim.

## Requirements

- NVIDIA SM90 or newer
- CUDA 12.9 or newer
- PyTorch 2.4 or newer

The setup currently recognizes `90a`, `100a`, `103a`, and `120a`. The new
performance results above are H100-only.

## Installation

```bash
git clone --recurse-submodules \
  https://github.com/Itssshikhar/FlashKDA-PlusPlus.git
cd FlashKDA-PlusPlus
pip install -v --no-build-isolation .
```

By default, the build detects the visible CUDA device. For wheel or CI
builds, compile every supported architecture explicitly:

```bash
FLASH_KDA_CUDA_ARCHS=all pip install -v --no-build-isolation .
```

`FLASH_KDA_CUDA_ARCHS` accepts `auto` (the default), `all`, or a
comma-separated list such as `90a,100a`.

## Python API

```python
import flash_kda

flash_kda.fwd(
    q, k, v, g, beta, scale, out,
    A_log, dt_bias, lower_bound,
    initial_state=None,
    final_state=None,
    cu_seqlens=None,
    uniform_seq_len=None,
)
```

| Parameter | Dtype | Shape | Description |
|---|---|---|---|
| `q` | bf16 | `[B, T, H, K]` | Query |
| `k` | bf16 | `[B, T, H, K]` | Key |
| `v` | bf16 | `[B, T, H, V]` | Value |
| `g` | bf16 | `[B, T, H, K]` | Gate before activation |
| `beta` | bf16 | `[B, T, H]` | Beta logits; sigmoid is applied internally |
| `scale` | float | scalar | Scaling factor |
| `out` | bf16 | `[B, T, H, V]` | Output buffer, written in place |
| `A_log` | fp32 | `[H]` | Log-gate parameter |
| `dt_bias` | fp32 | `[H, K]` | Gate bias |
| `lower_bound` | float | scalar | Gate lower bound in `[-5.0, 0]` |
| `initial_state` | bf16/fp32/None | `[B or N, H, V, K]` | Optional recurrent state |
| `final_state` | bf16/fp32/None | `[B or N, H, V, K]` | Optional final-state output |
| `cu_seqlens` | int64/None | `[N+1]` | Packed-sequence offsets |
| `uniform_seq_len` | int/None | scalar | Trusted equal-length hint for packed input |

Current constraints:

- `K = V = 128`.
- Inputs must be contiguous CUDA tensors with the dtypes above.
- With `cu_seqlens`, `B` must be 1 and `T` is the packed token count.
- `uniform_seq_len` is a trusted hint: pass it only when every sequence in
  `cu_seqlens` has exactly that length. It avoids synchronizing with the host
  to inspect the offsets.

## FLA backend

With `flash-linear-attention >= 0.5.0`, FlashKDA can be selected by FLA's
`chunk_kda` dispatch:

```bash
pip install -U flash-linear-attention
```

```python
import torch
from fla.ops.kda import chunk_kda

with torch.inference_mode():
    out, final_state = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
        use_gate_in_kernel=True,
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        transpose_state_layout=True,
        cu_seqlens=cu_seqlens,
    )
```

Set `FLA_FLASH_KDA=0` to opt out. Enable Python logging at `INFO` level to
inspect backend dispatch decisions. See
[fla-org/flash-linear-attention#852](https://github.com/fla-org/flash-linear-attention/pull/852)
for the integration.

## Correctness and benchmarking

Run the focused exact-equality gate:

```bash
python tests/check_optimized_fwd.py
```

Run the repository test suite:

```bash
bash tests/test.sh
```

Benchmark only FlashKDA++ across the three H100 workload shapes:

```bash
python benchmarks/bench_fwd.py \
  --flash-only --mode all --H 96 --D 128 \
  --warmup 30 --iters 200 --repeats 5
```

The exact gate checks nine representative fixed/packed, full/tail,
BF16/FP32/no-state configurations with `torch.equal`. It is stricter than an
`allclose` tolerance, but it is intentionally smaller than the complete
`T=8192, H=96` timing shape; see the results note for that limitation.

## Repository layout

| Path | Purpose |
|---|---|
| `flash_kda/` | Stable Python API |
| `csrc/` | Production CUDA/C++ implementation |
| `tests/` | Reference implementation and correctness tests |
| `benchmarks/` | Reproducible timing and profiling entry points |
| `docs/` | Upstream design notes and optimization evidence |
| `cutlass/` | Pinned CUTLASS submodule |

Generated extensions, build trees, raw benchmark artifacts, and Nsight
reports are intentionally excluded from Git.

## Upstream and citation

The FlashKDA algorithm, original CUDA implementation, FLA integration, and
design deep dive come from
[MoonshotAI/FlashKDA](https://github.com/MoonshotAI/FlashKDA). Read the
[FlashKDA v1 deep dive](docs/20260420-flashkda-v1-deep-dive.md) for the
original design.

```bibtex
@misc{flashkda2026,
  title        = {FlashKDA: Flash Kimi Delta Attention},
  author       = {Yutian Chen and Zhiyuan Li and Yucheng Wang and Ming Wei},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/MoonshotAI/FlashKDA}},
}
```
