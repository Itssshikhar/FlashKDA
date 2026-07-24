# KDA forward benchmark (Hopper / H100)

- Generated: 2026-07-22

- Command: `python benchmarks/bench_fwd.py --mode all --warmup 30 --iters 200 --repeats 5 --H 96 --D 128` (repeated with `--H 64`)

- Benchmark settings: `warmup=30`, `iters=200`, `repeats=5`

- Both columns measure the complete `flash_kda.fwd` call with FP32 initial and final state.
- `original_flash_kda` is upstream commit `4131040`; its reported mean averages baseline runs immediately before and after the optimized run.

### `T=8192`, `H=96`, `D=128`

| Case | `flash_kda` mean (ms) | `original_flash_kda` mean (ms) | Speedup vs original |
|------|----------------------:|-------------------------------:|--------------------:|
| Fixed | 1.0475 | 1.5142 | 1.45× |
| Varlen, `seq_lens`=[1300, 547, 2048, 963, 271, 3063] | 0.9210 | 1.5647 | 1.70× |
| Varlen, `seq_lens`=`1024 x 8` | 0.9120 | 1.3825 | 1.52× |

### `T=8192`, `H=64`, `D=128`

| Case | `flash_kda` mean (ms) | `original_flash_kda` mean (ms) | Speedup vs original |
|------|----------------------:|-------------------------------:|--------------------:|
| Fixed | 0.9094 | 1.2928 | 1.42× |
| Varlen, `seq_lens`=[1300, 547, 2048, 963, 271, 3063] | 0.6445 | 1.1103 | 1.72× |
| Varlen, `seq_lens`=`1024 x 8` | 0.6143 | 0.9329 | 1.52× |

## vs `flash-linear-attention` (same H100 allocation)

The same benchmark runs also timed FLA's `chunk_kda` and
`chunk_gated_delta_rule`. Raw logs:
`artifacts/official-h100-compare/20260722_100916_official_h100_compare/bench-latest-production-{h96,h64}.log`.

- `fla_chunk_kda` configuration: `use_gate_in_kernel=True`, `use_qk_l2norm_in_kernel=True`, `use_beta_sigmoid_in_kernel=True`, `lower_bound=-5`, `transpose_state_layout=True`
- `fla_chunk_gated_delta_rule` configuration: scalar per-head gate `g` of shape `(1, T, H)`, `use_qk_l2norm_in_kernel=True`, `transpose_state_layout=True`
- The `flash_kda` column repeats the FP32-state means from the tables above.

### `T=8192`, `H=96`, `D=128`

| Case | `flash_kda` mean (ms) | `fla_chunk_kda` mean (ms) | Speedup vs `chunk_kda` | `fla_chunk_gdn` mean (ms) | Speedup vs `gdn` |
|------|----------------------:|----------------------:|--------:|----------------------:|--------:|
| Fixed | 1.0475 | 3.4906 | 3.33× | 1.8929 | 1.81× |
| Varlen, `seq_lens`=[1300, 547, 2048, 963, 271, 3063] | 0.9210 | 3.5122 | 3.81× | 1.8873 | 2.05× |
| Varlen, `seq_lens`=`1024 x 8` | 0.9120 | 3.4880 | 3.82× | 1.8747 | 2.06× |

### `T=8192`, `H=64`, `D=128`

| Case | `flash_kda` mean (ms) | `fla_chunk_kda` mean (ms) | Speedup vs `chunk_kda` | `fla_chunk_gdn` mean (ms) | Speedup vs `gdn` |
|------|----------------------:|----------------------:|--------:|----------------------:|--------:|
| Fixed | 0.9094 | 2.2894 | 2.52× | 1.2454 | 1.37× |
| Varlen, `seq_lens`=[1300, 547, 2048, 963, 271, 3063] | 0.6445 | 2.3303 | 3.62× | 1.2938 | 2.01× |
| Varlen, `seq_lens`=`1024 x 8` | 0.6143 | 2.3325 | 3.80× | 1.2830 | 2.09× |

Across the six cases the `chunk_kda` speedup ranges from 2.52× to 3.82×
(geometric mean 3.45×). For reference, the original FlashKDA baseline
(upstream commit `4131040`) measured 1.78×–2.53× vs `fla_chunk_kda` in the
same allocation (`bench-baseline-{a,b}-{h96,h64}.log`).
