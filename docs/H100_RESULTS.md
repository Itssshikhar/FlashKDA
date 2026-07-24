# H100 optimization results

This note records the performance claim for FlashKDA++, how it differs from
the upstream FlashKDA-versus-FLA claim, and the evidence used to accept the
optimized source.

## Scope

The primary comparison is:

```text
original FlashKDA full-call latency
-----------------------------------
FlashKDA++ full-call latency
```

Both sides measure the complete `flash_kda.fwd` call with external FP32
initial and final state. The baseline is upstream commit `4131040`, before
the H100 optimization work. These are not FLA `chunk_kda` timings and are
not isolated K1 or K2 kernel spans.

## Current stripped implementation

The publishable source retains only the accepted production paths. Its H96
replay used:

- NVIDIA H100 80 GB HBM3;
- `T=8192`, `D=128`, `H=96`;
- external FP32 initial and final state;
- CUDA events around the full `flash_kda.fwd` call;
- 30 warmups;
- 200 timed calls per repeat and five repeats;
- the arithmetic mean of all 1,000 samples.

| Workload | Original, baseline A | Current stripped | Speedup | Reduction |
|---|---:|---:|---:|---:|
| Fixed `[8192]` | 1.5153 ms | 1.0628 ms | 1.426x | 29.9% |
| Uneven packed `[1300, 547, 2048, 963, 271, 3063]` | 1.5644 ms | 0.9331 ms | 1.677x | 40.4% |
| Uniform packed `[1024] x 8` | 1.3819 ms | 0.9270 ms | 1.491x | 32.9% |

The original baseline and stripped replay used the same protocol but were
recorded in different H100 allocations. They establish that the improvement
survives removal of the experiment-only paths, but they should not be read
as a formal confidence interval.

The current stripped build also produced the following complete-call means:

| External state | Fixed | Uneven packed | Uniform packed |
|---|---:|---:|---:|
| BF16 | 1.0274 ms | 0.9095 ms | 0.9050 ms |
| None | 1.0571 ms | 0.8960 ms | 0.8722 ms |
| FP32 | 1.0628 ms | 0.9331 ms | 0.9270 ms |

All three state modes passed the focused exact-equality checks.

## Paired optimization campaign

The preceding fully stacked snapshot was measured between two executions of
the original source in one H100 allocation. The paired baseline is the mean
of baseline A and baseline B.

| Case | Paired original | Optimized snapshot | Speedup | Reduction |
|---|---:|---:|---:|---:|
| H96 fixed | 1.5142 ms | 1.0475 ms | 1.4455x | 30.82% |
| H96 uneven packed | 1.5647 ms | 0.9210 ms | 1.6989x | 41.14% |
| H96 uniform packed | 1.3825 ms | 0.9120 ms | 1.5158x | 34.03% |
| H64 fixed | 1.2928 ms | 0.9094 ms | 1.4216x | 29.66% |
| H64 uneven packed | 1.1103 ms | 0.6445 ms | 1.7227x | 41.95% |
| H64 uniform packed | 0.9329 ms | 0.6143 ms | 1.5186x | 34.15% |

The equal-weight geometric mean is `1.5496x`. Summing one representative
latency from each row gives a 35.251% reduction. Baseline A and B differed by
at most 0.39%, much less than the observed 29.7-42.0% reductions.

The stripped H96 replay was 1.3-1.6% slower than this fully stacked snapshot.
Because those small differences came from separate allocations, they are not
assigned to a particular source change. H64 was not rerun after stripping.

## Accepted changes

The publishable implementation keeps five measured winners:

1. **Bulk-image K1-to-K2 workspace transport.** The original six TMA tensor
   stores moved 13,824 bytes per K1 CTA through highly fragmented TensorMap
   segments. Contiguous `cp.async.bulk` byte-image transfers removed the
   request-path backpressure. This was the largest isolated improvement.
2. **128-thread K1.** Higher CTA residency improved latency hiding by roughly
   5-6% in K1.
3. **Register-resident K2 state.** The recurrent BF16 state remains in the
   MMA accumulator representation instead of repeatedly round-tripping
   through shared memory.
4. **K2 cycle trimming.** Redundant per-chunk state work was removed while
   preserving the original arithmetic order.
5. **Packed-sequence scheduling.** Uneven sequences receive a better work
   order, while equal packed sequences may use the explicit
   `uniform_seq_len` fast path.

Several plausible alternatives did not survive end-to-end measurement:
defragmenting the user-facing K2 output transfer, explicit phase overlap,
shared-state WGMMA, fused K1/K2 CTAs, and a grid-synchronized persistent
kernel. They are not present in the production build.

## Correctness gate

Before timing the optimized snapshot, the comparison job ran nine
configurations now published as `tests/check_optimized_fwd.py`:

- BF16 full tile and tail;
- FP32-state tail;
- zero state;
- output-state only;
- uneven packed input;
- three uniform-packed cases covering a tail, a full tile, and stage reuse.

A case passes only when both available tensors satisfy:

```python
torch.equal(out_kernel, out_reference)
torch.equal(final_kernel, final_reference)
```

The diagnostic `allclose` calls in the checker run only after exact equality
has already failed; they cannot turn a mismatch into a pass.

This gate covers arithmetic order, tails, state import/export, and packed
dispatch with small `H=2` cases. It does not run the slow Torch reference at
the complete timed `T=8192, H=64/96` shapes. A stronger publication gate
would additionally require exact original-versus-optimized equality on every
full benchmark shape.

## Relation to upstream performance reports

The upstream H20 and GB200 reports answer a different question:

```text
FLA implementation latency
--------------------------
stock FlashKDA latency
```

For example, `BENCHMARK_H20.md` reports 1.85-2.31x against FLA `chunk_kda`.
The report generator times random inputs but performs no numerical comparison
inside the timing command; the repository's correctness tests are separate.

FlashKDA++ instead reports 1.43-1.68x for the current source against stock
FlashKDA on H100. Different hardware, revisions, and denominators make it
invalid to multiply the two ratios.

## Reproduction

After building FlashKDA++ on a supported CUDA device:

```bash
python tests/check_optimized_fwd.py

python benchmarks/bench_fwd.py \
  --flash-only --mode all --H 96 --D 128 \
  --warmup 30 --iters 200 --repeats 5
```

The benchmark prints means, minima, and maxima. Generated extensions, raw
cloud-run artifacts, and Nsight reports are intentionally excluded from the
repository; this note retains the exact headline values and protocol without
shipping hundreds of megabytes of machine-specific output.
