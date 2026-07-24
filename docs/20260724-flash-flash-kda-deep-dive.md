# Flash-Flash KDA: A Deep Dive

*2026-07-24*

Flash-Flash KDA keeps the algorithm, `CHUNK = 16` decomposition, and numerical
order of FlashKDA v1. The optimization work instead focuses on how the two
forward kernels move data and occupy an H100. This report covers the private
workspace redesign, K1 and K2 scheduling changes, packed-sequence fast paths,
and the experiments that did not survive end-to-end measurement.

## 1. Keeping the Two-Kernel Design

FlashKDA divides the forward pass along two different parallelism axes:

- **K1 (token-parallel):** gate activation, Q/K normalization, decay
  application, `L`/`Mqk` construction, and the `16 × 16` inverse.
- **K2 (sequence/head-parallel):** the chunk-by-chunk recurrence, output
  projection, and recurrent-state update.

K1 produces six tensors for every `(chunk, head)` and writes them to a private
workspace that K2 consumes. For `T=8192`, `H=96`, this boundary transfers
about 1.36 GB when the write and read sides are counted together.

Removing that round trip with a fused producer/consumer CTA was attractive,
but it also coupled K1's abundant chunk parallelism to K2's serial recurrence.
The fused CTA needed K1 storage and the full K2 state at the same time. At two
resident CTAs per SM, the resulting register pressure caused spills that
slowed both halves through L1; avoiding the spills reduced occupancy instead.
The tested fused designs ranged from break-even to substantially slower, so
Flash-Flash KDA retains the independently scheduled two-kernel design.

## 2. The Workspace Was Fragmented, Not Bandwidth-Bound

Each K1 CTA publishes 13,824 bytes:

```text
3 × [16,128] BF16  = 12,288 bytes
1 × [128] FP32     =    512 bytes
2 × [16,16] BF16   =  1,024 bytes
```

The original implementation stored these tensors through TensorMap
descriptors derived from their swizzled shared-memory layouts. The largest
common contiguous vector was only 16 bytes, so the three 4 KiB tensors were
traversed as 256 segments each, the two 512-byte matrices as 32 segments
each, and the flat gate vector as one segment: **833 segments per CTA**.

Profiling localized almost all of the workspace tail to the
`cp.async.bulk.tensor` issue intervals, not descriptor setup, commit-group
instructions, or the final wait. Reordering stores moved the latency with
issue position rather than tensor identity, while waiting after every store
did not remove the later stalls. The shared per-SM TMA engine was draining
many tiny segments from all resident CTAs even though HBM utilization was
only about 40-50%.

The fix was to change the private K1→K2 workspace ABI. K1 now stores the raw
byte image of each shared-memory tensor with one non-tensor
`cp.async.bulk` copy. K2 restores that byte image directly into the same
layout:

```text
K1 swizzled shared memory
    → contiguous private workspace slot
    → byte-identical K2 shared memory
```

No public tensor layout or arithmetic changes, and no repacking kernel is
needed. The per-store issue time fell from roughly 0.9-5.0 µs to timer
resolution. In the isolated transport comparison, this reduced K1+K2 time by
23% for fixed length, 34% for uneven packed input, and 37% for uniform packed
input. It was the largest optimization in the project.

## 3. Retuning K1 and K2

After the workspace boundary stopped dominating, the kernels responded to
different changes.

- **K1 uses 128 threads.** The original 256-thread CTA was capped at eight
  resident blocks by the H100 thread limit. Four warps let shared memory
  become the occupancy limit instead, allowing roughly ten resident CTAs and
  improving K1 by about 5-6%. A faster gate-prefix subphase did not improve
  the whole kernel; K1 benefited from more latency hiding, not isolated
  instruction shaving.
- **K2 keeps state in registers.** Each MMA warp owns 32 columns of the
  recurrent BF16 state across the complete chunk loop. Phase 1 converts the
  resident C-fragment ownership into an MMA-B operand with `MOVM_T`; Phase 6
  updates the C fragments in place with FP32 FMA and BF16 rounding. Shared
  memory is used only to import the initial state and export the optional
  final state.
- **K2 trims the chunk boundary.** Output-stage acquisition moves to the
  first phase that needs it, and a redundant post-update barrier is removed.
  The change reduced K2 by about 7% for uniform packed input and 4% for
  uneven input while remaining neutral within noise for the fixed case.

The state representation is deliberate. A corrected shared-state WGMMA
prototype passed correctness but made K2 2.06× slower for uniform packed
input and 3.35× slower for the fixed workload. The tensor instruction was
not the problem; repeatedly staging and updating the 128 × 128 state in
shared memory cost more than WGMMA saved.

## 4. Packed-Sequence Scheduling

Packed workloads add scheduling overhead and can leave a long recurrence
chain running after shorter sequences finish.

For uneven batches with at most 32 sequences, K2 ranks sequence lengths on
the device and launches longer chains first. Ties retain input order, and no
permutation buffer is materialized. This improves the tail without changing
the mapping of outputs or states back to their original sequence indices.

Uniform packed batches can supply the optional `uniform_seq_len` hint. That
lets K1 and K2 use exact tile counts and fixed-length indexing instead of
scanning `cu_seqlens` inside each CTA. The hint is intentionally trusted: it
avoids a device-to-host synchronization, so callers must provide it only when
every packed sequence has the stated length.

## 5. Optimizations We Rejected

Several changes improved a local metric but lost end to end:

- **Defragmenting K2's user-facing output TMA** shortened individual issue
  spans but regressed uneven workloads by about 7%. The `[T,H,D]` output is
  interleaved in global memory, unlike the private contiguous workspace.
- **Explicitly merging Phase 6 of chunk `t` with Phase 1 of chunk `t+1`**
  regressed K2 by 8-20%. The trimmed kernel already lets warps flow across
  that boundary without a barrier; the explicit merge exposed the next input
  wait and added instructions.
- **Fused K1/K2 CTAs** lost the occupancy and load-balancing advantages of
  the split design. A persistent producer/consumer variant also could not
  safely use a grid-wide software barrier when the logical grid exceeded
  resident capacity.
- **Changing arithmetic order** was not accepted as an optimization. One
  parallel prefix-sum prototype was faster in isolation but produced a
  roughly 0.66% mismatch, so it was replaced with the original sequential
  accumulation order.

The recurring lesson was that a shorter instruction span is not necessarily
a faster kernel. Every retained change had to improve the complete forward
call in a paired run.

## 6. Correctness and H100 Results

The optimized source is checked against the Torch reference on nine
configurations covering full tiles, tails, BF16 and FP32 state, missing
state, uneven packed input, uniform packed input, and pipeline-stage reuse.
Output and final state pass only with exact `torch.equal`; there is no
`atol` or `rtol` acceptance path.

The paired H100 comparison measures the complete `flash_kda.fwd` call with
external FP32 state. It runs original FlashKDA, the optimized source, and the
original source again in one H100 allocation. Each case uses 30 warmups, 200
timed calls, and five repeats.

| Workload | `H=96` speedup | `H=64` speedup |
|---|---:|---:|
| Fixed `[8192]` | 1.45× | 1.42× |
| Uneven packed | 1.70× | 1.72× |
| Uniform packed `[1024] × 8` | 1.52× | 1.52× |

The six-case geometric mean is **1.55×**, and the two original baseline runs
differed by at most 0.39%. The final stripped source separately reproduced a
1.43-1.68× improvement at `H=96`.

See [BENCHMARK_H100.md](../BENCHMARK_H100.md) for the concise tables and
[H100_RESULTS.md](H100_RESULTS.md) for detailed provenance and limitations.

Flash-Flash KDA's main gain does not come from changing KDA mathematics. It
comes from representing the same private data in a form that Hopper can move
efficiently, then scheduling each kernel around the bottlenecks left behind.
