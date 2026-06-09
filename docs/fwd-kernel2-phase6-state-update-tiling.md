# Kernel 2 Phase 6 State Update Tiling

This note explains the phase 6 GEMM in `csrc/smxx/fwd_kernel2.cuh`:

```cpp
gemm(thr_mma, ring_A_kr[slot](_,_,Int<0>{}), tCrB_u_arr[bi](_,_,Int<0>{}), u_acc[bi]);
```

The phase 6 state update is:

```text
s_acc[D, D] = s_acc[D, D] * g_total[D] + k_restored_t[D, 16] @ U[16, D]
```

For the current kernel configuration:

```text
D     = 128
CHUNK = 16
MMA   = 16x16x16 logical tile
```

So the update term is:

```text
k_restored_t[128, 16] @ U[16, 128] = update[128, 128]
```

Phase 6 computes that `128 x 128` update in `16 x 16` tiles.

## State Tile Grid

Think of the state matrix as an `8 x 8` grid of `16 x 16` tiles:

```text
state columns / D
        0    16    32    48    64    80    96   112
      +-----+-----+-----+-----+-----+-----+-----+-----+
r  0  | W0a | W0b | W1a | W1b | W2a | W2b | W3a | W3b |
o 16  | W0a | W0b | W1a | W1b | W2a | W2b | W3a | W3b |
w 32  | W0a | W0b | W1a | W1b | W2a | W2b | W3a | W3b |
s 48  | W0a | W0b | W1a | W1b | W2a | W2b | W3a | W3b |
  64  | W0a | W0b | W1a | W1b | W2a | W2b | W3a | W3b |
  80  | W0a | W0b | W1a | W1b | W2a | W2b | W3a | W3b |
  96  | W0a | W0b | W1a | W1b | W2a | W2b | W3a | W3b |
 112  | W0a | W0b | W1a | W1b | W2a | W2b | W3a | W3b |
      +-----+-----+-----+-----+-----+-----+-----+-----+
```

Each cell is one `16 x 16` state tile.

The four MMA warps split the state columns:

```text
warp 0 -> cols   0..31 = two 16-column tiles
warp 1 -> cols  32..63 = two 16-column tiles
warp 2 -> cols  64..95 = two 16-column tiles
warp 3 -> cols  96..127 = two 16-column tiles
```

Inside a warp:

```text
bi = 0 -> first  16-column tile owned by this warp
bi = 1 -> second 16-column tile owned by this warp
```

## Why `slot` Indexes `ring_A_kr`

`ring_A_kr[slot]` holds the current row tile of `k_restored_t`.

For a row block `m`, this tile is:

```text
k_restored_t[m*16 : m*16+16, 0 : 16]
```

That same left operand is reused for both `bi = 0` and `bi = 1`, because both
column blocks for a warp use the same row block of `k_restored_t`.

`slot` is the ring-buffer slot for the current prefetched row block. With the
current code:

```cpp
constexpr int PREFETCH = 1;
```

so:

```text
slot = m % PREFETCH = m % 1 = 0
```

The code is still written in ring-buffer form, so `slot` means "the current
prefetched row-block buffer" even though it is always index `0` today.

## Why `bi` Indexes `tCrB_u_arr` And `u_acc`

`tCrB_u_arr[bi]` holds the `U` tile for one of the two column blocks owned by
the warp:

```text
tCrB_u_arr[0] -> U[:, first  16 columns owned by this warp]
tCrB_u_arr[1] -> U[:, second 16 columns owned by this warp]
```

Each different `U` column tile produces a different `16 x 16` output/update
tile, so the accumulator also uses `bi`:

```text
u_acc[0] -> update for first  column tile
u_acc[1] -> update for second column tile
```

## Concrete Example

Take:

```text
warp_id = 1
m       = 3
```

The current row block is:

```text
rows = 3 * 16 .. 3 * 16 + 15 = 48..63
```

Warp 1 owns columns:

```text
cols 32..63
```

Split into two `bi` tiles:

```text
bi = 0 -> cols 32..47
bi = 1 -> cols 48..63
```

The two GEMMs are:

```text
u_acc[0] = k_restored_t[48..63, 0..15] @ U[0..15, 32..47]
u_acc[1] = k_restored_t[48..63, 0..15] @ U[0..15, 48..63]
```

Visually:

```text
                  B0                         B1
          U[0..15, 32..47]          U[0..15, 48..63]
                 |                          |
                 v                          v

A = k_restored_t[48..63, 0..15]

        [16x16] @ [16x16]        [16x16] @ [16x16]
             |                         |
             v                         v

     state[48..63,32..47]     state[48..63,48..63]
          u_acc[0]                 u_acc[1]
```

That maps directly to:

```text
ring_A_kr[slot] -> A, current row block, shared across both `bi`
tCrB_u_arr[0]   -> B0, first  U column block owned by this warp
tCrB_u_arr[1]   -> B1, second U column block owned by this warp
u_acc[0]        -> C0/update result for B0
u_acc[1]        -> C1/update result for B1
```

Short version:

```text
slot chooses the current row-block buffer.
bi chooses one of the two column tiles owned by this warp.
```
