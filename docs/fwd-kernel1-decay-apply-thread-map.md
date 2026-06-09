# Kernel 1 Decay-Apply Thread Map

This note explains the mapping used in `csrc/smxx/fwd_kernel1.cuh` around:

```cpp
int row = m_blk + ((warp_id + g) % 8);
int col_base = n_blk + g * 8;
int col_tile = col_base / 8;
```

The `decay_apply` section works on the shared tile:

```text
[CHUNK, D] = [16, 128]
```

Rows are token positions inside the current chunk. Columns are `D` feature
dimensions for the current head.

The code splits this tile into four large blocks:

```text
                 cols 0..63     cols 64..127
rows 0..7        tile_idx 0      tile_idx 1
rows 8..15       tile_idx 2      tile_idx 3
```

Each large block is `8 rows x 64 columns`.

## Warp, Group, And Lane

Each CTA has 256 threads:

```text
256 threads = 8 warps
1 warp      = 32 lanes
```

Inside one warp:

```cpp
int lane = compute_tid % 32;
int g = lane / 4;
int t = lane % 4;
```

This splits the warp into 8 groups of 4 lanes:

```text
lanes 0..3    -> g=0, t=0..3
lanes 4..7    -> g=1, t=0..3
lanes 8..11   -> g=2, t=0..3
...
lanes 28..31  -> g=7, t=0..3
```

`g` chooses one 8-column stripe inside a 64-column block:

```text
g=0 -> columns 0..7
g=1 -> columns 8..15
...
g=7 -> columns 56..63
```

The `t` value chooses 2 columns inside that 8-column stripe:

```text
t=0 -> first 2 columns
t=1 -> next 2 columns
t=2 -> next 2 columns
t=3 -> last 2 columns
```

So one 4-lane group covers:

```text
1 row x 8 columns
```

and one lane covers:

```text
1 row x 2 columns
```

## Why `row = (warp_id + g) % 8`

`m_blk` selects the current 8-row block:

```text
m_blk = 0 -> rows 0..7
m_blk = 8 -> rows 8..15
```

The expression:

```cpp
((warp_id + g) % 8)
```

selects a row offset inside that 8-row block.

For a fixed warp, as `g` increases, both the column stripe and row change.
For `warp_id = 0`:

```text
g=0 -> row0, stripe0
g=1 -> row1, stripe1
g=2 -> row2, stripe2
...
g=7 -> row7, stripe7
```

Drawn on a row-vs-stripe grid:

```text
           stripe0 stripe1 stripe2 stripe3 stripe4 stripe5 stripe6 stripe7
row0          X
row1                  X
row2                          X
row3                                  X
row4                                          X
row5                                                  X
row6                                                          X
row7                                                                  X
```

That is the "diagonal" mapping.

For `warp_id = 1`, the diagonal is shifted and wrapped:

```text
           stripe0 stripe1 stripe2 stripe3 stripe4 stripe5 stripe6 stripe7
row0                                                                  X
row1          X
row2                  X
row3                          X
row4                                  X
row5                                          X
row6                                                  X
row7                                                          X
```

Across `warp_id = 0..7`, every row/stripe cell in the `8x64` block is covered
exactly once.

Using only:

```cpp
row = m_blk + warp_id;
```

would also cover the elements, but each warp would stay on one row across all
stripes. The current mapping permutes the row assignment by group, likely to
produce a better shared-memory access pattern for these layouts.

## Why Each Thread Stores `reg_*[4][2]`

The full shared tile is:

```text
[CHUNK, D] = [16 rows, 128 columns]
```

The loop breaks it into four large tiles:

```cpp
for (int m_blk = 0; m_blk < CHUNK; m_blk += 8) {
    for (int n_blk = 0; n_blk < D; n_blk += 64) {
        ...
    }
}
```

With `CHUNK = 16` and `D = 128`, the loop values are:

```text
m_blk = 0, 8
n_blk = 0, 64
```

So the four large tiles are:

```text
tile 0: m_blk=0, n_blk=0   -> rows 0..7,  cols 0..63
tile 1: m_blk=0, n_blk=64  -> rows 0..7,  cols 64..127
tile 2: m_blk=8, n_blk=0   -> rows 8..15, cols 0..63
tile 3: m_blk=8, n_blk=64  -> rows 8..15, cols 64..127
```

Inside each large `8x64` tile, each thread handles a lane-level tile of:

```text
1 row x 2 columns = 2 values
```

That comes from:

```cpp
auto thr2_2d = make_shape(_1{}, _2{});
Tensor s_g = local_tile(tile_g, thr2_2d, make_coord(0, t));
```

Since there are four large tiles, each thread handles:

```text
2 values per large tile * 4 large tiles = 8 values total
```

That is why the register arrays are shaped like:

```cpp
float reg_g[N_TILES][2];
BF16  reg_q[N_TILES][2];
BF16  reg_k[N_TILES][2];
float reg_gt[N_TILES][2];
```

where:

```text
N_TILES = 4
```

So:

```text
reg_g[4][2]
```

means:

```text
4 large tiles
2 values owned by this thread for each large tile
```

Concrete example for one thread:

```text
warp_id = 0
g       = 3
t       = 2
```

For each large tile:

```text
tile 0: m_blk=0, n_blk=0
  row      = 0 + ((0 + 3) % 8) = 3
  col_base = 0 + 3*8 = 24
  t=2 picks cols 28,29
  -> row 3, cols 28,29

tile 1: m_blk=0, n_blk=64
  row      = 3
  col_base = 64 + 3*8 = 88
  t=2 picks cols 92,93
  -> row 3, cols 92,93

tile 2: m_blk=8, n_blk=0
  row      = 8 + ((0 + 3) % 8) = 11
  col_base = 24
  t=2 picks cols 28,29
  -> row 11, cols 28,29

tile 3: m_blk=8, n_blk=64
  row      = 11
  col_base = 88
  t=2 picks cols 92,93
  -> row 11, cols 92,93
```

So this one thread handles these 8 total values:

```text
(row 3,  cols 28,29)
(row 3,  cols 92,93)
(row 11, cols 28,29)
(row 11, cols 92,93)
```

The count matches the whole CTA:

```text
256 threads * 8 values/thread = 2048 values
16 rows * 128 columns         = 2048 values
```
