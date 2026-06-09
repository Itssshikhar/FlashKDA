# CuTe `local_tile` Coordinate Example

This note explains the `local_tile` usage in `csrc/smxx/fwd_kernel1.cuh` around the decay-apply section:

```cpp
Tensor tile_g  = local_tile(g_tile, vec8_2d, make_coord(row, col_tile));
Tensor tile_q  = local_tile(q_tile, vec8_2d, make_coord(row, col_tile));
Tensor tile_k  = local_tile(k_tile, vec8_2d, make_coord(row, col_tile));
Tensor tile_gt = local_tile(g_total, vec8_1d, make_coord(col_tile));
```

The important point is that `make_coord(row, col_tile)` is a tile coordinate, not a raw element coordinate.

For `g_tile`, `q_tile`, and `k_tile`:

```cpp
auto vec8_2d = make_shape(_1{}, _8{});
```

means the tiler shape is:

```text
[1 row, 8 columns]
```

So:

```cpp
local_tile(g_tile, make_shape(_1{}, _8{}), make_coord(row, col_tile))
```

selects one tile of shape:

```text
[1, 8]
```

For the simple contiguous layout used here, the selected element range is:

```text
rows: row : row + 1
cols: col_tile * 8 : col_tile * 8 + 8
```

Example:

```cpp
row = 5;
col_tile = 3;
```

Then:

```text
rows: 5 : 6      -> row 5 only
cols: 24 : 32    -> columns 24..31
```

So conceptually:

```text
g_tile[5, 24..31]
```

This is why the code computes:

```cpp
int col_base = n_blk + g * 8;
int col_tile = col_base / 8;
```

`col_base` is the raw starting column, while `col_tile` is the 8-column tile index expected by `local_tile`.

If `n_blk = 0` and `g = 3`:

```text
col_base = 0 + 3 * 8 = 24
col_tile = 24 / 8 = 3
```

Passing `make_coord(row, 3)` to a `[1, 8]` tiler selects columns `24..31`.

Passing `make_coord(row, 24)` would be wrong here, because CuTe would treat `24` as the 24th 8-column tile, not raw column 24.

## Local Confirmation

This behavior was confirmed with a small CuTe host-side example. The test filled a fake tensor as:

```cpp
g_tile[row, col] = row * 1000 + col;
```

Then it ran:

```cpp
auto tile = local_tile(
    g_tile,
    make_shape(Int<1>{}, Int<8>{}),
    make_coord(5, 3)
);
```

The printed values were:

```text
tile shape: (_1,_8)

tile(0,0) = 5024
tile(0,1) = 5025
tile(0,2) = 5026
tile(0,3) = 5027
tile(0,4) = 5028
tile(0,5) = 5029
tile(0,6) = 5030
tile(0,7) = 5031
```

Since each value is `row * 1000 + col`, these values prove that the tile selected:

```text
row 5
columns 24..31
```

The next-level split also matched the lane-level logic:

```cpp
auto sub_tile = local_tile(
    tile,
    make_shape(Int<1>{}, Int<2>{}),
    make_coord(0, 2)
);
```

Printed:

```text
sub_tile shape: (_1,_2)

sub_tile(0,0) = 5028
sub_tile(0,1) = 5029
```

That means inside the 8-column tile:

```text
coord 0 -> local columns 0,1 -> global columns 24,25
coord 1 -> local columns 2,3 -> global columns 26,27
coord 2 -> local columns 4,5 -> global columns 28,29
coord 3 -> local columns 6,7 -> global columns 30,31
```

## CuTe Source Wording

CuTe implements `local_tile` by calling `inner_partition`.

`inner_partition` first applies `zipped_divide`, then slices the second mode, called the "Rest" mode, using the coordinate.

So the exact CuTe mental model is:

```text
local_tile(tensor, tiler, coord)
  -> inner_partition(tensor, tiler, coord)
  -> zipped_divide(tensor, tiler)
  -> keep the tiler mode and index the rest/tile-grid mode with coord
```

For this simple stride-1 tiler, that reduces to the intuitive arithmetic:

```text
tile coordinate * tile size = first element covered by that tile
```

## More Complex Confirmation

The same behavior was also checked with a larger logical tensor:

```text
logical tensor shape: [12, 32]
value formula:        tensor[row, col] = row * 1000 + col
outer tiler:          [3, 8]
outer coord:          (2, 3)
```

The expected logical range is:

```text
row tile 2 with tile height 3 -> rows 2*3 .. 2*3+3 = 6..8
col tile 3 with tile width 8 -> cols 3*8 .. 3*8+8 = 24..31
```

The printed result was:

```text
row-major selected tile shape: (_3,_8)
  row 0: 6024 6025 6026 6027 6028 6029 6030 6031
  row 1: 7024 7025 7026 7027 7028 7029 7030 7031
  row 2: 8024 8025 8026 8027 8028 8029 8030 8031
```

That confirms the selected logical rows and columns:

```text
rows: 6, 7, 8
cols: 24..31
```

The same test was also run with a column-major physical layout. It printed the same logical values:

```text
column-major selected tile shape: (_3,_8)
  row 0: 6024 6025 6026 6027 6028 6029 6030 6031
  row 1: 7024 7025 7026 7027 7028 7029 7030 7031
  row 2: 8024 8025 8026 8027 8028 8029 8030 8031
```

This is useful because it shows that `local_tile` selects logical tensor coordinates. The tensor layout decides how those logical coordinates map to physical memory.

Nested tiling was also checked:

```text
inner tiler: [1, 2]
inner coord: (1, 2)
```

Applied inside the previous `[3, 8]` tile, this should select:

```text
local row 1 -> global row 7
local col tile 2 with width 2 -> local cols 4,5 -> global cols 28,29
```

The printed result was:

```text
nested selected tile shape: (_1,_2)
  values: 7028 7029
```

So the larger example confirms the same rule:

```text
tile_shape decides the size of each logical tile.
tile_coord chooses which tile in the tile grid.
layout decides where that logical tile lives in physical memory.
```
