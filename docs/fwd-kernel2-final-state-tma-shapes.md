# Kernel 2 Final State TMA Shapes

This note explains the final-state TMA store shape logic in
`csrc/smxx/fwd_kernel2.cuh`:

```cpp
Tensor g_final = tma_store_final_state.get_tma_tensor(make_shape(N * H, D, D));
auto state_off = g_final.layout()(seq_idx * H + head_idx, 0, 0);
Tensor g_final_tile = make_tensor(
    g_final.data() + state_off,
    make_layout(make_shape(Int<1>{}, Int<D>{}, Int<D>{}),
                stride(g_final.layout()))
);
```

## What The State Tensor Means

The external `final_state` tensor has shape:

```text
[number_of_sequences, number_of_heads, head_dim, head_dim]
```

For short names, the code writes this as:

```text
[N, H, D, D]
```

where:

```text
N = number of sequences in the batch
H = number of heads
D = head dimension
```

There is one recurrent state matrix per:

```text
(sequence, head)
```

Each state matrix has shape:

```text
[D, D]
```

So:

```text
final_state[sequence, head, :, :]
```

is the final recurrent state matrix for one sequence/head pair.

## Concrete Example

Use a small example:

```text
number_of_sequences = 3
number_of_heads     = 4
head_dim            = 8
```

Then:

```text
final_state shape = [3, 4, 8, 8]
```

That means there are:

```text
3 * 4 = 12
```

state matrices total:

```text
sequence 0, head 0 -> one 8x8 matrix
sequence 0, head 1 -> one 8x8 matrix
sequence 0, head 2 -> one 8x8 matrix
sequence 0, head 3 -> one 8x8 matrix

sequence 1, head 0 -> one 8x8 matrix
...
sequence 2, head 3 -> one 8x8 matrix
```

## Flattening `(sequence, head)`

The TMA code views the first two dimensions as one flattened dimension:

```cpp
make_shape(N * H, D, D)
```

So instead of:

```text
[sequence, head, row, col]
```

TMA sees:

```text
[sequence_head, row, col]
```

where:

```text
sequence_head = sequence * number_of_heads + head
```

In the example with `number_of_heads = 4`:

```text
seq0 head0 -> 0*4 + 0 = 0
seq0 head1 -> 0*4 + 1 = 1
seq0 head2 -> 0*4 + 2 = 2
seq0 head3 -> 0*4 + 3 = 3

seq1 head0 -> 1*4 + 0 = 4
seq1 head1 -> 1*4 + 1 = 5
seq1 head2 -> 1*4 + 2 = 6
seq1 head3 -> 1*4 + 3 = 7

seq2 head0 -> 2*4 + 0 = 8
seq2 head1 -> 2*4 + 1 = 9
seq2 head2 -> 2*4 + 2 = 10
seq2 head3 -> 2*4 + 3 = 11
```

So:

```text
final_state[3, 4, 8, 8]
```

is viewed as:

```text
g_final[12, 8, 8]
```

It is the same memory. Only the view changes.

## Which Matrix This CTA Owns

Each Kernel 2 CTA handles one sequence and one head:

```text
seq_idx  = blockIdx.x
head_idx = blockIdx.y
```

Suppose:

```text
seq_idx  = 2
head_idx = 1
```

With `number_of_heads = 4`, the flattened sequence/head index is:

```text
seq_idx * H + head_idx = 2 * 4 + 1 = 9
```

So this CTA owns:

```text
g_final[9, :, :]
```

which is the same memory as:

```text
final_state[2, 1, :, :]
```

## `state_off`

This line:

```cpp
auto state_off = g_final.layout()(seq_idx * H + head_idx, 0, 0);
```

asks CuTe for the raw memory offset of:

```text
g_final[sequence_head, 0, 0]
```

Using the example:

```text
sequence_head = 9
```

so `state_off` points to:

```text
g_final[9, 0, 0]
```

which is:

```text
final_state[2, 1, 0, 0]
```

That is the start of the `8x8` state matrix for sequence `2`, head `1`.

## Why The Tile Shape Is `[1, D, D]`

This code creates the global destination tile:

```cpp
Tensor g_final_tile = make_tensor(
    g_final.data() + state_off,
    make_layout(make_shape(Int<1>{}, Int<D>{}, Int<D>{}),
                stride(g_final.layout()))
);
```

The global TMA tensor has rank 3:

```text
[sequence_head, row, col]
```

This CTA stores exactly one `sequence_head` slice:

```text
g_final[sequence_head : sequence_head + 1, :, :]
```

So the tile remains rank 3:

```text
[1, D, D]
```

The leading `1` means:

```text
one sequence/head state matrix
```

In the example:

```text
g_final_tile shape = [1, 8, 8]
```

Then:

```text
g_final_tile[0, row, col]
```

maps to:

```text
final_state[2, 1, row, col]
```

## Shared-Memory Source Tile

The shared-memory state is this CTA's local recurrent state:

```cpp
shared_storage.state_acc
```

Logically it is one matrix:

```text
[D, D]
```

For TMA, the shared-memory layout presents it with the same leading size-1
dimension as the global tile:

```cpp
Tensor s_state = make_tensor(
    make_smem_ptr(shared_storage.state_acc.begin()),
    TMAStateSmemLayout{}
);
```

So the shared source is also viewed as:

```text
[1, D, D]
```

The leading `1` again means:

```text
one state matrix
```

## Final Copy Meaning

The TMA store:

```cpp
cute::copy(
    tma_store_final_state,
    cta_tma_store_state.partition_S(s_state),
    cta_tma_store_state.partition_D(g_final_tile)
);
```

copies:

```text
s_state[0, :, :]
```

to:

```text
g_final_tile[0, :, :]
```

which is:

```text
final_state[seq_idx, head_idx, :, :]
```

For the concrete example:

```text
shared CTA state[8, 8]
  -> final_state[2, 1, :, :]
```

Short version:

```text
original user tensor:
  [sequence, head, row, col]

TMA global view:
  [sequence_head, row, col]

this CTA's global tile:
  [1, row, col]

shared CTA state:
  [1, row, col]
```
