# Kernel 2 Pipeline Handoff

This note explains the pipeline handoff in `csrc/smxx/fwd_kernel2.cuh`:

```cpp
cutlass::arch::fence_view_async_shared();
store_pipeline.producer_commit(out_write);
load_pipeline.consumer_release(load_read);
++load_read;
++out_write;
```

There are two independent pipelines in Kernel 2:

```text
1. load_pipeline
   LOAD warp produces input[stage]
   MMA warps consume input[stage]

2. store_pipeline
   MMA warps produce output[stage]
   STORE warp consumes output[stage]
```

So the MMA warps are in the middle:

```text
LOAD warp -> input buffers -> MMA warps -> output buffers -> STORE warp
```

The handoff block is the MMA side saying:

```text
I finished reading this input stage.
I finished writing this output stage.
Move my cursors to the next circular stages.
```

## Pipeline State Objects

The compute side has two small cursor objects:

```cpp
LoadPipelineState load_read;
StorePipelineState out_write;
```

Think of them as:

```text
load_read -> which input stage MMA is currently reading
out_write -> which output stage MMA is currently writing
```

They expose the current circular-buffer index:

```cpp
load_read.index()
out_write.index()
```

At the start of each chunk, compute does:

```cpp
store_pipeline.producer_acquire(out_write);
load_pipeline.consumer_wait(load_read);

int load_stage = load_read.index();
int out_stage  = out_write.index();
```

This means:

```text
store_pipeline.producer_acquire(out_write)
  MMA wants to write output[out_stage].
  Wait until the STORE warp has freed that output slot.

load_pipeline.consumer_wait(load_read)
  MMA wants to read input[load_stage].
  Wait until the LOAD warp has filled that input slot.
```

Then compute phases 1-6 use:

```cpp
shared_storage.input[load_stage]
shared_storage.output[out_stage]
```

## End-Of-Chunk Handoff

After phase 6, compute has consumed the input stage and produced the output
stage.

### Shared-Memory Fence

```cpp
cutlass::arch::fence_view_async_shared();
```

This makes shared-memory writes visible before signaling an async/TMA-side
consumer.

The important write is phase 5:

```text
MMA warps write output into shared_storage.output[out_stage].out
```

Before compute tells the STORE warp that `output[out_stage]` is ready, the
shared-memory writes must be visible in the correct async proxy domain.

Conceptually:

```text
finish shared writes before announcing output-ready
```

### Commit Output To STORE Warp

```cpp
store_pipeline.producer_commit(out_write);
```

In `store_pipeline`, MMA is the producer.

This says:

```text
MMA finished producing output[out_stage].
The STORE warp may now consume it.
```

This pairs with the STORE side:

```cpp
store_pipeline.consumer_wait(out_read);
int stage = out_read.index();
```

The STORE warp blocks at `consumer_wait` until compute calls
`producer_commit` for the matching stage.

### Release Input To LOAD Warp

```cpp
load_pipeline.consumer_release(load_read);
```

In `load_pipeline`, MMA is the consumer.

This says:

```text
MMA finished consuming input[load_stage].
The LOAD warp may now reuse/fill this input stage for a future chunk.
```

This pairs with the LOAD side:

```cpp
load_pipeline.producer_acquire(load_write);
int stage = load_write.index();
```

The LOAD warp cannot overwrite an input stage until the MMA consumer has
released it.

### Advance Circular Cursors

```cpp
++load_read;
++out_write;
```

These advance the MMA side to the next circular-buffer slots.

For example:

```text
InputStages  = 3
OutputStages = 2
```

Then:

```text
load_read.index() cycles:
  0, 1, 2, 0, 1, 2, ...

out_write.index() cycles:
  0, 1, 0, 1, ...
```

So compute uses:

```text
chunk 0 -> read input[0], write output[0]
chunk 1 -> read input[1], write output[1]
chunk 2 -> read input[2], write output[0]
chunk 3 -> read input[0], write output[1]
```

The acquire/wait/commit/release calls make this reuse safe.

## Concrete Timeline

For chunk `t = 0`, the MMA warps do:

```text
wait input[0] ready
acquire output[0] free

read input[0]
write output[0]
update recurrent state

commit output[0] ready for STORE
release input[0] free for LOAD
advance to input[1], output[1]
```

For chunk `t = 1`:

```text
wait input[1] ready
acquire output[1] free

read input[1]
write output[1]
update recurrent state

commit output[1] ready for STORE
release input[1] free for LOAD
advance to input[2], output[0]
```

When output wraps back to `output[0]`, compute cannot reuse it until:

```cpp
store_pipeline.producer_acquire(out_write);
```

confirms that the STORE warp has released it.

Similarly, the LOAD warp cannot reuse an input slot until:

```cpp
load_pipeline.consumer_release(load_read);
```

has released that slot from the MMA side.

## Ownership Summary

For `load_pipeline`:

```text
buffer: input[stage]

LOAD warp:
  producer_acquire(load_write) -> wait until stage is free
  TMA load into input[stage]
  producer_commit(load_write)  -> mark stage ready

MMA warps:
  consumer_wait(load_read)     -> wait until stage is ready
  read input[stage]
  consumer_release(load_read)  -> mark stage free
```

For `store_pipeline`:

```text
buffer: output[stage]

MMA warps:
  producer_acquire(out_write)  -> wait until stage is free
  write output[stage]
  producer_commit(out_write)   -> mark stage ready

STORE warp:
  consumer_wait(out_read)      -> wait until stage is ready
  TMA store output[stage]
  consumer_release(out_read)   -> mark stage free
```

Short version:

```text
fence_view_async_shared()        -> make output shared writes visible
store_pipeline.producer_commit() -> output buffer ready for STORE warp
load_pipeline.consumer_release() -> input buffer free for LOAD warp
++load_read                      -> next input stage for MMA
++out_write                      -> next output stage for MMA
```
