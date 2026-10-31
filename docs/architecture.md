# Architecture

`kimi_low_memory` separates weight reads, computation, and memory management. Components can be tested independently on small inputs. Their presence does not mean that all optimizations have been combined into a validated runtime.

| Module | Purpose |
|---|---|
| `reference/reader.py` | Read tensor metadata and bounded safetensors ranges |
| `reference/safe_tokenizer.py` | Load the tokenizer through tiktoken without executing Python code from the model repository |
| `reference/model.py` | FP32 reference operations, attention, expert routing, and KV |
| `caches.py` | Cache with an explicit budget and source identity |
| `kv.py` | Latest KV generation per layer, state validation, and backing-allocation accounting |
| `prefetch.py` | Bounded prefetch of experts that have already been selected |
| `extents.py` | Combine nearby ranges while limiting extra bytes read |
| `memory.py` | Memory snapshots that distinguish allocator payload from system RSS |
| `mlx/bf16.py` | Blocked BF16 projections and experimental fused GEMV |
| `mlx/q4.py` | Q4 projections and deferred computation for a selected expert |
| `mlx/device_cache.py` | Reuse BF16 MLX arrays without retaining a second host copy |

## Computation and data order

The reference path computes attention and routing before reading the selected experts' weights. Prefetch receives expert IDs that are already known. It does not predict the next layer or change the selected set.

The order of disk reads can change independently of the order in which results are accumulated. The planner combines nearby intervals while limiting individual read size and read amplification. The resulting slices are returned to the consumer in their original logical order. This preserves FP32 accumulation order when the computation kernels remain unchanged.

MLX adapters accept a Reader with tensor metadata and a range-read method. Global memory/cache limits are configured explicitly. A library component must not silently reset those limits whenever a new Reader is created.

## KV ownership

The KV store updates layers in strict sequence. During a layer update, its old and new KV coexist; after replacement, the store releases its reference to the old buffer. It does not need to retain a second complete generation.

The store validates dtype, shapes, finite values, the current position, and preservation of the old prefix. An error after a partial update makes the store unusable for further generation. This component does not provide transactional rollback or recovery from disk. Saving a completed step and enforcing resume rules are the caller's responsibility.

## What the budgets cover

Retained payload, transient old+new storage, MLX active/cache allocations, and system resident/compressed/phys_footprint counters describe different aspects of memory use. They cannot be added without accounting for overlap. A caller-supplied callback may allocate additional temporary arrays, so the store's budget does not replace a process guard.

Apple Silicon uses unified CPU/GPU memory. A BF16 device cache does not create additional VRAM or guarantee that weights remain physically resident. The library also cannot protect against other applications' memory use; long-running inference requires a separate supervisor.
