# Kimi Low-Memory Research

English | [Русский](README.ru.md)

Components for experimenting with Kimi-K2-Thinking on computers where model weights greatly exceed available RAM. The project combines streamed weight reads, bounded caches, sequential KV updates, and MLX computation on Apple Silicon.

Status: a research library with a reference API and separate GPU components. It does not yet provide an inference server or a validated end-to-end coding runtime. Model weights are not included.

## Installation and tests

From the repository root, with Python 3.12 or later:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[reference]'
python -m unittest discover -s tests -v
```

By default, tests use small synthetic inputs, do not download weights, and skip GPU checks. Install the optional MLX components on macOS with Apple Silicon:

```sh
python -m pip install -e '.[mlx]'
```

Import them explicitly through `kimi_low_memory.mlx`. Installing MLX does not enable GPU tests automatically; those require an external resource guard and explicit opt-in with `KIMI_MLX_TESTS=1`.

The CLI calculates KV payload budgets, reads the header of one local shard, and samples its own process memory:

```sh
kimi-low-memory kv-budget --context 512
kimi-low-memory inspect path/to/model.safetensors
kimi-low-memory memory
```

`inspect` reads a header of up to 8 MiB without loading weights. `kv-budget` calculates expanded FP32 KV storage for 61 layers; its result excludes weights and temporary arrays. `memory` uses the Mach API and works only on macOS. The CLI does not run full inference.

## Results

On an Apple M4 Pro with 24 GiB of unified memory, the research prototype produced these measurements:

| Test | Result | Comparison |
|---|---|---|
| First five matching decode positions | 65.07 → 30.50 s/token, 2.13× | CPU/GPU medians, including checkpoint I/O |
| Time to first token | Approximately 516 → 128.7 s, about 4× | Historical CPU/GPU runs |
| One complete step through 61 layers | 28.73 → 21.32 s, 1.347× | Deferred Q4 with CPU dense projections versus GPU BF16 dense projections |

The experiment used one prompt, `Write Python function add(a,b).`, with FP32 computation and Q4/BF16 weights. The GPU generated 40 tokens but reached the length limit during reasoning. There is no completed coding answer or official BF16 reference comparison. Background load and the file cache were not fully isolated; speedup factors from different experiments must not be multiplied. See the [methodology and measurements](benchmarks/README.md).

## Components

- A streaming safetensors reader and reference Kimi computation.
- Bounded caches, prefetch, and read-range planning.
- A KV store that validates shapes, positions, budgets, and update order.
- Process memory telemetry and separate MLX adapters for Q4/BF16.
- An experimental fused BF16 GEMV without an intermediate FP32 weight matrix.

The range planner passed a synthetic test: 24 reads were combined into 4 with a 1.302% increase in requested bytes, and the original data was recovered byte for byte. Fused BF16 GEMV has been checked on small matrices; its effect on full Kimi inference speed has not been measured. See [research](docs/research.md) and [architecture](docs/architecture.md).

Memory budgets cover specific arrays and buffers, not the entire process. The MLX memory limit is advisory. Real model experiments require an external watchdog and monitoring of the machine's overall load.

The project's own code is distributed under [MIT](LICENSE). The borrowed tokenizer pattern has separate Moonshot AI terms. Dependencies and primary sources are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
