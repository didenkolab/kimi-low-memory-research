# Results and comparison method

[results.json](results.json) summarizes measurements from the research prototype and individual synthetic components. It is not a general Kimi benchmark or a performance promise for another machine.

Model measurements used an Apple M4 Pro with 24 GiB of unified memory, one prompt, `Write Python function add(a,b).`, 21 input tokens after applying the chat template, and FP32 computation with the original Q4/BF16 weights. Context length was limited to 64. An official BF16 runtime was not used as a reference.

| Metric | Value | Limitation |
|---|---|---|
| Median of five matching CPU/GPU decode intervals | 65.0744 / 30.4986 s, ratio 2.1337× | Separate sequential runs; checkpoint I/O included |
| Approximate CPU/GPU TTFT | 516 / 128.7 s | Historical estimate without background-load isolation |
| Median of all 39 GPU intervals | 22.6538 s | No CPU reference for the remaining 34 token IDs |
| One complete replay with CPU/GPU dense projections | 28.7275 / 21.3241 s, ratio 1.3472× | Compute time for one saved step, not a new answer |

The first six CPU/GPU token IDs matched. The GPU reached 40 output tokens and stopped at the length limit during reasoning. It did not produce completed code, and functional correctness of a solution was not evaluated.

The JSON reports synthetic fused BF16 and extent-planner checks separately. Lower peak allocations or fewer requests do not automatically translate into a speedup factor. Requested-byte counters do not measure physical SSD traffic either.

The next comparison needs identical inputs, computation backend, warmup, and checkpoint policy, along with multiple run orders, numerical checks, and recorded background load. TTFT, inter-token wall time, and pure replay compute time are kept separate. Speedup factors from different experiments are not multiplied.
