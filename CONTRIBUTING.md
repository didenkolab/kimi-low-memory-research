# Contributing

Install the project from the repository root and run the CPU checks:

```sh
python -m pip install -e '.[reference]'
python -m unittest discover -s tests -v
```

The default test run does not open model weights or use the GPU. MLX checks are enabled separately and require process memory monitoring.

For changes to computation, first add a small numerical test, then compare one real component, and only then run a complete pass. Preserve FP32 accumulation order unless changing it is the subject of the experiment.

Reports should identify the model and revision, input shape, dtype, hardware, cache state, background processes, repetition count, and measurement scope. TTFT, compute time, and inter-token intervals including I/O are different metrics. Smaller arrays or fewer read requests alone do not prove faster model execution.

Do not include weights, checkpoints, local run logs, or personal settings in pull requests. A compact table and a reproducible method are sufficient for a new result. Preserve third-party license texts and attribution.
