# Dependencies and primary sources

The project's own code is distributed under [MIT](LICENSE), Copyright (c) 2026 Vadym Didenko. This document does not change the terms of third-party components.

The `PATTERN` regular expression in [safe_tokenizer.py](src/kimi_low_memory/reference/safe_tokenizer.py) comes from `tokenization_kimi.py` in the [Kimi-K2-Thinking](https://huggingface.co/moonshotai/Kimi-K2-Thinking) repository. Copyright (c) 2025 Moonshot AI and the complete [Moonshot Modified MIT License](licenses/Moonshot-Kimi-K2-Thinking-LICENSE) are preserved for this fragment. The project's own license does not replace the terms covering that fragment.

| Component | Use | Upstream |
|---|---|---|
| MLX | Metal kernels, Q4/BF16 arrays, GPU computation, and allocator telemetry | [Project and license](https://github.com/ml-explore/mlx) |
| NumPy | FP32 reference computation and test arrays | [Project and license](https://github.com/numpy/numpy) |
| tiktoken | Tokenization without executing code from the model repository | [Project and license](https://github.com/openai/tiktoken) |
| Kimi-K2-Thinking / Moonshot AI | Architecture and separately obtained weights for research | [Model card](https://huggingface.co/moonshotai/Kimi-K2-Thinking), [Modified MIT license](https://huggingface.co/moonshotai/Kimi-K2-Thinking/blob/main/LICENSE) |

Weights, tokenizer data, and third-party runtimes are not bundled with the project. Use of the model is governed by its upstream license.

The research also draws on published ideas from [DeepSeek MLA](https://github.com/deepseek-ai/DeepSeek-V3), [Anemll Flash-MoE](https://github.com/Anemll/anemll-flash-llama.cpp), [ROMIO data sieving](https://ftp.mcs.anl.gov/pub/romio/users-guide/node5.html), and [PostgreSQL read scheduling](https://www.postgresql.org/docs/17/indexes-bitmap-scans.html). Those projects did not produce our measurements, and no endorsement is claimed. We do not claim scientific novelty for the underlying methods.
