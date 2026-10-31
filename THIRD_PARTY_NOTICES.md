# Зависимости и первичные источники

Собственный код проекта распространяется под [MIT](LICENSE), Copyright (c) 2026 Vadym Didenko. Этот документ не изменяет условия сторонних компонентов.

Регулярное выражение `PATTERN` в [safe_tokenizer.py](src/kimi_low_memory/reference/safe_tokenizer.py) взято из `tokenization_kimi.py` репозитория [Kimi-K2-Thinking](https://huggingface.co/moonshotai/Kimi-K2-Thinking). Для этого фрагмента сохранены Copyright (c) 2025 Moonshot AI и полный текст [Moonshot Modified MIT License](licenses/Moonshot-Kimi-K2-Thinking-LICENSE). Лицензия собственного кода не заменяет условия этого фрагмента.

| Компонент | Использование | Upstream |
|---|---|---|
| MLX | Metal kernels, Q4/BF16 arrays, GPU-вычисления и allocator telemetry | [Проект и лицензия](https://github.com/ml-explore/mlx) |
| NumPy | FP32 reference-вычисления и тестовые массивы | [Проект и лицензия](https://github.com/numpy/numpy) |
| tiktoken | Tokenizer без исполнения кода из репозитория модели | [Проект и лицензия](https://github.com/openai/tiktoken) |
| Kimi-K2-Thinking / Moonshot AI | Архитектура и отдельно полученные веса для исследования | [Model card](https://huggingface.co/moonshotai/Kimi-K2-Thinking), [Modified MIT license](https://huggingface.co/moonshotai/Kimi-K2-Thinking/blob/main/LICENSE) |

Веса, tokenizer-данные и сторонние среды исполнения не поставляются с проектом. Условия использования модели определяются её upstream-лицензией.

Исследование также использует опубликованные идеи [DeepSeek MLA](https://github.com/deepseek-ai/DeepSeek-V3), [Anemll Flash-MoE](https://github.com/Anemll/anemll-flash-llama.cpp), [ROMIO data sieving](https://ftp.mcs.anl.gov/pub/romio/users-guide/node5.html) и [планирования чтений PostgreSQL](https://www.postgresql.org/docs/17/indexes-bitmap-scans.html). Эти проекты не являются авторами наших измерений и не заявлены как одобряющие их стороны. Научная новизна базовых методов не утверждается.
