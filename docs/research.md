# Validated components and next experiments

The research aims to reduce unnecessary reads, conversions, and storage while keeping model computation verifiable. We distinguish mathematical equivalence, numerical agreement in FP32, and matching token IDs. None of these establishes the others.

## BF16 without a full FP32 copy

The conventional path expands a BF16 weight block to FP32 before multiplication. The experimental GEMV reads BF16 bits directly inside a Metal kernel and accumulates in FP32. Decoding was bit-exact for 65,280 finite BF16 bit patterns; nine small matrix shapes passed numerical comparisons. SIMD reduction order differs from MLX matmul, so bit-exact multiplication results are not claimed.

For a synthetic 19×7168 matrix, the additional peak in MLX active allocations was 557,132 bytes for the FP32-copy path and 76 bytes for the fused path. This is an allocator measurement with inputs already allocated. It does not measure physical memory traffic or establish a Kimi speedup. Timing from this experiment is not used to select an implementation because of background load.

The next check is one real projection, followed by a full layer with output, routing, and memory checks. Fused unpacking itself is not a new method: MLX Q4 `qmv` already unpacks data inside the kernel. See [MLX kernels](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/kernels/quantized.h) and the [custom Metal API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.metal_kernel.html).

## Combining read ranges

Data sieving reads several nearby regions as one range, then extracts the required slices. Fewer requests come at the cost of reading the gaps. This approach is established in HPC; database systems also schedule reads in physical order. See [ROMIO](https://ftp.mcs.anl.gov/pub/romio/users-guide/node5.html) and [PostgreSQL bitmap scans](https://www.postgresql.org/docs/17/indexes-bitmap-scans.html).

Our bounded planner was tested on artificial ranges: 24 requests → 4, 196,608 useful bytes → 199,168 requested bytes, with a maximum range of 58,112 bytes. All slices were recovered byte for byte in their original order. This is not yet a result for Kimi's file layout or SSD performance.

The next check is to build a plan from the metadata of eight already-selected experts, reject merges that read too many extra bytes, and compare I/O wait with the current prefetch implementation. Increasing queue depth without measuring read latency and compute utilization does not establish an improvement.

## KV: storage first, attention changes next

Expanded FP32 KV for 61 layers requires 4,013,312 bytes per position, approximately 245 MiB at a context length of 64. A separate store for keeping it in RAM has been prepared and tested synthetically, but this has not yet established a new multi-token inference speedup.

Latent MLA reduces KV storage by approximately 28.56× for the dimensions under study, but changes the operation count and FP32 rounding. One real attention computation passed a numerical comparison; a complete answer through an MLA backend has not been produced. The choice between expanded and latent representations must account for context length and measured cost, not just cache size. See [Hardware-Centric Analysis of DeepSeek’s Multi-Head Latent Attention](https://arxiv.org/html/2506.02523v1) and the [DeepSeek implementation](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py).

## Caching does not establish a speedup

A host cache of shared weights retained 5.28 GB and reduced requested read bytes by 16.02%, but the experiment under background load did not demonstrate a faster complete step. The BF16 device cache passed synthetic checks of buffer ownership, reuse, and release; its effect on real Kimi performance remains to be measured.

New multi-token configurations, longer context, and completed coding answers remain separate validation tasks. See the [results and limitations](../benchmarks/README.md).
