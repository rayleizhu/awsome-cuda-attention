# awsome-cuda-attention
Collection of effcient implementations of attention with CUDA.

## Flash Attention

These two papers describe how to implement **Exact attention** while avoid O(n^2) memory occupied by intermediate attenion matrix (i.e. QK^T).

* [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
* [Self-attention Does Not Need O(n2) Memory](https://arxiv.org/abs/2112.05682)

Implementations can be found at:

* [Official implementation by FlashAttention authors](https://github.com/HazyResearch/flash-attention)
* [Cutlass official example](https://github.com/NVIDIA/cutlass/tree/master/examples/42_fused_multi_head_attention)
* [OpenAI Triton project](https://triton-lang.org/master/getting-started/tutorials/06-fused-attention.html)

## Block Sparse Attention

Sparse GEMMs are not friendly with GPUs due to poor spatial-temporal locality. But stuctured block sparsity solves this problem. See [OpenAI's blog](https://openai.com/blog/block-sparse-gpu-kernels/) for details. 

* [OpenAI's implementation with manual PTX optimization](https://github.com/openai/blocksparse)
* [HuggingFace's implementation with Cutlass](https://github.com/huggingface/pytorch_block_sparse)
* [TileSparse](https://github.com/YulhwaKim/cutlass_tilesparse)
* [OpenAI triton project](https://github.com/openai/triton/tree/master/python/triton/ops/blocksparse)

## Reference

* [Cutlass: CUDA Templates for Linear Algebra Subroutines](https://github.com/NVIDIA/cutlass)
* [Triton: Open-Source GPU Programming for Neural Networks](https://openai.com/blog/triton/)
