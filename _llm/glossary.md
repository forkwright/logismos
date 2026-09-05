# Glossary

Domain terms used throughout the logismos codebase.

| Term | Definition |
|------|-----------|
| GDN | Gated Delta Net. Linear attention variant with gated delta rule. Phase 6a. |
| WMMA | Warp Matrix Multiply Accumulate. HIP intrinsic for matrix ops on wave32. |
| wave32 | A wavefront of 32 lanes; the execution shape used by Logismos's gfx1100 kernels. |
| gfx1100 | AMD RDNA3 ISA target shared by W7900 and RX 7900 XTX. Device identity and memory capacity are separate facts. |
| IS_TF32_SUPPORTED | False on gfx1100. IEEE fp32 mandatory. No TensorFloat-32 fallback. |
| FWHT | Fast Walsh-Hadamard Transform. Used in TurboQuant 3-bit KV quantization. |
| TurboQuant | 3-bit KV cache quantization scheme (FWHT + Lloyd-Max codebook). Phase 6b. |
| Lloyd-Max | Optimal scalar quantizer for a given distribution. Used for codebook design. |
| ModernBERT | Transformer encoder architecture (Alibaba-NLP GTE-reranker-modernbert-base). |
| GTE | General Text Embeddings. Alibaba-NLP reranker model family. |
| TEI | Text Embeddings Inference. Reference backend API for the Reranker contract. |
| hipBLASLt | AMD BLAS library studied as prior art; not linked or used as Logismos's GEMM backend. |
| preflight | CPU-side headless validation of algorithm and API shape before HIP port. |
| blast zone | Set of repo paths a dispatch agent is allowed to modify. |
