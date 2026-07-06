# Glossary

Domain terms used throughout the logismos codebase.

| Term | Definition |
|------|-----------|
| GDN | Gated Delta Net. Linear attention variant with gated delta rule. Phase 6a. |
| WMMA | Warp Matrix Multiply Accumulate. HIP intrinsic for matrix ops on wave32. |
| wave32 | RDNA3/CDNA3 default warp size (32 threads). Contrast with wave64 (GCN/CDNA2). |
| gfx1100 | AMD W7900 GPU shader ISA target. RDNA3, wave32. |
| IS_TF32_SUPPORTED | False on gfx1100. IEEE fp32 mandatory. No TensorFloat-32 fallback. |
| FWHT | Fast Walsh-Hadamard Transform. Used in TurboQuant 3-bit KV quantization. |
| TurboQuant | 3-bit KV cache quantization scheme (FWHT + Lloyd-Max codebook). Phase 6b. |
| Lloyd-Max | Optimal scalar quantizer for a given distribution. Used for codebook design. |
| ModernBERT | Transformer encoder architecture (Alibaba-NLP GTE-reranker-modernbert-base). |
| GTE | General Text Embeddings. Alibaba-NLP reranker model family. |
| TEI | Text Embeddings Inference. Reference backend API for the Reranker contract. |
| hipBLASLt | Lightweight BLAS for HIP with epilogue fusion. Primary GEMM backend. |
| preflight | CPU-side headless validation of algorithm and API shape before HIP port. |
| blast zone | Set of repo paths a dispatch agent is allowed to modify. |
