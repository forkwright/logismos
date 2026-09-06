# test-fixtures

This dev-only crate has deterministic synthetic GGUF sources for native test
paths. Runtime crates must not depend on it.

`Qwen35FixtureConfig` and `build_qwen35_fixture` create the small F32 hybrid
fixture used by tokenizer and session tests. `raw_qwen35_fixture` exposes the
same model as mutable metadata and tensor descriptors, while
`serialize_raw_gguf` writes arbitrary GGML format tags and payload bytes. That
keeps malformed-artifact and mixed-quant fixtures on one serializer without
using a runtime quant decoder as an oracle.
