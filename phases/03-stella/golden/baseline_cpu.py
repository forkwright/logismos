"""Measure a CPU-side Stella baseline throughput to serve as the 10x floor.

Not a perfect stand-in for candle-on-CPU (we no longer have candle installed
locally; kanon yanked it) but the same fp32 Qwen2 forward path on the same
hardware — a reasonable anchor for the Phase-3 gate. Provenance is recorded.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer
import transformers.models.qwen2.modeling_qwen2 as _mq
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask as _prep4d


def _non_causal_mask(**kwargs):
    inputs_embeds = kwargs["inputs_embeds"]
    attention_mask = kwargs["attention_mask"]
    if attention_mask is None:
        return None
    return _prep4d(attention_mask, dtype=inputs_embeds.dtype)


_mq.create_causal_mask = _non_causal_mask

from transformers import AutoModel  # noqa: E402
import safetensors.torch  # noqa: E402


def main() -> int:
    here = Path(__file__).parent
    model_path = Path("/models/stella-1.5b-v5")

    sentences = [
        line for line in (here / "inputs.txt").read_text(encoding="utf-8").splitlines() if line
    ]

    tok = Tokenizer.from_file(str(model_path / "tokenizer.json"))
    model = AutoModel.from_pretrained(str(model_path), dtype=torch.float32)
    model.eval()

    dense_w = safetensors.torch.load_file(str(model_path / "2_Dense_1024" / "model.safetensors"))
    weight = dense_w["linear.weight"].to(torch.float32)
    bias = dense_w["linear.bias"].to(torch.float32)

    # Warm-up on a single sentence.
    enc0 = tok.encode(sentences[0])
    ids = torch.tensor([enc0.ids], dtype=torch.long)
    mask = torch.tensor([enc0.attention_mask], dtype=torch.long)
    with torch.no_grad():
        for _ in range(2):
            _ = model(input_ids=ids, attention_mask=mask)

    # Timed loop.
    timings_ms: list[float] = []
    with torch.no_grad():
        for sent in sentences:
            enc = tok.encode(sent)
            ids = torch.tensor([enc.ids], dtype=torch.long)
            mask = torch.tensor([enc.attention_mask], dtype=torch.long)
            t0 = time.perf_counter()
            out = model(input_ids=ids, attention_mask=mask)
            h = out.last_hidden_state[0]
            mask_f = mask[0].to(torch.float32).unsqueeze(-1)
            pooled = (h * mask_f).sum(dim=0) / mask_f.sum().clamp(min=1.0)
            pooled = pooled / pooled.norm(p=2).clamp(min=1e-12)
            y = pooled @ weight.t() + bias
            y = y / y.norm(p=2).clamp(min=1e-12)
            t1 = time.perf_counter()
            timings_ms.append((t1 - t0) * 1000.0)

    total_sec = sum(timings_ms) / 1000.0
    throughput = len(sentences) / total_sec
    mean_ms = sum(timings_ms) / len(timings_ms)
    sys.stdout.write(
        f"sentences: {len(sentences)}  total: {total_sec:.2f}s  "
        f"throughput: {throughput:.3f} sent/s  mean_latency: {mean_ms:.1f} ms\n"
    )

    record = {
        "implementation": "transformers-cpu-fp32",
        "host": "menos",
        "n_sentences": len(sentences),
        "total_sec": total_sec,
        "mean_latency_ms": mean_ms,
        "throughput_sent_per_sec": throughput,
        "dim": 1024,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "note": "CPU stand-in for the candle-CPU baseline. 10x floor = 10 * throughput.",
    }
    (here / "cpu_baseline.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
