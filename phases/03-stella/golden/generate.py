"""Generate golden embedding vectors + tokenizer round-trip data for Stella 1.5B v5.

Runs outside the logismos CI. Uses plain `transformers` + `tokenizers` + `safetensors`
so we do not require `sentence-transformers` (which currently trips on Stella's
`trust_remote_code` modeling file).

Pipeline (matches the Stella + sentence-transformers default for dim=1024):
    1. Tokenize with `tokenizers::Tokenizer::from_file` (matches the logismos
       Rust-side `tokenize::Tokenizer` exactly; post-processor appends EOS).
    2. Forward through `Qwen2Model` on CPU, fp32.
    3. Mean-pool the last_hidden_state with the attention mask.
    4. L2-normalise the pooled vector.
    5. Apply the 2_Dense_1024 linear head (f32).
    6. L2-normalise the 1024-dim output.

Outputs at `phases/03-stella/golden/`:
    - tokens.jsonl — per-sentence {text, ids, attention_mask}
    - embeddings_dim1024.safetensors — [N, 1024] float32, row i <=> input line i
    - PROVENANCE.json — versions, hash, notes
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import safetensors.torch
from tokenizers import Tokenizer
# Force a non-causal mask — Stella is an encoder checkpoint and the bundled
# `modeling_qwen.py` defaults to `is_causal=False`. Stock `Qwen2Model` in
# transformers >= 4.50 applies causal masking, which diverges from the Stella
# training / serving pipeline. We patch before importing the model so the
# first `Qwen2Model.forward` call uses the override.
import transformers.models.qwen2.modeling_qwen2 as _mq
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask as _prep4d


def _non_causal_mask(**kwargs):
    inputs_embeds = kwargs["inputs_embeds"]
    attention_mask = kwargs["attention_mask"]
    if attention_mask is None:
        return None
    return _prep4d(attention_mask, dtype=inputs_embeds.dtype)


_mq.create_causal_mask = _non_causal_mask

from transformers import AutoModel  # noqa: E402 — after monkey-patch


def main() -> int:
    here = Path(__file__).parent
    repo_root = here.parents[2]
    model_path = Path("/models/stella-1.5b-v5")

    inputs_path = here / "inputs.txt"
    sentences = [
        line for line in inputs_path.read_text(encoding="utf-8").splitlines() if line
    ]
    sys.stdout.write(f"loaded {len(sentences)} sentences from {inputs_path}\n")

    # --- tokenizer ------------------------------------------------------------
    tok = Tokenizer.from_file(str(model_path / "tokenizer.json"))
    encodings = [tok.encode(s) for s in sentences]

    # --- model ----------------------------------------------------------------
    model = AutoModel.from_pretrained(str(model_path), dtype=torch.float32)
    model.eval()

    # Load dense head manually (no sentence-transformers).
    dense_path = model_path / "2_Dense_1024" / "model.safetensors"
    dense_w = safetensors.torch.load_file(str(dense_path))
    dense_weight = dense_w["linear.weight"]  # [1024, 1536]
    dense_bias = dense_w["linear.bias"]      # [1024]
    assert dense_weight.shape == (1024, 1536), dense_weight.shape
    assert dense_bias.shape == (1024,), dense_bias.shape

    # --- forward --------------------------------------------------------------
    out_vecs: list[np.ndarray] = []
    token_records: list[dict] = []

    with torch.no_grad():
        for idx, (sent, enc) in enumerate(zip(sentences, encodings)):
            ids = torch.tensor([enc.ids], dtype=torch.long)
            mask = torch.tensor([enc.attention_mask], dtype=torch.long)
            out = model(input_ids=ids, attention_mask=mask)
            # last_hidden_state: [1, S, 1536]
            h = out.last_hidden_state[0]  # [S, 1536]
            mask_f = mask[0].to(torch.float32).unsqueeze(-1)  # [S, 1]
            pooled = (h * mask_f).sum(dim=0) / mask_f.sum().clamp(min=1.0)  # [1536]
            pooled = pooled / pooled.norm(p=2).clamp(min=1e-12)
            # dense head on fp32
            y = pooled.to(torch.float32) @ dense_weight.t().to(torch.float32) + dense_bias.to(torch.float32)
            y = y / y.norm(p=2).clamp(min=1e-12)
            vec = y.detach().cpu().numpy().astype(np.float32)
            out_vecs.append(vec)
            token_records.append({
                "text": sent,
                "ids": [int(x) for x in enc.ids],
                "attention_mask": [int(x) for x in enc.attention_mask],
            })
            sys.stdout.write(
                f"  [{idx:02d}] tokens={len(enc.ids):3d}  first5={vec[:5].tolist()}\n"
            )

    # --- write artefacts ------------------------------------------------------
    # safetensors wants a dict of tensors; store the matrix as one tensor.
    mat = torch.from_numpy(np.stack(out_vecs, axis=0))  # [N, 1024]
    safetensors.torch.save_file(
        {"embeddings": mat},
        str(here / "embeddings_dim1024.safetensors"),
    )

    with (here / "tokens.jsonl").open("w", encoding="utf-8") as f:
        for rec in token_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    prov = {
        "generated_at": int(time.time()),
        "model_path": str(model_path),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "tokenizers": __import__("tokenizers").__version__,
        "safetensors": __import__("safetensors").__version__,
        "num_sentences": len(sentences),
        "dim": 1024,
        "inputs_sha256": hashlib.sha256(
            inputs_path.read_bytes()
        ).hexdigest(),
    }
    (here / "PROVENANCE.json").write_text(
        json.dumps(prov, indent=2) + "\n", encoding="utf-8"
    )

    sys.stdout.write(f"wrote {mat.shape}   sha(inputs)={prov['inputs_sha256'][:12]}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
