"""Cargadores de modelo. Separado de los scripts para evitar imports circulares.

El upstream (core.load_causal_model) exige exactamente una GPU CUDA y bfloat16.
Aqui estan las dos variantes que necesitamos: CPU/float32 y VL en GPU/bfloat16.
"""
from __future__ import annotations
import time
import torch
import transformers


def load_cpu(source: str):
    """Causal LM en CPU, float32. Cubre el caso qwen3_5, que necesita clase propia."""
    cfg = transformers.AutoConfig.from_pretrained(source, trust_remote_code=False)
    tok = transformers.AutoTokenizer.from_pretrained(source, trust_remote_code=False)
    cls = transformers.AutoModelForCausalLM
    if cfg.model_type in {"qwen3_5", "qwen3_5_text"}:
        native = getattr(transformers, "Qwen3_5ForCausalLM", None)
        if native is not None:
            cls, cfg = native, cfg.get_text_config()
    t0 = time.time()
    model = cls.from_pretrained(source, config=cfg, dtype=torch.float32,
                                low_cpu_mem_usage=True, trust_remote_code=False)
    model.eval()
    meta = {"source": source, "revision": "main", "dtype": "float32", "device": "cpu",
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__}
    return model, tok, meta, time.time() - t0
