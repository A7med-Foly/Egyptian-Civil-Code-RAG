"""Export embedding model to ONNX format and run parity verification."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModel, AutoTokenizer

from legal_rag.config import settings

logger = logging.getLogger("legal_rag.indexing.export")


def export_embedding_to_onnx(
    output_path: str | Path | None = None,
    model_name: str | None = None,
) -> Path:
    """Export HuggingFace transformer model to ONNX with dynamic batch dimension."""
    model_name = model_name or settings.EMBEDDING_MODEL_NAME
    out_file = Path(output_path or (settings.BASE_DIR / "models" / "embedder.onnx"))
    out_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading PyTorch model {model_name} for ONNX export...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    dummy_text = "القانون المدني المصري العقد شريعة المتعاقدين"
    inputs = tokenizer(dummy_text, return_tensors="pt")

    input_names = ["input_ids", "attention_mask"]
    output_names = ["last_hidden_state"]
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
    }

    if "token_type_ids" in inputs:
        input_names.append("token_type_ids")
        dynamic_axes["token_type_ids"] = {0: "batch_size", 1: "sequence_length"}
        dummy_inputs = (inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"])
    else:
        dummy_inputs = (inputs["input_ids"], inputs["attention_mask"])

    logger.info(f"Exporting to ONNX: {out_file}")
    torch.onnx.export(
        model,
        dummy_inputs,
        str(out_file),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
    )
    logger.info(f"ONNX export succeeded. File size: {out_file.stat().st_size / (1024*1024):.2f} MB")
    return out_file


def verify_onnx_parity(
    onnx_path: str | Path,
    model_name: str | None = None,
    test_texts: list[str] | None = None,
    atol: float = 1e-4,
) -> tuple[bool, float, dict[str, float]]:
    """Verify numeric parity between PyTorch eager model and ONNX Runtime."""
    model_name = model_name or settings.EMBEDDING_MODEL_NAME
    texts = test_texts or [
        "العقد شريعة المتعاقدين",
        "أحكام الالتزام والمسئولية المدنية",
        "حق الملكية والحقوق العينية الأصلية",
        "Legislative provisions as regards the legal capacity of a person",
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pt_model = AutoModel.from_pretrained(model_name)
    pt_model.eval()

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # 1. PyTorch timing & output
    t0 = time.perf_counter()
    with torch.no_grad():
        pt_out = pt_model(**inputs).last_hidden_state.numpy()
    pt_latency = (time.perf_counter() - t0) * 1000

    # 2. ONNX Runtime timing & output
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_inputs = {
        k: v.numpy() for k, v in inputs.items() if k in [inp.name for inp in session.get_inputs()]
    }

    t1 = time.perf_counter()
    onnx_out = session.run(None, onnx_inputs)[0]
    onnx_latency = (time.perf_counter() - t1) * 1000

    max_diff = float(np.max(np.abs(pt_out - onnx_out)))
    is_close = bool(np.allclose(pt_out, onnx_out, atol=atol))

    benchmarks = {
        "pytorch_latency_ms": round(pt_latency, 2),
        "onnx_latency_ms": round(onnx_latency, 2),
        "max_absolute_diff": max_diff,
        "speedup": round(pt_latency / max(onnx_latency, 1e-3), 2),
    }

    logger.info(
        f"Parity check: is_close={is_close}, max_diff={max_diff:.6f}, speedup={benchmarks['speedup']}x"
    )
    return is_close, max_diff, benchmarks


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    out_file = export_embedding_to_onnx()
    is_close, diff, bench = verify_onnx_parity(out_file)
    print(f"ONNX Parity: {is_close} (max diff={diff:.6e})")
    print(f"Benchmarks: {bench}")


if __name__ == "__main__":
    main()
