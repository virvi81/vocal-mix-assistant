"""One-shot offline worker: in-memory samples on stdin, description JSON on stdout.

Launched by app.py under macOS sandbox-exec with network access denied.
No audio paths, downloads, saved results, or diagnostic audio dumps.
"""
from __future__ import annotations

import contextlib
from io import BytesIO
import json
from math import gcd
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models/MOSS-Music-8B-Thinking-8bit"
MAX_INPUT_BYTES = 32 * 1024 * 1024
MAX_GENERATION_TOKENS = 8192


def no_network(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto", "urllib.Request"}:
        raise RuntimeError("Network is disabled in the MOSS worker")


def main():
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
                 "HF_HUB_DISABLE_TELEMETRY", "HF_HUB_DISABLE_IMPLICIT_TOKEN"):
        os.environ[name] = "1"
    os.environ["HF_HOME"] = str(ROOT / "cache/huggingface")
    sys.addaudithook(no_network)
    sys.path.insert(0, str(ROOT.parent))
    from moss_memory import memory_context
    from moss_descriptions import description_prompt
    # All library chatter goes to stderr, never into the JSON protocol.
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        from scipy.signal import resample_poly

        header_line = sys.stdin.buffer.readline(64001)
        if len(header_line) > 64000 or not header_line.endswith(b"\n"):
            raise ValueError("Invalid local request header")
        header = json.loads(header_line)
        prompt = description_prompt(header.get("description_task", "general"),
                                    measurement_context=header.get("measurement_context"))
        context = memory_context(header.get("memory_notes", []))
        sample_rate = header["sample_rate"]
        if type(sample_rate) is not int or not 8000 <= sample_rate <= 384000:
            raise ValueError("Invalid sample rate")
        payload = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
        if len(payload) > MAX_INPUT_BYTES:
            raise ValueError("Capture exceeds the memory input limit")
        audio = np.load(BytesIO(payload), allow_pickle=False)
        if (not isinstance(audio, np.ndarray) or audio.ndim != 2
                or audio.shape[0] not in (1, 2) or audio.dtype.kind != "f"
                or not 0 < audio.shape[1] / sample_rate <= 30
                or not np.isfinite(audio).all()):
            raise ValueError("Invalid capture samples")
        if not (MODEL / "config.json").is_file() or not any(MODEL.glob("*.safetensors")):
            raise RuntimeError("Local model is missing; automatic downloads are disabled")
        import mlx.core as mx
        from mlx_audio.stt import load

        if not mx.metal.is_available():
            raise RuntimeError("Metal is unavailable")
        model = load(str(MODEL))
        mono = np.mean(audio, axis=0, dtype=np.float32)
        common = gcd(sample_rate, model.sample_rate)
        if sample_rate != model.sample_rate:
            mono = resample_poly(mono, model.sample_rate // common, sample_rate // common).astype(np.float32)
        if header.get("description_task") == "measurements":
            from moss_linked import generate_linked_rows, LinkedGenerationError, block_numeric_output_tokens
            with block_numeric_output_tokens(model) as blocked_count:
                try:
                    response = generate_linked_rows(model.generate, mono,
                                                   header["measurement_context"], context)
                except LinkedGenerationError as exc:
                    response = {"description": json.dumps({"generation_error": str(exc)}, ensure_ascii=False),
                                "truncated": False, "generation_strategy": "fixed_row_text_v2",
                                "row_generation": exc.attempts}
                response["blocked_numeric_token_count"] = blocked_count
        else:
            output = model.generate(mono, prompt=prompt + context, temperature=0.0,
                                    max_tokens=MAX_GENERATION_TOKENS, strip_thinking=True)
            response = {"description": output.text,
                        "truncated": output.generation_tokens >= MAX_GENERATION_TOKENS}
    measurement_context = header.get("measurement_context")
    if measurement_context is not None:
        from moss_measurements import context_fingerprint
        response.update(analysis_id=measurement_context["analysis_id"],
                        measurement_fingerprint=context_fingerprint(measurement_context))
    print(json.dumps(response, ensure_ascii=False))


if __name__ == "__main__":
    main()
