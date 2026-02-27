#!/usr/bin/env python3
"""Download all GGUF models for Idea C quantization experiments."""

from huggingface_hub import hf_hub_download
from pathlib import Path
import sys

MODELS_DIR = Path(__file__).parent / "models"

# (repo_id, filename, local_dir, local_filename)
MODELS = [
    # === Qwen2.5-3B-Instruct (all 7 levels) ===
    ("bartowski/Qwen2.5-3B-Instruct-GGUF", "Qwen2.5-3B-Instruct-f16.gguf",       "qwen2.5-3b", "qwen2.5-3b-instruct-f16.gguf"),
    ("bartowski/Qwen2.5-3B-Instruct-GGUF", "Qwen2.5-3B-Instruct-Q8_0.gguf",      "qwen2.5-3b", "qwen2.5-3b-instruct-q8_0.gguf"),
    ("bartowski/Qwen2.5-3B-Instruct-GGUF", "Qwen2.5-3B-Instruct-Q6_K.gguf",      "qwen2.5-3b", "qwen2.5-3b-instruct-q6_k.gguf"),
    ("bartowski/Qwen2.5-3B-Instruct-GGUF", "Qwen2.5-3B-Instruct-Q5_K_M.gguf",    "qwen2.5-3b", "qwen2.5-3b-instruct-q5_k_m.gguf"),
    ("bartowski/Qwen2.5-3B-Instruct-GGUF", "Qwen2.5-3B-Instruct-Q4_K_M.gguf",    "qwen2.5-3b", "qwen2.5-3b-instruct-q4_k_m.gguf"),
    ("bartowski/Qwen2.5-3B-Instruct-GGUF", "Qwen2.5-3B-Instruct-Q3_K_M.gguf",    "qwen2.5-3b", "qwen2.5-3b-instruct-q3_k_m.gguf"),
    ("bartowski/Qwen2.5-3B-Instruct-GGUF", "Qwen2.5-3B-Instruct-Q2_K.gguf",      "qwen2.5-3b", "qwen2.5-3b-instruct-q2_k.gguf"),

    # === Llama-3.2-3B-Instruct (5 levels; Q3_K_L substitutes Q3_K_M, no Q2_K available) ===
    ("bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-f16.gguf",    "llama-3.2-3b", "llama-3.2-3b-instruct-f16.gguf"),
    ("bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q8_0.gguf",   "llama-3.2-3b", "llama-3.2-3b-instruct-q8_0.gguf"),
    ("bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q6_K.gguf",   "llama-3.2-3b", "llama-3.2-3b-instruct-q6_k.gguf"),
    ("bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q5_K_M.gguf", "llama-3.2-3b", "llama-3.2-3b-instruct-q5_k_m.gguf"),
    ("bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q4_K_M.gguf", "llama-3.2-3b", "llama-3.2-3b-instruct-q4_k_m.gguf"),
    ("bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q3_K_L.gguf", "llama-3.2-3b", "llama-3.2-3b-instruct-q3_k_l.gguf"),  # Q3_K_L (no Q3_K_M)

    # === Phi-3.5-mini-instruct (6 levels; no F16 available, Q8_0 is top baseline) ===
    ("bartowski/Phi-3.5-mini-instruct-GGUF", "Phi-3.5-mini-instruct-Q8_0.gguf",   "phi-3.5-mini", "phi-3.5-mini-instruct-q8_0.gguf"),
    ("bartowski/Phi-3.5-mini-instruct-GGUF", "Phi-3.5-mini-instruct-Q6_K.gguf",   "phi-3.5-mini", "phi-3.5-mini-instruct-q6_k.gguf"),
    ("bartowski/Phi-3.5-mini-instruct-GGUF", "Phi-3.5-mini-instruct-Q5_K_M.gguf", "phi-3.5-mini", "phi-3.5-mini-instruct-q5_k_m.gguf"),
    ("bartowski/Phi-3.5-mini-instruct-GGUF", "Phi-3.5-mini-instruct-Q4_K_M.gguf", "phi-3.5-mini", "phi-3.5-mini-instruct-q4_k_m.gguf"),
    ("bartowski/Phi-3.5-mini-instruct-GGUF", "Phi-3.5-mini-instruct-Q3_K_M.gguf", "phi-3.5-mini", "phi-3.5-mini-instruct-q3_k_m.gguf"),
    ("bartowski/Phi-3.5-mini-instruct-GGUF", "Phi-3.5-mini-instruct-Q2_K.gguf",   "phi-3.5-mini", "phi-3.5-mini-instruct-q2_k.gguf"),

    # === Mistral-7B-Instruct-v0.3 (6 levels; no F16, capped at Q5_K_M per CLAUDE.md but downloading all available) ===
    ("bartowski/Mistral-7B-Instruct-v0.3-GGUF", "Mistral-7B-Instruct-v0.3-Q8_0.gguf",   "mistral-7b", "mistral-7b-instruct-v0.3-q8_0.gguf"),
    ("bartowski/Mistral-7B-Instruct-v0.3-GGUF", "Mistral-7B-Instruct-v0.3-Q6_K.gguf",   "mistral-7b", "mistral-7b-instruct-v0.3-q6_k.gguf"),
    ("bartowski/Mistral-7B-Instruct-v0.3-GGUF", "Mistral-7B-Instruct-v0.3-Q5_K_M.gguf", "mistral-7b", "mistral-7b-instruct-v0.3-q5_k_m.gguf"),
    ("bartowski/Mistral-7B-Instruct-v0.3-GGUF", "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf", "mistral-7b", "mistral-7b-instruct-v0.3-q4_k_m.gguf"),
    ("bartowski/Mistral-7B-Instruct-v0.3-GGUF", "Mistral-7B-Instruct-v0.3-Q3_K_M.gguf", "mistral-7b", "mistral-7b-instruct-v0.3-q3_k_m.gguf"),
    ("bartowski/Mistral-7B-Instruct-v0.3-GGUF", "Mistral-7B-Instruct-v0.3-Q2_K.gguf",   "mistral-7b", "mistral-7b-instruct-v0.3-q2_k.gguf"),
]


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    skipped = 0
    downloaded = 0
    failed = 0

    for repo_id, filename, local_dir, local_name in MODELS:
        dest_dir = MODELS_DIR / local_dir
        dest_file = dest_dir / local_name
        dest_dir.mkdir(exist_ok=True)

        if dest_file.exists():
            size_mb = dest_file.stat().st_size / (1024 * 1024)
            print(f"  SKIP  {local_dir}/{local_name} ({size_mb:.0f} MB)")
            skipped += 1
            continue

        print(f"  DOWN  {repo_id} -> {local_dir}/{local_name}")
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=dest_dir,
                local_dir_use_symlinks=False,
            )
            # rename if hf_hub_download used original filename
            downloaded_path = Path(path)
            if downloaded_path.name != local_name:
                downloaded_path.rename(dest_file)
            downloaded += 1
            size_mb = dest_file.stat().st_size / (1024 * 1024)
            print(f"    OK  {size_mb:.0f} MB")
        except Exception as e:
            print(f"  FAIL  {e}", file=sys.stderr)
            failed += 1

    print(f"\nDone: {downloaded} downloaded, {skipped} skipped, {failed} failed")
    print(f"Total models: {len(MODELS)}")


if __name__ == "__main__":
    main()
