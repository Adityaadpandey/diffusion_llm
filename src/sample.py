"""Generate text from a trained masked diffusion model."""

from __future__ import annotations

import argparse

import torch
import yaml
from transformers import AutoTokenizer

from diffusion import sample
from model import build_diffusion_model, detect_arch
from train import get_device, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from a masked diffusion LM")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory")
    parser.add_argument("--prompt", default="", help="Optional text prompt")
    parser.add_argument("--length", type=int, default=None, help="Sequence length")
    parser.add_argument("--steps", type=int, default=None, help="Number of reverse steps")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--mode", choices=["random", "confidence"], default=None,
                        help="Override sampling mode from config")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cpu/mps")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else get_device()
    print(f"Using device: {device}")

    ckpt_path = args.checkpoint
    with open(f"{ckpt_path}/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    mask_token_id = tokenizer.convert_tokens_to_ids(config["mask_token"])
    config["mask_token_id"] = mask_token_id

    # Override config with CLI args when provided.
    if args.length is not None:
        config["model"]["max_seq_len"] = args.length
    if args.steps is not None:
        config["diffusion"]["inference_steps"] = args.steps
    if args.temperature is not None:
        config["diffusion"]["temperature"] = args.temperature
    if args.top_k is not None:
        config["diffusion"]["top_k"] = args.top_k
    if args.mode is not None:
        config["diffusion"]["sampling_mode"] = args.mode

    arch = detect_arch(f"{ckpt_path}/model.safetensors")
    print(f"Detected architecture: {arch}")
    model = build_diffusion_model(config["model"], arch=arch)
    model.to(device)
    load_checkpoint(ckpt_path, model, device=device)
    model.eval()

    prompt_ids = None
    if args.prompt:
        prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False, return_tensors="pt")

    seq_len = config["model"]["max_seq_len"]
    dcfg = config["diffusion"]

    for i in range(args.num_samples):
        out = sample(
            model,
            seq_len=seq_len,
            mask_token_id=mask_token_id,
            T=dcfg["T"],
            inference_steps=dcfg["inference_steps"],
            schedule_name=dcfg["schedule"],
            sampling_mode=dcfg["sampling_mode"],
            temperature=dcfg["temperature"],
            top_k=dcfg["top_k"],
            device=device,
            prompt_ids=prompt_ids,
        )
        text = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
        print(f"--- Sample {i + 1} ---")
        print(text)
        print()


if __name__ == "__main__":
    main()
