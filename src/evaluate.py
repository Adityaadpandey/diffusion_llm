"""Evaluate a trained masked diffusion model on validation perplexity."""

from __future__ import annotations

import argparse
import math

import torch
import yaml
from transformers import AutoTokenizer

from data import build_dataset, get_dataloaders
from diffusion import diffusion_loss, get_schedule
from model import DiffusionTransformer
from train import get_device, load_checkpoint


def evaluate(config: dict, checkpoint_dir: str, device: torch.device) -> float:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    mask_token_id = tokenizer.convert_tokens_to_ids(config["mask_token"])
    config["mask_token_id"] = mask_token_id

    model = DiffusionTransformer(**config["model"])
    model.to(device)
    load_checkpoint(checkpoint_dir, model, device=device)
    model.eval()

    _, val_ds, _ = build_dataset(
        dataset_name=config["dataset"],
        tokenizer_name=config["tokenizer_name"],
        max_seq_len=config["max_seq_len"],
        val_split=config["val_split"],
        text_column=config["text_column"],
        num_proc=4,
        streaming=config.get("streaming", False),
        max_samples=config.get("max_samples", None),
    )
    _, val_loader = get_dataloaders(
        None,
        val_ds,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"].get("num_workers", 0),
    )

    dcfg = config["diffusion"]
    T = dcfg["T"]
    schedule_fn = get_schedule(dcfg["schedule"])

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x0 in val_loader:
            x0 = x0.to(device)
            B, L = x0.shape
            t = torch.randint(1, T + 1, (B,), device=device)
            loss, info = diffusion_loss(model, x0, t, mask_token_id, schedule_fn, T)
            # loss is already average CE per masked token, weighted. Approximate PPL
            # using number of masked tokens per batch for aggregation.
            total_loss += loss.item() * info["num_masked"] * B
            total_tokens += info["num_masked"] * B

    ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")
    return ppl


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate masked diffusion LM")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory")
    parser.add_argument("--device", default=None, help="cpu/mps")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    with open(f"{args.checkpoint}/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    ppl = evaluate(config, args.checkpoint, device)
    print(f"Validation perplexity: {ppl:.3f}")


if __name__ == "__main__":
    main()
