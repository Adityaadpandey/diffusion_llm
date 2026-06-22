# Masked Discrete Diffusion Language Model

A working masked discrete diffusion LM trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), optimized for Apple Silicon (M4 Pro / 24 GB unified RAM).

The model is a **BERT-style bidirectional transformer** trained to denoise text that has been corrupted by randomly masking tokens. At inference time, generation starts from a fully `[MASK]`-ed sequence and progressively unmasks tokens using model confidence.

---

## Setup

Requires Python 3.11 or 3.12 (the system Python 3.14 is too new for stable ML wheels).

```bash
# Install dependencies (uv is used for speed)
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .
```

Verify MPS is available:

```python
import torch
print(torch.backends.mps.is_available())  # True on Apple Silicon
```

---

## Project layout

```
configs/tinystories_m4pro.yaml  # Hyperparameters
src/data.py                     # TinyStories loader + packing
src/model.py                    # Timestep-conditioned bidirectional transformer
src/diffusion.py                # Schedules, ELBO loss, sampling
src/train.py                    # Training loop
src/sample.py                   # Text generation
src/evaluate.py                 # Validation perplexity
```

---

## Train

```bash
python src/train.py --config configs/tinystories_m4pro.yaml
```

The first run downloads TinyStories and the BERT tokenizer (~2 GB dataset, cached by HuggingFace). Training will use MPS automatically if available.

A more practical first-run config (`configs/tinystories_first_run.yaml`) trains a ~22 M parameter model for 50k steps:

```bash
python src/train.py --config configs/tinystories_first_run.yaml
```

Monitor progress:

```bash
tail -f train.log
```

Resume from a checkpoint:

```bash
python src/train.py --config configs/tinystories_first_run.yaml --resume ./checkpoints/step_0005000
```

---

## Generate text

```bash
python src/sample.py \
  --checkpoint ./checkpoints/step_0100000 \
  --prompt "Once upon a time" \
  --num_samples 5
```

You can override sampling hyperparameters on the fly:

```bash
python src/sample.py \
  --checkpoint ./checkpoints/step_0100000 \
  --prompt "A little rabbit" \
  --steps 64 \
  --temperature 0.9 \
  --top_k 40 \
  --mode random \
  --num_samples 3
```

**Sampling mode:** `random` (default) uses the analytic masked-diffusion posterior and gives more diverse, story-like output. `confidence` unmasks the most confident tokens first; it can collapse to repetitive tokens on undertrained models, so `random` is recommended especially early in training.

---

## Evaluate

```bash
python src/evaluate.py --checkpoint ./checkpoints/step_0100000
```

---

## Key design choices

- **Bidirectional attention** — the reverse process must see all current tokens to predict any position.
- **Timestep conditioning** — sinusoidal timestep embeddings added to token embeddings.
- **Cosine corruption schedule** — spends more steps near the clean end where fine-grained predictions matter.
- **ELBO loss** — cross-entropy only on masked positions, averaged per masked token.
- **Random posterior sampling** — at inference, use the analytic masked-diffusion posterior for diverse, stable generation.

---

## Hardware tuning

The default config (`d_model=384`, 6 layers, seq_len 128) is ~30 M parameters and fits comfortably in 24 GB unified RAM on an M4 Pro.

To scale up:
- Increase `model.d_model` / `model.num_layers`
- Increase `max_seq_len` to 256 or 512
- Reduce `training.batch_size` and increase `training.gradient_accumulation_steps` if you hit memory limits

To train faster:
- Lower `diffusion.inference_steps` for quick samples
- Use `streaming: true` and `max_samples` for quick experiments
- Lower `training.num_workers` to 0 if you see MPS/dataloader instability

---

## Expected results

After a small proof-of-concept run (~50k steps, BERT-small), samples should be grammatical English. After 100k–200k steps with the default config, generated TinyStories-style stories should be coherent.

Target validation perplexity:
- < 30 after first small run
- < 15 after longer training on the default config

Exact numbers depend heavily on training time and model size.
