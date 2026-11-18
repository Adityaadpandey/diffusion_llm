# Masked Diffusion Language Model on TinyStories

A small text generator that works differently from GPT-style models. Instead of writing one token at a time, left to right, it starts from a sequence of blanks and fills them in over several passes. It is trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) and built to run on Apple Silicon (MPS), tested on an M4 Pro with 24 GB of unified memory.

This is a learning/experiment project. It produces readable, child-story-flavoured text in small pieces, but it is not a polished model (see [Known limitations](#known-limitations)).

## How it works, in plain English

1. **Corrupt.** Take a real story chunk, e.g. `the cat sat on the mat`. Pick a noise level at random and replace that fraction of the tokens with `[MASK]`, e.g. `the [MASK] sat [MASK] the mat`.
2. **Denoise.** A bidirectional transformer (it can look left and right, like BERT) sees the masked sequence plus the noise level, and predicts what the original tokens were. The loss is cross-entropy on the masked positions only.
3. **Generate.** Start with a sequence that is entirely `[MASK]`. Run the model, commit to some of its guesses, leave the rest masked, and repeat. After `steps` rounds nothing is masked and you have a story.

Two ways to decide which positions to commit each round:

- `random`: reveal a random subset, following the masked-diffusion posterior. More varied output. This is the default for most configs.
- `confidence`: reveal the positions the model is most sure about first. Can get repetitive on weaker models.

## Setup

Needs Python 3.11 or 3.12 (`pyproject.toml` pins `>=3.11,<3.13`).

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .
python -c "import torch; print(torch.backends.mps.is_available())"   # True on Apple Silicon
```

The first training run downloads TinyStories and the tokenizer through Hugging Face (cached afterwards).

## Project layout

```
configs/    YAML configs (model size, tokenizer, schedule, training)
src/data.py        TinyStories loading, tokenizing, packing into fixed-length chunks
src/model.py       Bidirectional transformer + noise-level conditioning, detect_arch()
src/diffusion.py   Noise schedules, masking, loss, reverse-process sampler
src/train.py       Training loop, checkpointing, resume
src/sample.py      Generate text from a checkpoint
src/evaluate.py    Validation perplexity of a checkpoint
```

Checkpoint folders (`checkpoints*/`) and `*.log` files are gitignored, so they are not in the repo. You train one yourself or copy one in.

## Configs

| Config | Use it for |
|---|---|
| `configs/tinystories_smoke.yaml` | Pipeline check. Tiny model (d_model 64, 2 layers), 1000 streamed samples, 20 steps. The output is not meaningful. |
| `configs/tinystories_fast.yaml` | The run behind the results below. 41M params (d_model 512, 8 layers), bert-base-uncased tokenizer, seq len 128, T=100, 50k steps. |
| `configs/tinystories_rope_bpe.yaml` | Newer architecture: adaLN conditioning + RoPE, GPT-2 byte-level BPE tokenizer, EMA weights. Same size as `fast`. Not yet trained or evaluated here. |
| `configs/tinystories_ultimate.yaml` | Bigger and longer: d_model 768, 12 layers, 75k steps, 64 sampling steps in `confidence` mode. Has a training log in the repo directory but no results are reported here. |

There are a few more (`m4pro`, `first_run`, `big`, and some `*_test` configs used for experiments). `m4pro` is the default for `train.py`.

## Train

```bash
python src/train.py --config configs/tinystories_smoke.yaml     # quick check
python src/train.py --config configs/tinystories_fast.yaml      # real run
```

Checkpoints go to the config's `checkpoint_dir` (for example `./checkpoints_fast/step_0005000`), every `save_every` steps. Each folder holds `model.safetensors`, `config.yaml`, the tokenizer files and `training.pt`.

To keep a log, pipe it yourself:

```bash
python src/train.py --config configs/tinystories_fast.yaml 2>&1 | tee train_fast.log
```

Resume from a checkpoint folder:

```bash
python src/train.py --config configs/tinystories_fast.yaml --resume ./checkpoints_fast/step_0030000
```

## Generate

```bash
python src/sample.py --checkpoint ./checkpoints_fast/step_0050000 --num_samples 3
```

Options (anything you leave out falls back to the checkpoint's saved config):

| Flag | Meaning |
|---|---|
| `--checkpoint` | Checkpoint folder (required) |
| `--prompt "Once upon a time"` | Text to start from. Those tokens are kept fixed, the rest is generated |
| `--num_samples` | How many stories (default 5) |
| `--steps` | Number of unmasking rounds. Fewer is faster, more is slower and a bit more careful |
| `--mode random\|confidence` | How positions get revealed (see above) |
| `--temperature` | Lower is safer and blander, higher is more varied |
| `--top_k` | Only sample from the k most likely tokens at each position |
| `--length` | Sequence length override |
| `--seed` | Random seed (default 42) |
| `--device` | `cpu` or `mps` (auto-detected otherwise) |

Example:

```bash
python src/sample.py --checkpoint ./checkpoints_fast/step_0050000 \
  --prompt "Once upon a time" --steps 64 --temperature 0.9 --top_k 40 --mode random
```

The architecture (legacy or new) is detected from the checkpoint's tensor names, so you do not need to tell it which one you have.

## Evaluate

```bash
python src/evaluate.py --checkpoint ./checkpoints_fast/step_0050000
```

Prints a validation perplexity. It rebuilds the validation split from the checkpoint's config, so it needs the dataset available. See the note on what this number means below.

## Reproduce the demo

If you have the `checkpoints_fast/step_0050000` folder:

```bash
source .venv/bin/activate
uv pip install -e .
python src/sample.py --checkpoint ./checkpoints_fast/step_0050000 --num_samples 3 --mode random
```

If you do not have it, train it first with `configs/tinystories_fast.yaml` (50k steps).

## Results

Checkpoint: `checkpoints_fast/step_0050000`. Legacy architecture, 41,175,552 parameters, d_model 512, 8 layers, `bert-base-uncased` tokenizer, sequence length 128.

Validation perplexity from `train_fast.log` (evaluated every 2500 steps; these are the numbers logged at the steps below):

| Step | Val PPL |
|---|---|
| 7,500 | 9.201 |
| 15,000 | 8.579 |
| 22,500 | 8.139 |
| 30,000 | 6.554 |
| 37,500 | 4.298 |
| 45,000 | 3.793 |

A sample from the final checkpoint (trimmed with `...`; the model output is lowercased because of the tokenizer):

```
once upon a time, there was a very playful girl named lily. but every day, she
found a picture on her wall. ... " let ' s play outside! " ...
```

Read this as: the model has learned the TinyStories voice (names, simple sentences, dialogue) and local grammar. It has not learned to keep a plot going. See below.

## Known limitations

- **Drift and repetition.** Samples are fluent over a sentence or two, then wander or loop. Odd repeats such as "song song" show up. There is no long-range story planning.
- **Tokenizer artifacts.** BERT WordPiece lowercases everything and decoding leaves spacing like `let ' s` and `" hello "`. This is the main reason the `rope_bpe` config (GPT-2 BPE) exists, but that config has not been trained here, so there is no evidence yet that it fixes it.
- **Parallel steps are independent.** With 32 steps over a 128-token sequence, several tokens are committed in the same round, each sampled without seeing the others' choices. That is where inconsistencies like repeated words come from. More steps help but cost time.
- **The perplexity is not GPT perplexity.** It is `exp` of the masked-token cross-entropy, averaged over random noise levels. It measures how well the model fills in blanks, not how well it predicts the next token, so do not compare it to autoregressive PPL numbers from other models. The drop from 6.5 to 3.8 late in training also depends on the noise schedule and sampled noise levels, not only on model quality. It does not by itself say the samples got better.
- **`evaluate.py` and legacy checkpoints.** `evaluate.py` is meant to pick the architecture with `detect_arch` like `sample.py` does, but it has not been run against the legacy `step_0050000` checkpoint in this repo, and may fail on it. Treat it as unverified for legacy checkpoints.
- **Sequence length.** The legacy model was trained at 128 tokens. The RoPE config is meant to lift that cap but is untested here.
- **No benchmarks.** There are no comparisons against other models, and no human or automatic quality scores beyond the validation numbers above.
