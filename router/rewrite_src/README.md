# Rewrite questions with Gemma

Gemma rewrites a question to help find tables. The router and cell search
still use the original question. If a rewrite fails its checks, search uses
the original question too.

## Prepare the model

Run from `router/` after the [setup](../README.md#get-the-model-and-data):

```bash
python -m rewrite_src.rewrite --download
```

This downloads about 16 GB into `model/gemma-4-E4B-it/` without using a GPU.
It reuses files already downloaded. `rewrite_src/settings.json` contains the
model path, GPU choice and search settings.

## Check whether rewriting helps

Use an idle 48 GB GPU for Gemma and table search together.
The full model runs in 16-bit precision.

Try three validation questions first, then run the full comparison:

```bash
python -m rewrite_src.evaluate --limit 3
python -m rewrite_src.evaluate
```

To use physical GPU 1, put `CUDA_VISIBLE_DEVICES=1` before either command.

Both commands compare the original and rewritten searches on the same questions.
Only validation data is used. Three questions are enough to check the connection,
but use the full comparison to judge search quality.

Each run creates a new `results/gemma_rewriting_val_*/` folder. Look at
`metrics.json` for scores, `traces.jsonl` for individual rewrites and evidence,
and `summary.json` for completion or errors. The folder also saves the questions
and settings used.

See the [main README](../README.md) for setup.
