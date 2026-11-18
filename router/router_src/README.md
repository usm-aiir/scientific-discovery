# ModernBERT router

ModernBERT reads a question and chooses tables, figures, text, or a mix.
Only the question enters the model.

## Train

Run from `router/` after the [setup](../README.md#get-the-model-and-data):

```bash
python -m router_src.train --check-data
python -m router_src.train
```

The first command checks the data without training. The second trains for
three epochs using `data/Train.json` and saves the model that scores best on
`data/Val.json`. Each run creates new folders in `model/` and `results/`.

## Check a saved model

```bash
python -m router_src.evaluate
```

This checks the newest finished model against Val. To choose a different model,
add `--checkpoint PATH`. Scores and predictions go into a new `results/val_*/` folder.

The scores tell you:

- How often the chosen route matches a known acceptable route.
- How often it includes a known sufficient set of sources.
- How many source types it selects on average.

These scores check routing choices. They do not measure search or answer quality.
Use Train for training, Val for choosing settings, and Test for the final evaluation.

The saved model used earlier training inputs. The current training command
does not repeat its paired-question training method. Use `data/frozen/` when
comparing earlier scores. See the [data notes](../data/README.md) for details.

See the [main README](../README.md) for setup, and
[AGENTS.md](../AGENTS.md) for development rules and experiment details.
