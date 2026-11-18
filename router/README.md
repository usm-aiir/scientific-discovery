# Scientific question router

ModernBERT chooses whether a question needs tables, figures, text, or a mix.
Table search is connected. Figure and text search still need to be connected.
Gemma can rewrite a question to help find tables; the original question is
still used to choose the route and find cells.

The main folders are:

- `router_src/`: train and use the router.
- `rewrite_src/`: rewrite questions with Gemma and compare the results.
- `pipeline/`: run the searches together and check that they work together.
- `data/`, `model/`, `results/`: questions, saved models, and run outputs.

## Get the model and data

In your Python environment, run these once from `router/`:

```bash
python -m pip install -r ../requirements.txt
python -m pip install -e ..
python -m pipeline.assets download
```

This gets the exact saved router and data from release `router-v1` and checks
the files. Gemma has its own download command in the [Gemma guide](rewrite_src/README.md).
Full table search also needs the separate table index, cell-search model and corpus.

`pipeline/settings.json` and `rewrite_src/settings.json` contain model paths,
device choices, and search options.
Paths inside them are relative to the settings file.

## Check table search

Check the connection to real table search using three validation questions:

```bash
python -m pipeline.check_integration
```

This needs the saved models and table index. It saves a new folder in `results/`.
It checks that the connected search matches direct table search.

See the [router guide](router_src/README.md) for training and evaluation.
