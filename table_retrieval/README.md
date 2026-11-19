# Table retrieval

Find relevant tables with BGE + BM25 fusion, then select evidence cells with
the saved cell model. The downloaded setup uses the selected checkpoint and
reranks the table candidates before selecting five tables and five cells.

## Get the saved models

Run from the repository root in your Python environment:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m table_retrieval.assets download
```

This downloads release `table-retrieval-v1` and checks the files. The standard
BGE search model and reranker download automatically on first use.
The table corpus stays in Behrooz's directory and must match the saved index.

For a private repository, download the release attachments while signed in,
then add `--from-dir FOLDER` to the download command.

## Run a batch

The question file is a JSON list with `query_id` and `query` for each question.
Run the test questions from Behrooz's directory:

```bash
python -m table_retrieval.run_batch --input /mnt/netstore1_home/behrooz.mansouri/SciDis/tableGen/table_query_output_v2/Test.json
```

Each run saves a new folder under `results/`. `predictions.jsonl` contains the
retrieved tables and cells with their source IDs. `summary.json` records errors
and timings. This command only retrieves evidence; it does not read answer labels.

`table_retrieval/settings.json` contains the saved model paths and search options.
Add `--device cuda:0` or `--device cpu` to choose a device.

## Evaluate a batch

After a run on `Val.json` or `Test.json`, score it against the labels:

```bash
python -m table_retrieval.evaluate
```

This scores the newest run folder; add a folder path to score another one.
It saves `metrics.json` in that folder: precision, recall and nDCG at 5 for
tables, and precision, recall, F1, nDCG and MRR at 5 for cells.

See [experiments/](experiments/README.md) for training and the full experiment evaluation.
