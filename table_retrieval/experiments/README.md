# Table retrieval experiments

The experiments share the live retrieval code and keep their data and commands
beside the scripts that use them.

- [Stage 1](stage1/README.md): BM25 + BGE-M3 indexes and table retrieval evaluation.
- [Stage 2 BGE](stage2_bge/README.md): fixed train/val/test data, BGE training, and
  the Structural BGE benchmark that produced 31.53% test Recall@5.

Start with the Stage 2 README to reproduce that reported result. It includes
the original predictions, so checking the metrics requires no model download.
Model weights and generated run folders remain outside Git.
