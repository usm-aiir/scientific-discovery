# Table retrieval

Retrieve evidence from scientific tables in two stages: find relevant tables,
then rank the cells within those tables.

## Organization

- **[experiments/](experiments/README.md)** — Setup, indexing, training, and
  evaluation for Stage 1 (tables) and Stage 2 (cells). **Start here to run the
  experiments.**
- **[live/](live/)** — The shared Stage 1 and Stage 2 retrieval implementations
  for use in the live system.

The live code is intended for the future multimodal system, where a router will
combine table, figure, and text retrieval once the other retrieval systems are
ready. The experiments use these same implementations to keep evaluation and
live retrieval aligned.
