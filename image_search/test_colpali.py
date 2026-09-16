from sentence_transformers import MultiVectorEncoder
from pathlib import Path
import json

model = MultiVectorEncoder("vidore/colqwen2.5-v0.2")

with open("/mnt/netstore1_home/behrooz.mansouri/SIGIRSciDis/figureGen/figure_query_output/Test.json") as f:
    test_queries = json.load(f)

def uid_to_stem(uid):
    parts = uid.split("::")
    paper = parts[0]
    fig = parts[-1]
    return f"{paper}_{fig.lstrip('F')}"

images_dir = Path("/mnt/netstore1_home/behrooz.mansouri/SIGIRSciDis/25_04/figures/images")

for entry in test_queries[:5]:
    query = entry["query"]
    target_stems = [uid_to_stem(u) for u in entry["source_figure_uids"]]

    target_paths = [str(images_dir / f"{s}.png") for s in target_stems if (images_dir / f"{s}.png").exists()]

    if not target_paths:
        print(f"\nQuery: {query[:80]}...")
        print(f"  Target images not found in 25_04: {target_stems}")
        continue

    doc_embeddings = model.encode_document(target_paths)
    scores = model.similarity(model.encode_query([query]), doc_embeddings)[0]
    best_idx = scores.argmax()
    print(f"\nQuery: {query[:80]}...")
    print(f"  Best match: {Path(target_paths[best_idx]).name}  score={scores[best_idx]:.3f}")
