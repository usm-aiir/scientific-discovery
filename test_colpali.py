from sentence_transformers import MultiVectorEncoder
from pathlib import Path
import glob, json

model = MultiVectorEncoder("vidore/colqwen2.5-v0.2")

all_images = glob.glob("/mnt/netstore1_home/behrooz.mansouri/SIGIRSciDis/25_04/figures/images/*.png")[:5]
print(f"Corpus: {len(all_images)} images")

doc_embeddings = model.encode_document(all_images)

with open("/mnt/netstore1_home/behrooz.mansouri/SIGIRSciDis/figureGen/figure_query_output/Test.json") as f:
    test_queries = json.load(f)

def uid_to_stem(uid):
    paper, fig = uid.split("::")
    return f"{paper}_{fig.lstrip('F')}"

for entry in test_queries[:5]:
    scores = model.similarity(model.encode_query([entry["query"]]), doc_embeddings)[0]
    best_idx = scores.argmax()
    print(f"\nQuery: {entry['query'][:80]}...")
    print(f"  Best match: {Path(all_images[best_idx]).name}  score={scores[best_idx]:.3f}")
