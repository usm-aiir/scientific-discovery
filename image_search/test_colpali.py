from sentence_transformers import MultiVectorEncoder
from pathlib import Path
import glob

model = MultiVectorEncoder("vidore/colqwen2.5-v0.2")

# Grab a few test images from your data
image_paths = glob.glob("/mnt/netstore1_home/behrooz.mansouri/SIGIRSciDis/25_04/figures/images/*.png")[:5]
print(f"Testing with {len(image_paths)} images:")
for p in image_paths:
    print(f"  {p}")

queries = [
    "a diagram showing neural network architecture",
    "a bar chart comparing model performance",
]

query_embeddings = model.encode_query(queries)
document_embeddings = model.encode_document(image_paths)
scores = model.similarity(query_embeddings, document_embeddings)

for i, query in enumerate(queries):
    print(f"\nQuery: '{query}'")
    for j, path in enumerate(image_paths):
        print(f"  {Path(path).name}: {scores[i][j]:.3f}")