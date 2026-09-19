import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from sentence_transformers import MultiVectorEncoder
from pathlib import Path
from PIL import Image
import glob, json, csv, numpy as np, torch

model = MultiVectorEncoder("vidore/colqwen2.5-v0.2")

with open("/mnt/netstore1_home/behrooz.mansouri/SIGIRSciDis/figureGen/figure_query_output/Test.json", encoding="utf-8") as f:
    test_queries = json.load(f)

all_image_paths = glob.glob("/mnt/netstore1_home/behrooz.mansouri/SIGIRSciDis/25_04/figures/images/*.png")
print(f"Found {len(all_image_paths)} image files")

def uid_to_stem(uid):
    parts = uid.split("::")
    return f"{parts[0]}_{parts[-1].lstrip('F')}"

BATCH_SIZE = 8
all_embeddings = []
final_image_paths = []
skipped = 0

print("Encoding corpus in batches...")
for i in range(0, len(all_image_paths), BATCH_SIZE):
    batch_paths = all_image_paths[i:i + BATCH_SIZE]
    batch_pil = []
    batch_valid_paths = []

    for path in batch_paths:
        try:
            img = Image.open(path).convert('RGB')
            batch_pil.append(img)
            batch_valid_paths.append(path)
        except Exception:
            skipped += 1
            continue

    if batch_pil:
        try:
            batch_emb = model.encode_document(batch_pil)
            all_embeddings.append(np.array(batch_emb))
            final_image_paths.extend(batch_valid_paths)
        except Exception as e:
            print(f"  Batch {i//BATCH_SIZE} failed: {e}, skipping {len(batch_pil)} images")
            skipped += len(batch_pil)
        finally:
            torch.cuda.empty_cache()
        del batch_pil

    if (i // BATCH_SIZE) % 200 == 0 and i > 0:
        print(f"  Encoded {len(final_image_paths)} images so far ({skipped} skipped)...")

print(f"Done encoding. Valid: {len(final_image_paths)}, Skipped: {skipped}")
doc_embeddings = np.vstack(all_embeddings)

output_path = "/home/adah.holt/scientific-discovery/ret_result.tsv"
with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["query_id", "0", "fig_id", "rank", "relevance"])

    for i, entry in enumerate(test_queries):
        query_id = entry["query_id"]
        query = entry["query"]
        ground_truth = set(uid_to_stem(u) for u in entry["source_figure_uids"])

        scores = model.similarity(model.encode_query([query]), doc_embeddings)[0]
        ranked = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)[:100]

        for rank, idx in enumerate(ranked, start=1):
            fig_stem = Path(final_image_paths[idx]).stem
            relevance = 2 if fig_stem in ground_truth else 0
            writer.writerow([query_id, 0, fig_stem, rank, relevance])

        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(test_queries)} queries...")

print(f"Done! Results saved to {output_path}")