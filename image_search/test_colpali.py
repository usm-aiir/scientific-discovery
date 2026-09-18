from sentence_transformers import MultiVectorEncoder
from pathlib import Path
import glob, json, csv

model = MultiVectorEncoder("vidore/colqwen2.5-v0.2")

with open("/mnt/netstore1_home/behrooz.mansouri/SIGIRSciDis/figureGen/figure_query_output/Test.json", encoding="utf-8") as f:
    test_queries = json.load(f)

all_images = glob.glob("/mnt/netstore1_home/behrooz.mansouri/SIGIRSciDis/25_04/figures/images/*.png")
print(f"Corpus: {len(all_images)} images")

def uid_to_stem(uid):
    parts = uid.split("::")
    return f"{parts[0]}_{parts[-1].lstrip('F')}"

print("Encoding corpus...")
doc_embeddings = model.encode_document(all_images)
print("Done encoding.")

output_path = "/home/adah.holt/scientific-discovery/ret_result.tsv"
with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["query_id", "0", "fig_id", "rank", "relevance"])

    for i, entry in enumerate(test_queries):
        query_id = entry["query_id"]
        query = entry["query"]
        ground_truth = set(uid_to_stem(u) for u in entry["source_figure_uids"])

        scores = model.similarity(model.encode_query([query]), doc_embeddings)[0]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:100]

        for rank, idx in enumerate(ranked, start=1):
            fig_stem = Path(all_images[idx]).stem
            relevance = 2 if fig_stem in ground_truth else 0
            writer.writerow([query_id, 0, fig_stem, rank, relevance])

        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(test_queries)} queries...")

print(f"Done! Results saved to {output_path}")
