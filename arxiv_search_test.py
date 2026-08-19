import os 
import torch
import pickle #saving/loading embeddings
from llm2vec import LLM2Vec #model
import numpy as np #for the math at the end

#Gets da stuff
CAPTIONS_DIR = os.path.expanduser("~/arxiv_data/captions")
EMBEDDINGS_CACHE = os.path.expanduser("~/arxiv_data/embeddings.pkl")

#reads every tsv in the captions file
def load_captions(captions_dir):
    records = []
    for fname in os.listdir(captions_dir):
        fpath = os.path.join(captions_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    paper_id = parts[0]
                    fig_num = parts[1]
                    caption = parts[-1].strip('"')
                    records.append((paper_id, fig_num, caption))
    return records

print("Loading captions...")
records = load_captions(CAPTIONS_DIR)
print(f"Loaded {len(records)} captions")

#Loads llama in 2 layers because of size, bfloat16 cuts memory in half
print("Loading LLM2Vec model...")
l2v = LLM2Vec.from_pretrained(
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
)

#The cache saves the result as a .pkl file so every run after the first skips that step entirely and loads in seconds.
if os.path.exists(EMBEDDINGS_CACHE):
    print("Loading cached embeddings...")
    with open(EMBEDDINGS_CACHE, "rb") as f:
        embeddings = pickle.load(f)
else:
    print("Encoding captions...")
    captions = [r[2] for r in records]
    embeddings = l2v.encode(captions, batch_size=32, show_progress_bar=True)
    with open(EMBEDDINGS_CACHE, "wb") as f:
        pickle.dump(embeddings, f)
    print("Embeddings saved to cache!")

#Query instruction, llm2vec uses instruction prefix for queries
instruction = "Given a scientific figure search query, retrieve the most relevant figure caption:"
query = input("\nEnter your search query: ")

#Cosine stuff, the result is a score between 0 and 1 for every caption and the higher means more similarity to the queery
q_emb = l2v.encode([[instruction, query]])
q_np = q_emb.cpu().float().numpy()
if hasattr(embeddings, 'numpy'):
    emb_np = embeddings.cpu().float().numpy()
else:
    emb_np = np.array(embeddings)

q_norm = q_np / np.linalg.norm(q_np, axis=1, keepdims=True)
d_norm = emb_np / np.linalg.norm(emb_np, axis=1, keepdims=True)
scores = (q_norm @ d_norm.T)[0].tolist()

#Ranked output
ranked = sorted(zip(scores, records), reverse=True)[:10]
print(f"\nTop 10 results for: '{query}'\n")
for score, (paper_id, fig_num, caption) in ranked:
    print(f"  {round(score*100,1)}% — Paper {paper_id}, Figure {fig_num}")
    print(f"    {caption[:150]}...")
    print()