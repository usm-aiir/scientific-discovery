#Combines BM25 and BGE-M3 rankings with deterministic reciprocal-rank fusion.
from collections import defaultdict

#Combine both rankings equally, give missing tables no score, and use table IDs to break ties.
def fuse(first, second, depth=100, constant=60):
    scores = defaultdict(float)
    for hits in (first, second):
        for rank, (uid, _) in enumerate(hits[:depth], 1):
            scores[uid] += 1 / (constant + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
