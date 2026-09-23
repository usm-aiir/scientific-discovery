from ranx import Qrels, Run, evaluate, compare

qrels = Qrels.from_file("/mnt/netstore1_home/behrooz.mansouri/SIGIRSciDis/figureGen/figure_query_output/Test_figure_qrels.tsv", kind="trec")

run_pretrained = Run.from_file("/home/adah.holt/scientific-discovery/run_pretrained.trec")
run_finetuned  = Run.from_file("/home/adah.holt/scientific-discovery/run_finetuned.trec")

metrics = ["ndcg@10", "mrr", "recall@10", "recall@100", "precision@10"]

print("=== Pretrained ===")
print(evaluate(qrels, run_pretrained, metrics))

print("\n=== Finetuned ===")
print(evaluate(qrels, run_finetuned, metrics))

print("\n=== Comparison ===")
print(compare(qrels, runs=[run_pretrained, run_finetuned], metrics=metrics))