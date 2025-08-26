from datasets import load_dataset
# query_dataset = load_dataset("google-research-datasets/nq_open", split="validation")
# query_dataset = load_dataset("hotpot_qa", "fullwiki", split="validation")  # 或 "distractor" 配置，根据需要
query_dataset = load_dataset("mandarjoshi/trivia_qa", "rc", split="validation")  # "rc" 是阅读理解配置
