import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
# 配置matplotlib中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# 文件路径（嵌入和索引基于Wikipedia，不随查询数据集变）
EMBEDDINGS_PATH = "doc_embeddings.npy"
INDEX_PATH = "hnsw_index.bin"

# 解析命令行参数
parser = argparse.ArgumentParser(description="RAG热门文章分布分析")
parser.add_argument("--dataset", type=str, default="mmlu", choices=["mmlu", "nq", "hotpotqa", "triviaqa"],
                    help="选择查询数据集: mmlu, nq, hotpotqa, triviaqa")
parser.add_argument("--topk", type=int, default=10,
                    help="检索的top-k值 (默认: 10)")
args = parser.parse_args()

dataset_name = args.dataset.lower()
topk = args.topk

# 输出文件命名，反映数据集和top-k
DIST_PLOT_PATH = f"hot_docs_distribution_{dataset_name}_top{topk}.png"
FREQ_STATS_PATH = f"freq_stats_{dataset_name}_top{topk}.txt"

# 步骤1: 加载Wikipedia知识库（使用rag-mini-wikipedia子集）
print("加载Wikipedia数据集...")
wiki_dataset = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus", split="passages")
documents = wiki_dataset["passage"]  # 列表 of strings
doc_ids = list(range(len(documents)))  # 简单ID

# 步骤2: 加载本地嵌入模型（使用BAAI/bge-large-en-v1.5，BERT家族针对RAG优化）
print("加载本地嵌入模型...")
# 优先使用本地缓存模型
local_model_paths = [
    r"L:\huggingface\cache\hub",  # HuggingFace cache路径
    "./models/BAAI_bge-large-en-v1.5",  # 本地模型路径1
    "./BAAI_bge-large-en-v1.5",        # 本地模型路径2
    "models/BAAI_bge-large-en-v1.5",    # 本地模型路径3
    "BAAI_bge-large-en-v1.5"           # 本地模型路径4
]

model = None
for local_path in local_model_paths:
    if os.path.exists(local_path):
        try:
            print(f"使用本地缓存模型: {local_path}")
            model = SentenceTransformer(local_path)
            break
        except Exception as e:
            print(f"加载本地模型失败 {local_path}: {e}")
            continue

if model is None:
    print("未找到本地缓存模型，从Hugging Face下载...")
    model = SentenceTransformer('BAAI/bge-large-en-v1.5')  # 维度1024，基于BERT，专为RAG语义搜索设计

# 生成或加载文档嵌入
def get_embedding(texts, batch_size=100):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_emb = model.encode(batch, normalize_embeddings=True)  # 标准化嵌入
        embeddings.extend(batch_emb)
        print(f"嵌入生成进度: {len(embeddings)} / {len(texts)}")
    return np.array(embeddings, dtype=np.float32)

print("检查或生成Wikipedia文档嵌入...")
if os.path.exists(EMBEDDINGS_PATH):
    print(f"找到嵌入文件 {EMBEDDINGS_PATH}，加载中...")
    doc_embeddings = np.load(EMBEDDINGS_PATH)
else:
    print("未找到嵌入文件，生成嵌入...")
    doc_embeddings = get_embedding(documents)
    np.save(EMBEDDINGS_PATH, doc_embeddings)
    print(f"嵌入已保存到 {EMBEDDINGS_PATH}")

# 步骤3: 构建或加载HNSW索引
embedding_dim = doc_embeddings.shape[1]  # 1024 for bge-large-en-v1.5
if os.path.exists(INDEX_PATH):
    print(f"找到索引文件 {INDEX_PATH}，加载中...")
    index = faiss.read_index(INDEX_PATH)
else:
    print("未找到索引文件，构建HNSW索引...")
    index = faiss.IndexHNSWFlat(embedding_dim, 32)  # M=32是常见HNSW参数，平衡速度和准确性
    index.add(doc_embeddings)
    faiss.write_index(index, INDEX_PATH)
    print(f"索引已保存到 {INDEX_PATH}")
print("HNSW索引准备完成。")

# 步骤4: 根据参数加载查询数据集
print(f"加载 {dataset_name.upper()} 数据集...")
if dataset_name == "mmlu":
    query_dataset = load_dataset("cais/mmlu", "all", split="validation")
    query_key = "question"
elif dataset_name == "nq":
    query_dataset = load_dataset("google-research-datasets/nq_open", split="validation")
    query_key = "question"
elif dataset_name == "hotpotqa":
    query_dataset = load_dataset("hotpot_qa", "fullwiki", split="validation")  # 或 "distractor" 配置，根据需要
    query_key = "question"
elif dataset_name == "triviaqa":
    query_dataset = load_dataset("mandarjoshi/trivia_qa", "rc", split="validation")  # "rc" 是阅读理解配置
    query_key = "question"
else:
    raise ValueError(f"未知数据集: {dataset_name}")

queries = [item[query_key] for item in query_dataset]  # 提取查询

# 步骤5: 对于每个查询，进行检索并统计（top-k，但统计频率基于所有检索结果）
retrieved_docs = []
retrieved_sequences = []  # 存储每个查询的top-10序列
for query in queries:
    query_emb = get_embedding([query])[0]  # 单查询嵌入
    distances, indices = index.search(np.array([query_emb]), k=topk)  # 检索top-k
    seq = list(indices[0])  # top-10序列
    retrieved_sequences.append(seq)
    for idx in seq:  # 对于top-k，每个检索到的文档都计入频率
        retrieved_docs.append(idx)

# 步骤6: 统计频率分布


# 新功能: 统计连续的2,3,4个文章对（n-gram）
def extract_ngrams(sequences, n):
    ngrams = []
    for seq in sequences:
        for i in range(len(seq) - n + 1):
            ngram = tuple(seq[i:i+n])  # 有序tuple
            ngrams.append(ngram)
    return ngrams

for n in [2, 3, 4]:
    ngrams = extract_ngrams(retrieved_sequences, n)
    ngram_freq = Counter(ngrams)
# Create a new, empty Counter object. And if given, count elements from an input iterable. Or, initialize the count from another mapping of elements to their counts.

# >>> c = Counter()                           # a new, empty counter
# >>> c = Counter('gallahad')                 # a new counter from an iterable
# >>> c = Counter({'a': 4, 'b': 2})           # a new counter from a mapping
# >>> c = Counter(a=4, b=2)                   # a new counter from keyword args

    ngram_freq_sorted = sorted(ngram_freq.values(), reverse=True)  # 降序频率
    
    # 输出文件命名
    NGRAM_STATS_PATH = f"ngram_stats_n{n}_{dataset_name}_top{topk}.txt"
    NGRAM_PLOT_PATH = f"ngram_distribution_n{n}_{dataset_name}_top{topk}.png"
    
    # 打印并保存top-10 n-gram频率
    print(f"\n连续 {n} 个文章对 Top-10 频率:")
    with open(NGRAM_STATS_PATH, "w") as f:
        for rank, freq in enumerate(ngram_freq_sorted[:10], 1):
            stat = f"Rank {rank}: {freq} 次"
            print(stat)
            f.write(stat + "\n")
    
    # 计算累积分布
    total_ngrams = len(ngrams)  # 总n-gram次数
    ngram_cumulative = np.cumsum(ngram_freq_sorted) / total_ngrams

# # 假设有以下数据：
# ngram_freq_sorted = [100, 80, 60, 40, 20, 10, 5, 3, 2, 1]  # 10个n-gram的频率
# total_freq = sum(ngram_freq_sorted)  # 总频率 = 321
# ngram_cumulative = [100/321, 180/321, 240/321, ...]  # 累积占比

# # 计算前10% (即前1个) 的占比：
# # int(0.1 * 10) - 1 = int(1.0) - 1 = 0
# # ngram_cumulative[0] = 100/321 ≈ 0.311
# # ngram_top10_percent = 0.311 * 100 = 31.1%

    ngram_top10_percent = ngram_cumulative[int(0.1 * len(ngram_freq_sorted)) - 1] * 100 if len(ngram_freq_sorted) > 0 else 0
    print(f"Top 10% {n}-gram 占总访问的 {ngram_top10_percent:.2f}%")
    with open(NGRAM_STATS_PATH, "a") as f:
        f.write(f"Top 10% {n}-gram 占总访问的 {ngram_top10_percent:.2f}%\n")
    print(f"{n}-gram 统计保存到 {NGRAM_STATS_PATH}")
    
    # 绘制n-gram分布图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(ngram_freq_sorted) + 1), ngram_freq_sorted, marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f"连续 {n} 个文章对频率分布 (Log-Log Scale) - {dataset_name.upper()} Top-{topk}")
    plt.xlabel("对排名 (由频率降序)")
    plt.ylabel("访问频率")
    plt.grid(True)
    plt.savefig(NGRAM_PLOT_PATH)
    print(f"{n}-gram 分布图保存为 {NGRAM_PLOT_PATH}")