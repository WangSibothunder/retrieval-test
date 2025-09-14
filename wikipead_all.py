import argparse
import os
import json
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

# 文件路径
EMBEDDINGS_PATH = "doc_embeddings_100k.npy"
INDEX_PATH = "hnsw_index_100k.bin"
WIKI_DATA_PATH = os.path.join("wikipedia_data", "wikipedia_100k.json")
DATASET_CACHE_DIR = "dataset_cache"

# 确保目录存在
os.makedirs("wikipedia_data", exist_ok=True)
os.makedirs(DATASET_CACHE_DIR, exist_ok=True)

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
HIGH_LEVEL_STATS_PATH = f"high_level_stats_{dataset_name}_top{topk}.txt"
HIGH_LEVEL_PLOT_PATH = f"high_level_distribution_{dataset_name}_top{topk}.png"
NGRAM_STATS_PATH_BASE = f"ngram_stats_n{{}}_{dataset_name}_top{topk}.txt"
NGRAM_PLOT_PATH_BASE = f"ngram_distribution_n{{}}_{dataset_name}_top{topk}.png"

# 步骤1: 加载Wikipedia知识库（使用完整Wikipedia数据集的前100000篇文章）
print("加载Wikipedia数据集...")
if os.path.exists(WIKI_DATA_PATH):
    print(f"找到本地Wikipedia数据 {WIKI_DATA_PATH}，加载中...")
    with open(WIKI_DATA_PATH, "r", encoding="utf-8") as f:
        wiki_data = json.load(f)
    documents = wiki_data["text"]
    doc_ids = list(range(len(documents)))
else:
    print("未找到本地Wikipedia数据，从Hugging Face下载...")
    wiki_dataset = load_dataset("wikipedia", "20220301.en", split="train[:100000]")
    documents = wiki_dataset["text"]
    doc_ids = list(range(len(documents)))
    wiki_data = {"text": documents}
    with open(WIKI_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(wiki_data, f, ensure_ascii=False)
    print(f"Wikipedia数据保存到 {WIKI_DATA_PATH}")

# 步骤2: 加载本地嵌入模型（使用BAAI/bge-large-en-v1.5，BERT家族针对RAG优化）
print("加载本地嵌入模型...")
local_model_paths = [
    r"L:\huggingface\cache\hub",  # HuggingFace cache路径
    "./models/BAAI_bge-large-en-v1.5",
    "./BAAI_bge-large-en-v1.5",
    "models/BAAI_bge-large-en-v1.5",
    "BAAI_bge-large-en-v1.5"
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
    model = SentenceTransformer('BAAI/bge-large-en-v1.5')

# 生成或加载文档嵌入
def get_embedding(texts, batch_size=100):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_emb = model.encode(batch, normalize_embeddings=True)
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
embedding_dim = doc_embeddings.shape[1]
if os.path.exists(INDEX_PATH):
    print(f"找到索引文件 {INDEX_PATH}，加载中...")
    index = faiss.read_index(INDEX_PATH)
else:
    print("未找到索引文件，构建HNSW索引...")
    index = faiss.IndexHNSWFlat(embedding_dim, 32)
    index.add(doc_embeddings)
    faiss.write_index(index, INDEX_PATH)
    print(f"索引已保存到 {INDEX_PATH}")
print("HNSW索引准备完成.")

# 新功能: 提取HNSW节点层级信息
hnsw = index.hnsw
levels = faiss.vector_to_array(hnsw.levels)
entry_point = hnsw.entry_point
max_level = hnsw.max_level
print(f"最大层级: {max_level} (共 {max_level + 1} 层)")
print(f"入口节点ID: {entry_point}")

# 统计每层节点数
level_counts = {}
for level in range(max_level + 1):
    count = sum(1 for l in levels if l >= level)
    level_counts[level] = count
    print(f"层级 {level}: {count} 个节点")

# 高层节点数 (level > 0)
high_level_nodes = sum(1 for l in levels if l > 0)
high_level_ratio = (high_level_nodes / len(levels)) * 100 if len(levels) > 0 else 0
print(f"高层节点 (level > 0) 总数: {high_level_nodes} ({high_level_ratio:.2f}% of total nodes)")

# 步骤4: 加载查询数据集（本地缓存或从Hugging Face下载）
print(f"加载 {dataset_name.upper()} 数据集...")
dataset_cache_path = os.path.join(DATASET_CACHE_DIR, f"{dataset_name}_validation.json")
if os.path.exists(dataset_cache_path):
    print(f"找到本地缓存数据集 {dataset_cache_path}，加载中...")
    with open(dataset_cache_path, "r", encoding="utf-8") as f:
        query_data = json.load(f)
    queries = [item["question"] for item in query_data]
else:
    print("未找到本地缓存数据集，从Hugging Face下载...")
    if dataset_name == "mmlu":
        query_dataset = load_dataset("cais/mmlu", "all", split="validation")
        query_key = "question"
    elif dataset_name == "nq":
        query_dataset = load_dataset("google-research-datasets/nq_open", split="validation")
        query_key = "question"
    elif dataset_name == "hotpotqa":
        query_dataset = load_dataset("hotpot_qa", "fullwiki", split="validation")
        query_key = "question"
    elif dataset_name == "triviaqa":
        query_dataset = load_dataset("mandarjoshi/trivia_qa", "rc", split="validation")
        query_key = "question"
    else:
        raise ValueError(f"未知数据集: {dataset_name}")
    query_data = [{"question": item[query_key]} for item in query_dataset]
    with open(dataset_cache_path, "w", encoding="utf-8") as f:
        json.dump(query_data, f, ensure_ascii=False)
    print(f"数据集保存到 {dataset_cache_path}")
    queries = [item["question"] for item in query_data]

# 步骤5: 对于每个查询，进行检索并统计（top-k，但统计频率基于所有检索结果）
retrieved_docs = []
retrieved_sequences = []
for query in queries:
    query_emb = get_embedding([query])[0]
    distances, indices = index.search(np.array([query_emb]), k=topk)
    seq = list(indices[0])
    retrieved_sequences.append(seq)
    for idx in seq:
        retrieved_docs.append(idx)

# 步骤6: 统计频率分布
doc_freq = Counter(retrieved_docs)
freq_sorted = sorted(doc_freq.items(), key=lambda x: x[1], reverse=True)

# 打印并保存top-10热门文档频率
print("Top-10热门文档频率:")
with open(FREQ_STATS_PATH, "w") as f:
    for rank, (doc_id, freq) in enumerate(freq_sorted[:10], 1):
        stat = f"Rank {rank}: Doc {doc_id} - {freq} 次"
        print(stat)
        f.write(stat + "\n")

# 计算累积分布
total_retrievals = len(retrieved_docs)
cumulative = np.cumsum([f[1] for f in freq_sorted]) / total_retrievals
top10_percent = cumulative[int(0.1 * len(freq_sorted)) - 1] * 100 if len(freq_sorted) > 0 else 0
print(f"Top 10% 文档占总检索的 {top10_percent:.2f}%")
with open(FREQ_STATS_PATH, "a") as f:
    f.write(f"Top 10% 文档占总检索的 {top10_percent:.2f}%\n")
print(f"频率统计保存到 {FREQ_STATS_PATH}")

# 步骤7: 绘制频率分布图
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(freq_sorted) + 1), [f[1] for f in freq_sorted], marker='o')
plt.xscale('log')
plt.yscale('log')
plt.title(f"检索到的文章频率分布 (Log-Log Scale) - {dataset_name.upper()} Top-{topk}")
plt.xlabel("文章排名 (由频率降序)")
plt.ylabel("检索频率")
plt.grid(True)
plt.savefig(DIST_PLOT_PATH)
print(f"频率分布图保存为 {DIST_PLOT_PATH}")

# 新功能: 统计连续的2,3,4个文章对（n-gram）
def extract_ngrams(sequences, n):
    ngrams = []
    for seq in sequences:
        for i in range(len(seq) - n + 1):
            ngram = tuple(seq[i:i+n])
            ngrams.append(ngram)
    return ngrams

for n in [2, 3, 4]:
    ngrams = extract_ngrams(retrieved_sequences, n)
    ngram_freq = Counter(ngrams)
    ngram_freq_sorted = sorted(ngram_freq.values(), reverse=True)
    
    # 输出文件命名
    NGRAM_STATS_PATH = NGRAM_STATS_PATH_BASE.format(n)
    NGRAM_PLOT_PATH = NGRAM_PLOT_PATH_BASE.format(n)
    
    # 打印并保存top-10 n-gram频率
    print(f"\n连续 {n} 个文章对 Top-10 频率:")
    with open(NGRAM_STATS_PATH, "w") as f:
        for rank, freq in enumerate(ngram_freq_sorted[:10], 1):
            stat = f"Rank {rank}: {freq} 次"
            print(stat)
            f.write(stat + "\n")
    
    # 计算累积分布
    total_ngrams = len(ngrams)
    ngram_cumulative = np.cumsum(ngram_freq_sorted) / total_ngrams
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

# 新功能: 探索top10%热门文章中HNSW高层节点占比 (level > 0)
num_top10 = max(1, int(0.1 * len(freq_sorted)))
top10_docs = [doc_id for doc_id, freq in freq_sorted[:num_top10]]

high_level_count = sum(1 for doc_id in top10_docs if levels[doc_id] > 0)
high_level_ratio = high_level_count / len(top10_docs) * 100 if top10_docs else 0

# 打印示例 (前10热门是否高层)
print("\nTop-10热门文章中高层节点 (level > 0):")
with open(HIGH_LEVEL_STATS_PATH, "w") as f:
    for rank, (doc_id, freq) in enumerate(freq_sorted[:10], 1):
        is_high_level = levels[doc_id] > 0
        level = levels[doc_id]
        stat = f"Rank {rank}: Doc {doc_id} (Freq {freq}) - 高层节点: {is_high_level} (层级 {level})"
        print(stat)
        f.write(stat + "\n")
    f.write(f"\nTop 10% 热门文章中高层节点占比: {high_level_ratio:.2f}%\n")

print(f"高层节点统计保存到 {HIGH_LEVEL_STATS_PATH}")

# 绘制热门文章层级分布图
plt.figure(figsize=(10, 6))
hot_levels = [levels[doc_id] for doc_id in top10_docs]
plt.hist(hot_levels, bins=range(max(hot_levels)+2), edgecolor='black')
plt.title(f"Top 10% 热门文章层级分布 - {dataset_name.upper()} Top-{topk}")
plt.xlabel("层级")
plt.ylabel("文章数")
plt.grid(True)
plt.savefig(HIGH_LEVEL_PLOT_PATH)
print(f"高层节点分布图保存为 {HIGH_LEVEL_PLOT_PATH}")