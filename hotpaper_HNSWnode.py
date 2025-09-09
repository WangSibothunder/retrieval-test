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
HIGH_LEVEL_STATS_PATH = f"high_level_stats_{dataset_name}_top{topk}.txt"
HIGH_LEVEL_PLOT_PATH = f"high_level_distribution_{dataset_name}_top{topk}.png"

# 步骤1: 加载Wikipedia知识库（使用rag-mini-wikipedia子集）
print("加载Wikipedia数据集...")
wiki_dataset = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus", split="passages")
documents = wiki_dataset["passage"]  # 列表 of strings
doc_ids = list(range(len(documents)))  # 简单ID

# 步骤2: 加载本地嵌入模型（使用BAAI/bge-large-en-v1.5，BERT家族针对RAG优化）
print("加载本地嵌入模型...")
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

# 新功能: 提取HNSW节点层级信息
hnsw = index.hnsw
levels = faiss.vector_to_array(hnsw.levels)  # 转换为NumPy数组
entry_point = hnsw.entry_point  # 入口节点
max_level = hnsw.max_level
print(f"最大层级: {max_level} (共 {max_level + 1} 层)")
print(f"入口节点ID: {entry_point}")

# 统计每层节点数
level_counts = {}
for level in range(max_level + 1):
    count = sum(1 for l in levels if l >= level)  # l >= level 表示在该层或更高
    level_counts[level] = count
    print(f"层级 {level}: {count} 个节点")

# 高层节点数 (level > 0)
high_level_nodes = sum(1 for l in levels if l > 0)
high_level_ratio = (high_level_nodes / len(levels)) * 100 if len(levels) > 0 else 0
print(f"高层节点 (level > 0) 总数: {high_level_nodes} ({high_level_ratio:.2f}% of total nodes)")

# 步骤4: 根据参数加载查询数据集
print(f"加载 {dataset_name.upper()} 数据集...")
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

queries = [item[query_key] for item in query_dataset]  # 提取查询

# 步骤5: 对于每个查询，进行检索并统计（top-k，但统计频率基于所有检索结果）
retrieved_docs = []
for query in queries:
    query_emb = get_embedding([query])[0]  # 单查询嵌入
    distances, indices = index.search(np.array([query_emb]), k=topk)  # 检索top-k
    for idx in indices[0]:  # 对于top-k，每个检索到的文档都计入频率
        retrieved_docs.append(idx)

# 步骤6: 统计频率分布
doc_freq = Counter(retrieved_docs)
freq_sorted = sorted(doc_freq.items(), key=lambda x: x[1], reverse=True)  # (doc_id, freq) 降序


# 新功能: 探索top10%热门文章中HNSW高层节点占比 (level > 0)
num_top10 = max(1, int(0.1 * len(freq_sorted)))  # top10%文章数
top10_docs = [doc_id for doc_id, freq in freq_sorted[:num_top10]]  # top10% doc_ids

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