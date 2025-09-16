import argparse
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
import logging

# 配置matplotlib中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 解析命令行参数
parser = argparse.ArgumentParser(description="RAG热门文章分布分析")
parser.add_argument("--dataset", type=str, default="mmlu", choices=["mmlu", "nq", "hotpotqa", "triviaqa"],
                    help="选择查询数据集: mmlu, nq, hotpotqa, triviaqa")
parser.add_argument("--topk", type=int, default=10,
                    help="检索的top-k值 (默认: 10)")
parser.add_argument("--batch_size", type=int, default=512,
                    help="批次大小 (默认: 512)")
args = parser.parse_args()

dataset_name = args.dataset.lower()
topk = args.topk

# 配置日志（在args可用后配置）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"log_{dataset_name}_{topk}.log"),
        logging.StreamHandler()  # 同时输出到控制台
    ]
)

# 文件路径
EMBEDDINGS_PATH = "doc_embeddings_100k.npy"
INDEX_PATH = "hnsw_index_100k.bin"
WIKI_DATA_PATH = os.path.join("wikipedia_data", "wikipedia_100k.json")
DATASET_CACHE_DIR = "dataset_cache"
QUERY_EMBEDDINGS_PATH = os.path.join(DATASET_CACHE_DIR, f"query_embeddings_{dataset_name}.npy")
DIST_PLOT_PATH = f"hot_docs_distribution_{dataset_name}_top{topk}.png"
FREQ_STATS_PATH = f"freq_stats_{dataset_name}_top{topk}.txt"
HIGH_LEVEL_STATS_PATH = f"high_level_stats_{dataset_name}_top{topk}.txt"
HIGH_LEVEL_PLOT_PATH = f"high_level_distribution_{dataset_name}_top{topk}.png"
DEGREE_DIST_PLOT_PATH = f"degree_distribution_{dataset_name}_top{topk}.png"
HOT_DEGREE_PLOT_PATH = f"hot_degree_distribution_{dataset_name}_top{topk}.png"
TOP10_HOT_DOCS_PATH = f"top10_hot_docs_{dataset_name}_top{topk}.txt"
DEGREE_STATS_PATH = f"degree_stats_{dataset_name}_top{topk}.txt"

# 确保目录存在
os.makedirs("wikipedia_data", exist_ok=True)
os.makedirs(DATASET_CACHE_DIR, exist_ok=True)

# 日志记录实验配置
logging.info("=== 实验配置 ===")
logging.info(f"数据集: {dataset_name}")
logging.info(f"Top-k值: {topk}")
logging.info(f"批次大小: {args.batch_size}")
logging.info("================")

# 步骤1: 加载Wikipedia知识库（使用Wikipedia 100K子集）
logging.info("加载Wikipedia数据集...")
if os.path.exists(WIKI_DATA_PATH):
    logging.info(f"找到本地Wikipedia数据 {WIKI_DATA_PATH}，加载中...")
    with open(WIKI_DATA_PATH, "r", encoding="utf-8") as f:
        wiki_data = json.load(f)
    documents = wiki_data["text"]
    doc_ids = list(range(len(documents)))
else:
    logging.info("未找到本地Wikipedia数据，从Hugging Face下载...")
    wiki_dataset = load_dataset("wikipedia", "20220301.en", split="train[:100000]")
    documents = wiki_dataset["text"]
    doc_ids = list(range(len(documents)))
    wiki_data = {"text": documents}
    with open(WIKI_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(wiki_data, f, ensure_ascii=False)
    logging.info(f"Wikipedia数据保存到 {WIKI_DATA_PATH}")

# 步骤2: 加载本地嵌入模型（使用BAAI/bge-large-en-v1.5，BERT家族针对RAG优化）
logging.info("加载本地嵌入模型...")
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
            logging.info(f"使用本地缓存模型: {local_path}")
            model = SentenceTransformer(local_path)
            break
        except Exception as e:
            logging.error(f"加载本地模型失败 {local_path}: {e}")
            continue

if model is None:
    logging.info("未找到本地缓存模型，从Hugging Face下载...")
    model = SentenceTransformer('BAAI/bge-large-en-v1.5')

# 生成或加载文档嵌入
def get_embedding(texts, batch_size=args.batch_size):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_emb = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        embeddings.extend(batch_emb)
        logging.info(f"嵌入生成进度: {len(embeddings)} / {len(texts)}")
    return np.array(embeddings, dtype=np.float32)

logging.info("检查或生成Wikipedia文档嵌入...")
if os.path.exists(EMBEDDINGS_PATH):
    logging.info(f"找到嵌入文件 {EMBEDDINGS_PATH}，加载中...")
    doc_embeddings = np.load(EMBEDDINGS_PATH)
else:
    logging.info("未找到嵌入文件，生成嵌入...")
    doc_embeddings = get_embedding(documents)
    np.save(EMBEDDINGS_PATH, doc_embeddings)
    logging.info(f"嵌入已保存到 {EMBEDDINGS_PATH}")

# 步骤3: 构建或加载HNSW索引
embedding_dim = doc_embeddings.shape[1]
if os.path.exists(INDEX_PATH):
    logging.info(f"找到索引文件 {INDEX_PATH}，加载中...")
    index = faiss.read_index(INDEX_PATH)
else:
    logging.info("未找到索引文件，构建HNSW索引...")
    index = faiss.IndexHNSWFlat(embedding_dim, 32)  # M=32是常见HNSW参数，平衡速度和准确性
    index.add(doc_embeddings)
    faiss.write_index(index, INDEX_PATH)
    logging.info(f"索引已保存到 {INDEX_PATH}")
logging.info("HNSW索引准备完成.")

# 新功能: 提取HNSW节点层级信息
hnsw = index.hnsw
levels = faiss.vector_to_array(hnsw.levels)  # 转换为NumPy数组
entry_point = hnsw.entry_point  # 入口节点
max_level = hnsw.max_level
logging.info(f"最大层级: {max_level} (共 {max_level + 1} 层)")
logging.info(f"入口节点ID: {entry_point}")

# 统计每层节点数
level_counts = {}
for level in range(max_level + 1):
    count = sum(1 for l in levels if l >= level)  # l >= level 表示在该层或更高
    level_counts[level] = count
    logging.info(f"层级 {level}: {count} 个节点")

# 高层节点数 (level > 0)
high_level_nodes = sum(1 for l in levels if l > 0)
high_level_ratio = (high_level_nodes / len(levels)) * 100 if len(levels) > 0 else 0
logging.info(f"高层节点 (level > 0) 总数: {high_level_nodes} ({high_level_ratio:.2f}% of total nodes)")

# 新功能: 统计HNSW中各个节点的度 (总邻居数)
offsets = faiss.vector_to_array(hnsw.offsets)
neighbors = faiss.vector_to_array(hnsw.neighbors)
ntotal = len(levels)
degrees = [int(offsets[i+1] - offsets[i]) for i in range(ntotal)]

# 统计度分布
degree_freq = Counter(degrees)
degree_freq_sorted = sorted(degree_freq.items(), key=lambda x: x[0])  # 按度升序

logging.info("\nHNSW节点度分布:")
for deg, count in degree_freq_sorted:
    logging.info(f"度 {deg}: {count} 个节点")

# 绘制度分布图
plt.figure(figsize=(10, 6))
plt.bar([d[0] for d in degree_freq_sorted], [d[1] for d in degree_freq_sorted])
plt.title(f"HNSW节点度分布 - {dataset_name.upper()} Top-{topk}")
plt.xlabel("度")
plt.ylabel("节点数")
plt.grid(True)
plt.savefig(DEGREE_DIST_PLOT_PATH)
logging.info(f"度分布图保存为 {DEGREE_DIST_PLOT_PATH}")

# 步骤4: 根据参数加载查询数据集
logging.info(f"加载 {dataset_name.upper()} 数据集...")
dataset_cache_path = os.path.join(DATASET_CACHE_DIR, f"{dataset_name}_validation.json")
if os.path.exists(dataset_cache_path):
    logging.info(f"找到本地缓存数据集 {dataset_cache_path}，加载中...")
    with open(dataset_cache_path, "r", encoding="utf-8") as f:
        query_data = json.load(f)
    queries = [item["question"] for item in query_data]
else:
    logging.info("未找到本地缓存数据集，从Hugging Face下载...")
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
    logging.info(f"数据集保存到 {dataset_cache_path}")
    queries = [item["question"] for item in query_data]

# 步骤5: 生成或加载查询嵌入
logging.info("检查或生成查询嵌入...")
if os.path.exists(QUERY_EMBEDDINGS_PATH):
    logging.info(f"找到查询嵌入文件 {QUERY_EMBEDDINGS_PATH}，加载中...")
    query_embs = np.load(QUERY_EMBEDDINGS_PATH)
else:
    logging.info("未找到查询嵌入文件，生成嵌入...")
    query_embs = get_embedding(queries)
    np.save(QUERY_EMBEDDINGS_PATH, query_embs)
    logging.info(f"查询嵌入保存到 {QUERY_EMBEDDINGS_PATH}")

# 步骤6: 对于每个查询，进行检索并统计（top-k，但统计频率基于所有检索结果）
retrieved_docs = []
for i in range(0, len(query_embs), args.batch_size):
    batch_embs = query_embs[i:i + args.batch_size]
    distances, indices = index.search(batch_embs, k=topk)  # 批量检索
    for idx_batch in indices:
        for idx in idx_batch:
            retrieved_docs.append(idx)
logging.info(f"检索完成，总检索文档数: {len(retrieved_docs)}")

# 步骤7: 统计频率分布
doc_freq = Counter(retrieved_docs)
freq_sorted = sorted(doc_freq.items(), key=lambda x: x[1], reverse=True)  # (doc_id, freq) 降序

# 打印并保存top-10热门文档频率
logging.info("\nTop-10热门文档频率:")
with open(FREQ_STATS_PATH, "w") as f:
    for rank, (doc_id, freq) in enumerate(freq_sorted[:10], 1):
        stat = f"Rank {rank}: Doc {doc_id} - {freq} 次"
        logging.info(stat)
        f.write(stat + "\n")

# 计算累积分布（验证热门现象）
total_retrievals = len(retrieved_docs)  # 注意：现在是top-k的总检索次数
cumulative = np.cumsum([f[1] for f in freq_sorted]) / total_retrievals
top10_percent = cumulative[int(0.1 * len(freq_sorted)) - 1] * 100 if len(freq_sorted) > 0 else 0
logging.info(f"Top 10% 文档占总检索的 {top10_percent:.2f}%")
with open(FREQ_STATS_PATH, "a") as f:
    f.write(f"Top 10% 文档占总检索的 {top10_percent:.2f}%\n")

logging.info(f"频率统计保存到 {FREQ_STATS_PATH}")

# 步骤8: 绘制频率分布图
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(freq_sorted) + 1), [f[1] for f in freq_sorted], marker='o')
plt.xscale('log')
plt.yscale('log')
plt.title(f"检索到的文章频率分布 (Log-Log Scale) - {dataset_name.upper()} Top-{topk}")
plt.xlabel("文章排名 (由频率降序)")
plt.ylabel("检索频率")
plt.grid(True)
plt.savefig(DIST_PLOT_PATH)
logging.info(f"频率分布图保存为 {DIST_PLOT_PATH}")

# 新功能: 探索top10%热门文章中HNSW高层节点占比 (level > 0)
num_top10 = max(1, int(0.1 * len(freq_sorted)))  # top10%文章数
top10_docs = [doc_id for doc_id, freq in freq_sorted[:num_top10]]  # top10% doc_ids

high_level_count = sum(1 for doc_id in top10_docs if levels[doc_id] > 0)
high_level_ratio = high_level_count / len(top10_docs) * 100 if top10_docs else 0

# 打印示例 (前10热门是否高层)
logging.info("\nTop-10热门文章中高层节点 (level > 0):")
with open(HIGH_LEVEL_STATS_PATH, "w") as f:
    for rank, (doc_id, freq) in enumerate(freq_sorted[:10], 1):
        is_high_level = levels[doc_id] > 0
        level = levels[doc_id]
        stat = f"Rank {rank}: Doc {doc_id} (Freq {freq}) - 高层节点: {is_high_level} (层级 {level})"
        logging.info(stat)
        f.write(stat + "\n")
    f.write(f"\nTop 10% 热门文章中高层节点占比: {high_level_ratio:.2f}%\n")

logging.info(f"高层节点统计保存到 {HIGH_LEVEL_STATS_PATH}")

# 绘制热门文章层级分布图
plt.figure(figsize=(10, 6))
hot_levels = [levels[doc_id] for doc_id in top10_docs]
plt.hist(hot_levels, bins=range(int(min(hot_levels)-1), int(max(hot_levels)+2)), edgecolor='black')
plt.title(f"Top 10% 热门文章层级分布 - {dataset_name.upper()} Top-{topk}")
plt.xlabel("层级")
plt.ylabel("文章数")
plt.grid(True)
plt.savefig(HIGH_LEVEL_PLOT_PATH)
logging.info(f"高层节点分布图保存为 {HIGH_LEVEL_PLOT_PATH}")

# 新功能: 统计HNSW中各个节点的度 (总邻居数)
offsets = faiss.vector_to_array(hnsw.offsets)
neighbors = faiss.vector_to_array(hnsw.neighbors)
ntotal = len(levels)
degrees = []
for i in range(ntotal):
    start = offsets[i]
    end = offsets[i+1]
    valid_neighbors = [n for n in neighbors[start:end] if n != -1]
    degrees.append(len(valid_neighbors))

# 统计度分布
degree_freq = Counter(degrees)
degree_freq_sorted = sorted(degree_freq.items(), key=lambda x: x[0])  # 按度升序

logging.info("\nHNSW节点度分布:")
with open(DEGREE_STATS_PATH, "w") as f:
    for deg, count in degree_freq_sorted:
        stat = f"度 {deg}: {count} 个节点"
        logging.info(stat)
        f.write(stat + "\n")

# 热门文章的度
hot_degrees = [degrees[doc_id] for doc_id in top10_docs]

logging.info("\nTop-10热门文章的度:")
for rank, (doc_id, freq) in enumerate(freq_sorted[:10], 1):
    deg = degrees[doc_id]
    logging.info(f"Rank {rank}: Doc {doc_id} (Freq {freq}) - 度: {deg}")

# 计算热门文章平均度 vs 整体平均度
avg_degree_all = np.mean(degrees)
avg_degree_hot = np.mean(hot_degrees)
logging.info(f"\n整体平均度: {avg_degree_all:.2f}")
logging.info(f"Top 10% 热门文章平均度: {avg_degree_hot:.2f}")
logging.info(f"热门文章平均度是否高于整体: {avg_degree_hot > avg_degree_all}")

with open(DEGREE_STATS_PATH, "a") as f:
    f.write(f"\n整体平均度: {avg_degree_all:.2f}\n")
    f.write(f"Top 10% 热门文章平均度: {avg_degree_hot:.2f}\n")
    f.write(f"热门文章平均度是否高于整体: {avg_degree_hot > avg_degree_all}\n")

logging.info(f"度统计保存到 {DEGREE_STATS_PATH}")

# 绘制度分布图 (整体)
plt.figure(figsize=(10, 6))
plt.bar([d[0] for d in degree_freq_sorted], [d[1] for d in degree_freq_sorted])
plt.title(f"HNSW节点度分布 - {dataset_name.upper()} Top-{topk}")
plt.xlabel("度")
plt.ylabel("节点数")
plt.grid(True)
plt.savefig(DEGREE_DIST_PLOT_PATH)
logging.info(f"度分布图保存为 {DEGREE_DIST_PLOT_PATH}")

# 绘制热门文章度分布图
plt.figure(figsize=(10, 6))
plt.hist(hot_degrees, bins=range(int(min(hot_degrees)-1), int(max(hot_degrees)+2)), edgecolor='black')
plt.title(f"Top 10% 热门文章度分布 - {dataset_name.upper()} Top-{topk}")
plt.xlabel("度")
plt.ylabel("文章数")
plt.grid(True)
plt.savefig(HOT_DEGREE_PLOT_PATH)
logging.info(f"热门文章度分布图保存为 {HOT_DEGREE_PLOT_PATH}")

# 新功能: 保存top10%热门文章数据 (rank, id, 度, 层级)
num_top10 = max(1, int(0.1 * len(freq_sorted)))
top10_docs = [doc_id for doc_id, freq in freq_sorted[:num_top10]]  # top10% doc_ids

logging.info("\n保存Top 10% 热门文章数据...")
with open(TOP10_HOT_DOCS_PATH, "w") as f:
    f.write("Rank,ID,度,层级\n")
    for rank, (doc_id, freq) in enumerate(freq_sorted[:num_top10], 1):
        deg = degrees[doc_id]
        level = levels[doc_id]
        stat = f"{rank},{doc_id},{deg},{level}\n"
        f.write(stat)

logging.info(f"Top 10% 热门文章数据保存到 {TOP10_HOT_DOCS_PATH}")