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
parser.add_argument("--topk", type=int, default=1,
                    help="检索的top-k值 (默认: 1)")
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
    r"L:\huggingface\cache\hub\models--BAAI--bge-large-en-v1.5",  # HuggingFace cache路径
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
retrieved_ordered_combos = []
retrieved_unordered_combos = []
for query in queries:
    query_emb = get_embedding([query])[0]  # 单查询嵌入
    distances, indices = index.search(np.array([query_emb]), k=topk)  # 检索top-k
    ordered_combo = tuple(indices[0])  # 有序组合：检索顺序
    unordered_combo = frozenset(indices[0])  # 无序组合：忽略顺序
    retrieved_ordered_combos.append(ordered_combo)
    retrieved_unordered_combos.append(unordered_combo)
    for idx in indices[0]:  # 对于top-k，每个检索到的文档都计入频率
        retrieved_docs.append(idx)

# # 步骤6: 统计频率分布
# doc_freq = Counter(retrieved_docs)
# freq_sorted = sorted(doc_freq.values(), reverse=True)  # 降序频率

# # 打印并保存top-10热门文档频率
# print("Top-10热门文档频率:")
# with open(FREQ_STATS_PATH, "w") as f:
#     for rank, freq in enumerate(freq_sorted[:10], 1):
#         stat = f"Rank {rank}: {freq} 次"
#         print(stat)
#         f.write(stat + "\n")

# # 计算累积分布（验证热门现象）
# total_retrievals = len(retrieved_docs)  # 注意：现在是top-k的总检索次数
# cumulative = np.cumsum(freq_sorted) / total_retrievals
# top10_percent = cumulative[int(0.1 * len(freq_sorted)) - 1] * 100 if len(freq_sorted) > 0 else 0
# print(f"Top 10% 文档占总检索的 {top10_percent:.2f}%")
# with open(FREQ_STATS_PATH, "a") as f:
#     f.write(f"Top 10% 文档占总检索的 {top10_percent:.2f}%\n")

# print(f"频率统计保存到 {FREQ_STATS_PATH}")

# # 步骤7: 绘制频率分布图
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(freq_sorted) + 1), freq_sorted, marker='o')
# plt.xscale('log')
# plt.yscale('log')
# plt.title(f"检索到的文章频率分布 (Log-Log Scale) - {dataset_name.upper()} Top-{topk}")
# plt.xlabel("文章排名 (由频率降序)")
# plt.ylabel("检索频率")
# plt.grid(True)
# plt.savefig(DIST_PLOT_PATH)
# print(f"频率分布图保存为 {DIST_PLOT_PATH}")

#==========================================================================
# 新功能: 统计有序和无序文章组合
# 有序组合频率
ordered_combo_freq = Counter(retrieved_ordered_combos)
ordered_freq_sorted = sorted(ordered_combo_freq.values(), reverse=True)

# 无序组合频率
unordered_combo_freq = Counter(retrieved_unordered_combos)
unordered_freq_sorted = sorted(unordered_combo_freq.values(), reverse=True)

# 输出文件命名
ORDERED_STATS_PATH = f"ordered_combo_stats_{dataset_name}_top{topk}.txt"
UNORDERED_STATS_PATH = f"unordered_combo_stats_{dataset_name}_top{topk}.txt"
ORDERED_PLOT_PATH = f"ordered_combo_distribution_{dataset_name}_top{topk}.png"
UNORDERED_PLOT_PATH = f"unordered_combo_distribution_{dataset_name}_top{topk}.png"

# 总查询次数（每个查询一个组合）
total_queries = len(queries)

# 有序组合统计
print("\n有序文章组合 Top-10 频率:")
with open(ORDERED_STATS_PATH, "w") as f:
    for rank, freq in enumerate(ordered_freq_sorted[:10], 1):
        stat = f"Rank {rank}: {freq} 次"
        print(stat)
        f.write(stat + "\n")
ordered_cumulative = np.cumsum(ordered_freq_sorted) / total_queries
ordered_top10_percent = ordered_cumulative[int(0.1 * len(ordered_freq_sorted)) - 1] * 100 if len(ordered_freq_sorted) > 0 else 0
print(f"Top 10% 有序组合占总访问的 {ordered_top10_percent:.2f}%")
with open(ORDERED_STATS_PATH, "a") as f:
    f.write(f"Top 10% 有序组合占总访问的 {ordered_top10_percent:.2f}%\n")
print(f"有序组合统计保存到 {ORDERED_STATS_PATH}")

# 无序组合统计
print("\n无序文章组合 Top-10 频率:")
with open(UNORDERED_STATS_PATH, "w") as f:
    for rank, freq in enumerate(unordered_freq_sorted[:10], 1):
        stat = f"Rank {rank}: {freq} 次"
        print(stat)
        f.write(stat + "\n")
unordered_cumulative = np.cumsum(unordered_freq_sorted) / total_queries
unordered_top10_percent = unordered_cumulative[int(0.1 * len(unordered_freq_sorted)) - 1] * 100 if len(unordered_freq_sorted) > 0 else 0
print(f"Top 10% 无序组合占总访问的 {unordered_top10_percent:.2f}%")
with open(UNORDERED_STATS_PATH, "a") as f:
    f.write(f"Top 10% 无序组合占总访问的 {unordered_top10_percent:.2f}%\n")
print(f"无序组合统计保存到 {UNORDERED_STATS_PATH}")

# 绘制有序组合分布图
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(ordered_freq_sorted) + 1), ordered_freq_sorted, marker='o')
plt.xscale('log')
plt.yscale('log')
plt.title(f"有序文章组合频率分布 (Log-Log Scale) - {dataset_name.upper()} Top-{topk}")
plt.xlabel("组合排名 (由频率降序)")
plt.ylabel("访问频率")
plt.grid(True)
plt.savefig(ORDERED_PLOT_PATH)
print(f"有序组合分布图保存为 {ORDERED_PLOT_PATH}")

# 绘制无序组合分布图
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(unordered_freq_sorted) + 1), unordered_freq_sorted, marker='o')
plt.xscale('log')
plt.yscale('log')
plt.title(f"无序文章组合频率分布 (Log-Log Scale) - {dataset_name.upper()} Top-{topk}")
plt.xlabel("组合排名 (由频率降序)")
plt.ylabel("访问频率")
plt.grid(True)
plt.savefig(UNORDERED_PLOT_PATH)
print(f"无序组合分布图保存为 {UNORDERED_PLOT_PATH}")

