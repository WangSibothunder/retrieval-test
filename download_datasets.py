import os
import json
from datasets import load_dataset
from datasets.download import DownloadConfig

# 显式设置代理
os.environ["HTTP_PROXY"] = "http://127.0.0.1:49844"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:49844"

# 文件路径（与wikipead_all.py一致）
WIKI_DATA_PATH = os.path.join("wikipedia_data", "wikipedia_100k.json")
DATASET_CACHE_DIR = "dataset_cache"

# 确保目录存在
os.makedirs("wikipedia_data", exist_ok=True)
os.makedirs(DATASET_CACHE_DIR, exist_ok=True)

# 自定义下载配置，指定下载目录
download_config = DownloadConfig(
    download_desc="Downloading Wikipedia data",
    extract_compressed_file=True,
    cache_dir=os.path.join("wikipedia_data", ".cache")  # 将缓存放在 wikipedia_data 目录下
)

# 步骤1: 下载并保存Wikipedia子集（前100,000篇文章）
print("下载并保存Wikipedia子集...")
if os.path.exists(WIKI_DATA_PATH):
    print(f"本地文件 {WIKI_DATA_PATH} 已存在，跳过下载。")
else:
    wiki_dataset = load_dataset("wikipedia", "20220301.en", split="train[:100000]", trust_remote_code=True, download_config=download_config)
    documents = wiki_dataset["text"]
    wiki_data = {"text": documents}
    with open(WIKI_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(wiki_data, f, ensure_ascii=False)
    print(f"Wikipedia子集保存到 {WIKI_DATA_PATH}")

# 步骤2: 下载并保存查询数据集
datasets = {
    "mmlu": ("cais/mmlu", "all", "validation", "question"),
    "nq": ("google-research-datasets/nq_open", None, "validation", "question"),
    "hotpotqa": ("hotpot_qa", "fullwiki", "validation", "question"),
    "triviaqa": ("mandarjoshi/trivia_qa", "rc", "validation", "question")
}

for dataset_name, (dataset_path, config, split, query_key) in datasets.items():
    dataset_cache_path = os.path.join(DATASET_CACHE_DIR, f"{dataset_name}_validation.json")
    if os.path.exists(dataset_cache_path):
        print(f"本地文件 {dataset_cache_path} 已存在，跳过下载。")
    else:
        print(f"下载并保存 {dataset_name.upper()} 数据集...")
        query_dataset = load_dataset(dataset_path, config, split=split, trust_remote_code=True, download_config=download_config)
        query_data = [{"question": item[query_key]} for item in query_dataset]
        with open(dataset_cache_path, "w", encoding="utf-8") as f:
            json.dump(query_data, f, ensure_ascii=False)
        print(f"{dataset_name.upper()} 数据集保存到 {dataset_cache_path}")

print("所有数据集下载并保存完成.")