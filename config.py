# RAG文档热度分析工具配置文件

# 模型配置
MODEL_NAME = "BAAI/bge-large-en-v1.5"
LOCAL_MODEL_PATHS = [
    r"L:\huggingface\cache\hub\models--BAAI--bge-large-en-v1.5",
    r"L:\huggingface\cache\hub",
    "./models/BAAI_bge-large-en-v1.5",
    "./BAAI_bge-large-en-v1.5",
    "models/BAAI_bge-large-en-v1.5",
    "BAAI_bge-large-en-v1.5"
]

# 数据路径配置
WIKIPEDIA_DATA_PATH = "wikipedia_data/wikipedia_100k.json"
EMBEDDINGS_PATH = "doc_embeddings.npy"
INDEX_PATH = "hnsw_index.bin"
DATASET_CACHE_DIR = "dataset_cache"

# 输出路径配置
STATS_OUTPUT_DIR = "data/stats"
CHARTS_OUTPUT_DIR = "output/charts"

# FAISS配置
HNSW_M = 32  # HNSW连接数
EMBEDDING_BATCH_SIZE = 100

# 支持的数据集
SUPPORTED_DATASETS = ["mmlu", "nq", "hotpotqa", "triviaqa"]

# 数据集配置映射
DATASET_CONFIG = {
    "mmlu": {
        "huggingface_name": "cais/mmlu",
        "huggingface_config": "all",
        "query_key": "question"
    },
    "nq": {
        "huggingface_name": "google-research-datasets/nq_open",
        "huggingface_config": None,
        "query_key": "question"
    },
    "hotpotqa": {
        "huggingface_name": "hotpot_qa",
        "huggingface_config": "fullwiki",
        "query_key": "question"
    },
    "triviaqa": {
        "huggingface_name": "mandarjoshi/trivia_qa",
        "huggingface_config": "rc",
        "query_key": "question"
    }
}

# 图表配置
FIGURE_SIZE = (12, 8)
DPI = 300
COLORS = {
    "top1": "#FF6B6B",
    "top5": "#4ECDC4", 
    "top10": "#45B7D1",
    "ngram2": "#45B7D1",
    "ngram3": "#96CEB4",
    "ngram4": "#FFEAA7"
}