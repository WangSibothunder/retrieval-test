# hotpaper-text

一个基于RAG（检索增强生成）的热门文章分布分析工具，用于分析不同数据集中文档检索的频率分布模式。

## 项目描述

本项目实现了一个完整的RAG系统，用于分析在不同查询数据集上检索Wikipedia文档时的热门度分布。它使用了BAAI/bge-large-en-v1.5嵌入模型和FAISS HNSW索引来进行高效的语义搜索。

## 主要功能

- ? **多数据集支持**: 支持MMLU、Natural Questions、HotpotQA、TriviaQA等常见问答数据集
- ? **频率分析**: 分析检索到的文档频率分布，识别"热门文章"现象
- ? **可配置Top-K**: 支持不同的检索数量设置
- ? **可视化**: 生成Log-Log尺度的频率分布图
- ? **高效检索**: 使用FAISS HNSW索引进行快速相似性搜索

## 技术栈

- **嵌入模型**: BAAI/bge-large-en-v1.5 (1024维，专为RAG优化)
- **搜索引擎**: FAISS HNSW索引
- **知识库**: rag-mini-wikipedia (Wikipedia子集)
- **可视化**: Matplotlib
- **数据处理**: Pandas, NumPy

## 安装与使用

### 环境要求

```bash
pip install -r requirements.txt
```

### 基本使用

```bash
# 使用MMLU数据集，检索Top-1文档
python hot.py --dataset mmlu --topk 1

# 使用Natural Questions数据集，检索Top-5文档
python hot.py --dataset nq --topk 5

# 使用HotpotQA数据集，检索Top-10文档
python hot.py --dataset hotpotqa --topk 10

# 使用TriviaQA数据集，检索Top-1文档
python hot.py --dataset triviaqa --topk 1
```

### 支持的数据集

- `mmlu`: MMLU (Massive Multitask Language Understanding)
- `nq`: Natural Questions
- `hotpotqa`: HotpotQA
- `triviaqa`: TriviaQA

### 输出文件

运行完成后会生成以下文件：

- `hot_docs_distribution_{dataset}_top{k}.png`: 频率分布可视化图
- `freq_stats_{dataset}_top{k}.txt`: 详细的频率统计信息
- `doc_embeddings.npy`: Wikipedia文档嵌入（首次运行生成）
- `hnsw_index.bin`: FAISS HNSW搜索索引（首次运行生成）

## 分析结果

该工具可以帮助您：

1. **发现热门文章现象**: 识别在RAG系统中被频繁检索的Wikipedia文章
2. **分析分布模式**: 通过Log-Log图观察是否符合幂律分布
3. **评估检索多样性**: 了解检索结果的集中程度
4. **优化RAG系统**: 基于热门度分布调整检索策略

## 示例命令

参考 `commands.txt` 文件中的示例命令：

```bash
python hot.py --dataset mmlu --topk 1
python hot.py --dataset nq --topk 5
python hot.py --dataset hotpotqa --topk 10
python hot.py --dataset triviaqa --topk 1
```

## 注意事项

- 首次运行需要下载和处理Wikipedia数据集，可能需要较长时间
- 嵌入文件和索引文件较大（~25MB），首次生成后会本地缓存
- 推荐在有足够内存的环境中运行（建议8GB+）

## 许可证

本项目采用MIT许可证 - 详见LICENSE文件。