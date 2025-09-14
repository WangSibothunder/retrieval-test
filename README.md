# RAG文档热度分析工具 - hotpaper-text

一个专为RAG（检索增强生成）文档热度分布分析设计的工具，用于发现在不同查询数据集下文档访问频率分布模式。

## 🎯 项目概述

本项目实现了一个基于RAG系统的分析工具，深入研究在不同查询数据集上检索Wikipedia文档时的热度分布现象。使用了先进的BAAI/bge-large-en-v1.5嵌入模型和FAISS HNSW索引技术进行高效的向量检索和统计分析。

## ✨ 核心功能

### 基础分析功能
- 🗂️ **多数据集支持**: 支持MMLU、Natural Questions、HotpotQA、TriviaQA等问答数据集
- 📊 **频率分析**: 统计文档访问频率分布，识别"长尾分布"模式
- 🔢 **灵活Top-K**: 支持不同的检索数量配置
- 📈 **可视化**: 生成Log-Log尺度的频率分布图表
- ⚡ **高效检索**: 使用FAISS HNSW索引实现快速的向量相似度检索

### 高级分析功能  
- 🔗 **组合分析**: 分析文档组合的有序和无序访问模式
- 📋 **多数据集对比**: 支持多数据集间的对比分析
- 🌐 **N-gram序列分析**: 分析连续2,3,4个文档对的频率分布
- 🏗️ **HNSW层级分析**: 分析热门文档在HNSW索引中的层级分布
- 📊 **综合仪表板**: 提供多维度的可视化对比工具

## 🛠️ 技术栈

| 组件 | 版本 | 说明 |
|:---|:---|:---|
| **嵌入模型** | BAAI/bge-large-en-v1.5 | 1024维，专为RAG优化的文本嵌入模型 |
| **向量数据库** | FAISS HNSW | 高效的近似最近邻搜索索引 |
| **知识库** | Wikipedia | Wikipedia子集，约100k文档 |
| **可视化** | Matplotlib | 生成Log-Log尺度的频率分布图 |
| **数据处理** | Pandas, NumPy | 数据加载、处理和数值计算 |
| **数据集加载** | datasets | HuggingFace datasets库 |

## 📋 Python文件分类说明

### 📊 数据集类型说明

本项目支持两种不同规模的Wikipedia数据集：

- **Wikipedia3.2k**: 约3200个文档的小规模数据集，适合快速实验和概念验证
- **Wikipedia100k**: 约100,000个文档的大规模数据集，适合深入研究和性能分析

**重要提示**: `wikipead_all.py` 专门用于 **Wikipedia100k** 数据集分析，其他所有脚本都用于 **Wikipedia3.2k** 数据集分析。

### 🔄 核心分析模块

#### Wikipedia3.2k 数据集分析模块

##### 1. `hot.py` - 基础文档频率分析
- **数据集**: Wikipedia3.2k (约3200文档)
- **功能**: 分析单个数据集下文档的检索热度分布
- **输入**: 数据集名称（mmlu/nq/hotpotqa/triviaqa）、Top-K参数
- **输出**: 频率统计文件、Log-Log分布图
- **推荐配置**: top1, top5, top10
- **特色**: 最基础的热度分析，适合初学者

##### 2. `hotpair.py` - 文档组合分析
- **数据集**: Wikipedia3.2k (约3200文档)
- **功能**: 分析文档组合的有序和无序访问模式
- **输入**: 数据集名称、Top-K参数
- **输出**: 有序/无序组合统计文件、对比分布图
- **推荐配置**: top3, top5, top10
- **特色**: 揭示检索顺序对结果的影响

##### 3. `hot_pair_in_seq.py` - 序列中的文档对分析
- **数据集**: Wikipedia3.2k (约3200文档)
- **功能**: 分析检索序列中的文档对模式
- **输入**: 数据集名称、Top-K参数
- **输出**: 序列文档对统计结果
- **推荐配置**: top10
- **特色**: 专注于相邻文档的相关性分析

##### 4. `hotpaper_HNSWnode.py` - HNSW节点层级分析
- **数据集**: Wikipedia3.2k (约3200文档)
- **功能**: 专门分析HNSW索引中节点的层级分布
- **输入**: 数据集名称、Top-K参数
- **输出**: HNSW层级统计和可视化结果
- **推荐配置**: top10
- **特色**: 深入了解索引结构与文档热度的关系

#### Wikipedia100k 数据集分析模块

##### 5. `wikipead_all.py` - 综合多维度分析 ⭐
- **数据集**: Wikipedia100k (约100,000文档)
- **功能**: 最全面的分析工具，包含所有分析维度
- **输入**: 数据集名称、Top-K参数
- **输出**: 文档频率、N-gram序列、HNSW层级分析等
- **推荐配置**: top10, top16, top32
- **特色**: 一站式分析工具，功能最全面，适合大规模数据研究

### 📊 可视化工具模块

#### 6. `draw_chart.py` - 基础数据集对比图表
- **功能**: 绘制四个数据集的基础对比图表
- **输入**: 无命令行参数，自动读取统计文件
- **输出**: `database_comparison_chart.png`
- **特色**: 简单直观的数据集对比

#### 7. `draw_combo_chart.py` - 组合分析对比图表
- **功能**: 绘制有序/无序组合的对比图表
- **输入**: 无命令行参数，读取combo统计文件
- **输出**: `combo_comparison_chart.png`、`combo_side_by_side_chart.png`
- **特色**: 上下/左右对比布局，便于对比分析

#### 8. `draw_comprehensive_chart.py` - 综合对比仪表板
- **功能**: 整合所有分析维度的综合图表
- **输入**: 无命令行参数，自动收集数据
- **输出**: 多种综合对比图表
- **特色**: 一站式可视化方案，功能最全面

#### 9. `draw_single_dataset_chart.py` - 单数据集详细分析
- **功能**: 为指定数据集生成详细的分析仪表板
- **输入**: `--dataset`、`--topk`参数
- **输出**: `{dataset}_top{k}_dashboard.png`
- **特色**: 专注于单个数据集的深入分析

#### 10. `draw_wikipead_all_chart.py` - Wikipead_all专用图表
- **功能**: 为`wikipead_all.py`的输出生成专用图表
- **输入**: `--dataset`、`--topk`参数（可选all）
- **输出**: 多种专业化分析图表
- **特色**: 支持N-gram和HNSW层级分析的可视化

#### 11. `draw_hnsw_level_chart.py` - HNSW层级专用图表
- **功能**: 专门绘制HNSW层级分析的图表
- **输入**: 无命令行参数，读取HNSW统计文件
- **输出**: HNSW层级分布相关图表
- **特色**: 专业化的索引结构分析

### 📈 分析报告模块

#### 12. `generate_wikipead_analysis_report.py` - 智能分析报告生成器
- **功能**: 生成详细的文本分析报告
- **输入**: `--dataset`、`--topk`参数（可选all）
- **输出**: `wikipead_analysis_report.txt`
- **特色**: 包含统计摘要、趋势分析、优化建议

### 🛠️ 工具和配置模块

#### 13. `dateset.py` - 数据集加载工具
- **功能**: 预加载和缓存数据集
- **输入**: 无命令行参数
- **输出**: 数据集缓存文件
- **特色**: 提前加载数据集，提高后续分析效率

#### 14. `download_datasets.py` - 数据集下载工具
- **功能**: 批量下载和缓存所有所需数据集
- **输入**: 无命令行参数
- **输出**: 在`dataset_cache/`目录下生成缓存文件
- **特色**: 一次性下载所有数据集，避免重复下载

#### 15. `config.py` - 项目配置文件
- **功能**: 统一管理项目的配置参数
- **内容**: 颜色配置、字体设置、文件路径等
- **特色**: 集中化配置管理，便于维护和定制

## 📁 项目文件结构

```
hotpaper-text/
├── 📋 核心分析模块
│   ├── hot.py                        # 基础文档频率分析
│   ├── hotpair.py                    # 文档组合分析
│   ├── wikipead_all.py               # 综合多维度分析⭐
│   ├── hot_pair_in_seq.py            # 序列文档对分析
│   └── hotpaper_HNSWnode.py          # HNSW节点层级分析
├── 📊 可视化工具模块
│   ├── draw_chart.py                 # 基础数据集对比图表
│   ├── draw_combo_chart.py           # 组合分析对比图表
│   ├── draw_comprehensive_chart.py   # 综合对比仪表板⭐
│   ├── draw_single_dataset_chart.py  # 单数据集详细分析
│   ├── draw_wikipead_all_chart.py    # Wikipead_all专用图表
│   └── draw_hnsw_level_chart.py      # HNSW层级专用图表
├── 📈 分析报告模块
│   └── generate_wikipead_analysis_report.py # 智能分析报告生成器
├── 🛠️ 工具和配置模块
│   ├── dateset.py                    # 数据集加载工具
│   ├── download_datasets.py          # 数据集下载工具
│   └── config.py                     # 项目配置文件
├── 📝 文档和配置
│   ├── README.md                     # 项目主文档
│   ├── WIKIPEAD_VISUALIZATION_GUIDE.md # 可视化工具使用指南
│   ├── requirements.txt              # Python依赖列表
│   ├── commands.txt                  # 命令示例集合
│   └── .gitignore                    # Git忽略文件配置
├── 📂 数据目录
│   ├── data/stats/                   # 统计结果文件目录
│   │   ├── freq_stats_*.txt          # 文档频率统计
│   │   ├── ordered_combo_stats_*.txt # 有序组合统计
│   │   ├── unordered_combo_stats_*.txt # 无序组合统计
│   │   ├── ngram_stats_*.txt         # N-gram序列统计
│   │   └── high_level_stats_*.txt    # HNSW高层节点统计
│   └── output/charts/                # 图表输出文件目录
│       ├── hot_docs_distribution_*.png # 文档频率分布图
│       ├── ngram_distribution_*.png    # N-gram分布图
│       ├── high_level_distribution_*.png # HNSW层级分布图
│       ├── *_dashboard.png           # 各种仪表板图表
│       └── wikipead_analysis_report.txt # 分析报告
└── 💾 缓存目录（被.gitignore排除）
    ├── wikipedia_data/               # Wikipedia数据缓存
    ├── dataset_cache/                # 查询数据集缓存
    ├── doc_embeddings*.npy           # 文档嵌入文件
    └── hnsw_index*.bin               # FAISS HNSW索引文件
```

⭐ 表示推荐优先使用的工具

## 🚀 快速开始

### 1. 环境配置
```bash
pip install -r requirements.txt
```

### 2. 数据准备（可选）
```bash
# 预先下载所有数据集，提高后续运行效率
python download_datasets.py

# 或者使用基础数据集加载工具
python dateset.py
```

### 3. Wikipedia3.2k 数据集分析命令

#### 基础文档频率分析 (hot.py)
```bash
# 为所有数据集进行基础热度分析
# 推荐配置: top1, top5, top10

# MMLU 数据集
python hot.py --dataset mmlu --topk 1
python hot.py --dataset mmlu --topk 5
python hot.py --dataset mmlu --topk 10

# Natural Questions 数据集
python hot.py --dataset nq --topk 1
python hot.py --dataset nq --topk 5
python hot.py --dataset nq --topk 10

# HotpotQA 数据集
python hot.py --dataset hotpotqa --topk 1
python hot.py --dataset hotpotqa --topk 5
python hot.py --dataset hotpotqa --topk 10

# TriviaQA 数据集
python hot.py --dataset triviaqa --topk 1
python hot.py --dataset triviaqa --topk 5
python hot.py --dataset triviaqa --topk 10
```

#### 文档组合分析 (hotpair.py)
```bash
# 分析文档组合的有序和无序访问模式
# 推荐配置: top3, top5, top10

# MMLU 数据集
python hotpair.py --dataset mmlu --topk 3
python hotpair.py --dataset mmlu --topk 5
python hotpair.py --dataset mmlu --topk 10

# Natural Questions 数据集
python hotpair.py --dataset nq --topk 3
python hotpair.py --dataset nq --topk 5
python hotpair.py --dataset nq --topk 10

# HotpotQA 数据集
python hotpair.py --dataset hotpotqa --topk 3
python hotpair.py --dataset hotpotqa --topk 5
python hotpair.py --dataset hotpotqa --topk 10

# TriviaQA 数据集
python hotpair.py --dataset triviaqa --topk 3
python hotpair.py --dataset triviaqa --topk 5
python hotpair.py --dataset triviaqa --topk 10
```

#### 序列文档对分析 (hot_pair_in_seq.py)
```bash
# 分析检索序列中的文档对模式
# 推荐配置: top10

python hot_pair_in_seq.py --dataset mmlu --topk 10
python hot_pair_in_seq.py --dataset nq --topk 10
python hot_pair_in_seq.py --dataset hotpotqa --topk 10
python hot_pair_in_seq.py --dataset triviaqa --topk 10
```

#### HNSW节点层级分析 (hotpaper_HNSWnode.py)
```bash
# 分析HNSW索引中节点的层级分布
# 推荐配置: top10

python hotpaper_HNSWnode.py --dataset mmlu --topk 10
python hotpaper_HNSWnode.py --dataset nq --topk 10
python hotpaper_HNSWnode.py --dataset hotpotqa --topk 10
python hotpaper_HNSWnode.py --dataset triviaqa --topk 10
```

### 4. Wikipedia100k 数据集综合分析命令

#### 综合多维度分析 (wikipead_all.py) ⭐
```bash
# 最全面的分析工具，适合大规模数据研究
# 推荐配置: top10, top16, top32

# MMLU 数据集
python wikipead_all.py --dataset mmlu --topk 10
python wikipead_all.py --dataset mmlu --topk 16
python wikipead_all.py --dataset mmlu --topk 32

# Natural Questions 数据集
python wikipead_all.py --dataset nq --topk 10
python wikipead_all.py --dataset nq --topk 16
python wikipead_all.py --dataset nq --topk 32

# HotpotQA 数据集
python wikipead_all.py --dataset hotpotqa --topk 10
python wikipead_all.py --dataset hotpotqa --topk 16
python wikipead_all.py --dataset hotpotqa --topk 32

# TriviaQA 数据集
python wikipead_all.py --dataset triviaqa --topk 10
python wikipead_all.py --dataset triviaqa --topk 16
python wikipead_all.py --dataset triviaqa --topk 32
```

### 5. 可视化命令

#### 自动化图表生成（无需参数）
```bash
# 基础数据集对比图表
python draw_chart.py

# 组合分析对比图表
python draw_combo_chart.py

# 综合对比仪表板⭐
python draw_comprehensive_chart.py

# HNSW层级分析图表
python draw_hnsw_level_chart.py
```

#### 可配置图表生成
```bash
# 单数据集详细分析
python draw_single_dataset_chart.py --dataset mmlu --topk 10

# Wikipead_all专用图表
python draw_wikipead_all_chart.py --dataset all --topk all
python draw_wikipead_all_chart.py --dataset mmlu --topk 10

# 智能分析报告生成器
python generate_wikipead_analysis_report.py --dataset all --topk all
python generate_wikipead_analysis_report.py --dataset nq --topk 10
```

## 📋 完整命令参考

### Wikipedia3.2k 数据集命令

#### 基础文档频率分析 (hot.py)
```bash
# 数据准备
python dateset.py

# MMLU 数据集分析
python hot.py --dataset mmlu --topk 1
python hot.py --dataset mmlu --topk 5
python hot.py --dataset mmlu --topk 10

# Natural Questions 数据集分析
python hot.py --dataset nq --topk 1
python hot.py --dataset nq --topk 5
python hot.py --dataset nq --topk 10

# HotpotQA 数据集分析
python hot.py --dataset hotpotqa --topk 1
python hot.py --dataset hotpotqa --topk 5
python hot.py --dataset hotpotqa --topk 10

# TriviaQA 数据集分析
python hot.py --dataset triviaqa --topk 1
python hot.py --dataset triviaqa --topk 5
python hot.py --dataset triviaqa --topk 10
```

#### 文档组合分析 (hotpair.py)
```bash
# MMLU 数据集组合分析
python hotpair.py --dataset mmlu --topk 3
python hotpair.py --dataset mmlu --topk 5
python hotpair.py --dataset mmlu --topk 10

# Natural Questions 数据集组合分析
python hotpair.py --dataset nq --topk 3
python hotpair.py --dataset nq --topk 5
python hotpair.py --dataset nq --topk 10

# HotpotQA 数据集组合分析
python hotpair.py --dataset hotpotqa --topk 3
python hotpair.py --dataset hotpotqa --topk 5
python hotpair.py --dataset hotpotqa --topk 10

# TriviaQA 数据集组合分析
python hotpair.py --dataset triviaqa --topk 3
python hotpair.py --dataset triviaqa --topk 5
python hotpair.py --dataset triviaqa --topk 10
```

#### 专项分析命令
```bash
# 序列中的文档对分析
python hot_pair_in_seq.py --dataset mmlu --topk 10
python hot_pair_in_seq.py --dataset nq --topk 10
python hot_pair_in_seq.py --dataset hotpotqa --topk 10
python hot_pair_in_seq.py --dataset triviaqa --topk 10

# HNSW节点层级分析
python hotpaper_HNSWnode.py --dataset mmlu --topk 10
python hotpaper_HNSWnode.py --dataset nq --topk 10
python hotpaper_HNSWnode.py --dataset hotpotqa --topk 10
python hotpaper_HNSWnode.py --dataset triviaqa --topk 10
```

### Wikipedia100k 数据集命令

#### 综合多维度分析 (wikipead_all.py)
```bash
# 数据准备（如需要）
python download_datasets.py

# MMLU 数据集全面分析
python wikipead_all.py --dataset mmlu --topk 10
python wikipead_all.py --dataset mmlu --topk 16
python wikipead_all.py --dataset mmlu --topk 32

# Natural Questions 数据集全面分析
python wikipead_all.py --dataset nq --topk 10
python wikipead_all.py --dataset nq --topk 16
python wikipead_all.py --dataset nq --topk 32

# HotpotQA 数据集全面分析
python wikipead_all.py --dataset hotpotqa --topk 10
python wikipead_all.py --dataset hotpotqa --topk 16
python wikipead_all.py --dataset hotpotqa --topk 32

# TriviaQA 数据集全面分析
python wikipead_all.py --dataset triviaqa --topk 10
python wikipead_all.py --dataset triviaqa --topk 16
python wikipead_all.py --dataset triviaqa --topk 32
```

### 可视化和报告生成命令

#### 自动化图表生成（无需参数）
```bash
# 基础数据集对比图表
python draw_chart.py

# 组合分析对比图表
python draw_combo_chart.py

# 综合对比仪表板
python draw_comprehensive_chart.py

# HNSW层级分析图表
python draw_hnsw_level_chart.py
```

#### 可配置图表生成
```bash
# 单数据集详细分析图表
python draw_single_dataset_chart.py --dataset mmlu --topk 10
python draw_single_dataset_chart.py --dataset nq --topk 10
python draw_single_dataset_chart.py --dataset hotpotqa --topk 10
python draw_single_dataset_chart.py --dataset triviaqa --topk 10

# Wikipedia100k专用图表
python draw_wikipead_all_chart.py --dataset all --topk all
python draw_wikipead_all_chart.py --dataset mmlu --topk 10

# 智能分析报告生成
python generate_wikipead_analysis_report.py --dataset all --topk all
python generate_wikipead_analysis_report.py --dataset mmlu --topk 10
```

### 数据集参数说明
- `--dataset`: 选择数据集
  - `mmlu`: MMLU (大规模多任务语言理解)
  - `nq`: Natural Questions (自然问题回答)
  - `hotpotqa`: HotpotQA (多跳推理问答)
  - `triviaqa`: TriviaQA (琐碎知识问答)
  - `all`: 所有数据集（仅部分脚本支持）

- `--topk`: 检索数量
  - **Wikipedia3.2k推荐**: 1, 3, 5, 10
  - **Wikipedia100k推荐**: 10, 16, 32
  - `all`: 所有配置（仅部分脚本支持）

## 📝 输出文件详细说明

### 📊 统计文件 (`data/stats/` 目录)

#### 基础频率统计
- `freq_stats_{dataset}_top{k}.txt`: 文档频率统计
  - 包含: Top-10热门文档排名、频率、Top 10%集中度
  - 示例: `freq_stats_mmlu_top10.txt`

#### 组合模式统计
- `ordered_combo_stats_{dataset}_top{k}.txt`: 有序组合统计
  - 包含: 检索顺序相关的组合模式
- `unordered_combo_stats_{dataset}_top{k}.txt`: 无序组合统计
  - 包含: 忽略顺序的文档组合模式

#### N-gram序列统计
- `ngram_stats_n{n}_{dataset}_top{k}.txt`: N-gram序列统计
  - n=2,3,4: 分别表示2-gram、3-gram、4-gram
  - 包含: 连续文档对的频率分布

#### HNSW层级统计
- `high_level_stats_{dataset}_top{k}.txt`: HNSW高层节点统计
  - 包含: 热门文档在HNSW索引中的层级信息
  - 包含: 高层节点占比、层级分布等

### 📈 图表文件 (`output/charts/` 目录)

#### 基础分布图表
- `hot_docs_distribution_{dataset}_top{k}.png`: 文档频率Log-Log分布图
- `ordered_combo_distribution_{dataset}_top{k}.png`: 有序组合分布图
- `unordered_combo_distribution_{dataset}_top{k}.png`: 无序组合分布图

#### N-gram分析图表
- `ngram_distribution_n{n}_{dataset}_top{k}.png`: N-gram分布图
  - 示例: `ngram_distribution_n2_mmlu_top10.png`

#### HNSW层级图表
- `high_level_distribution_{dataset}_top{k}.png`: 高层节点分布图

#### 综合对比图表
- `database_comparison_chart.png`: 基础数据集对比图
- `combo_comparison_chart.png`: 组合分析对比图
- `combo_side_by_side_chart.png`: 组合分析并排对比图
- `comprehensive_dashboard.png`: 综合对比仪表板
- `{dataset}_top{k}_dashboard.png`: 单数据集详细仪表板

#### 分析报告
- `wikipead_analysis_report.txt`: 智能生成的文本分析报告

### 🗂️ 系统文件（被.gitignore排除）
- `doc_embeddings.npy` / `doc_embeddings_100k.npy`: Wikipedia文档嵌入文件
- `hnsw_index.bin` / `hnsw_index_100k.bin`: FAISS HNSW索引文件
- `wikipedia_data/`: Wikipedia数据缓存目录
- `dataset_cache/`: 查询数据集缓存目录

## ?? 配置说明

### 模型配置
项目支持本地缓存模型，优先使用顺序：
1. `L:\huggingface\cache\hub\models--BAAI--bge-large-en-v1.5`
2. `L:\huggingface\cache\hub`
3. 本地模型目录
4. 在线下载

### 数据集支持
- **MMLU**: 大规模多任务语言理解
- **Natural Questions**: 自然问题回答
- **HotpotQA**: 多跳推理问答
- **TriviaQA**: 琐事问答

## ? 使用技巧

### 1. 批量分析
```bash
# 分析所有数据集的top-1, top-5, top-10
for dataset in mmlu nq hotpotqa triviaqa; do
    for topk in 1 5 10; do
        python wikipead_all.py --dataset $dataset --topk $topk
    done
done

# 生成综合图表
python draw_comprehensive_chart.py
```

### 2. 内存优化
- 对于内存有限的环境，建议先从小的top-k值开始
- Wikipedia数据和嵌入文件会被缓存，首次运行时间较长

### 3. 图表定制
- 修改 `config.py` 中的颜色和尺寸配置
- 图表支持中文显示（SimHei、Microsoft YaHei字体）

## ? 分析结果解读

### 文档频率分布
- **Log-Log图上的直线**: 表明服从幂律分布
- **Top 10%占比**: 热门文档的集中程度

### N-gram分析
- **高频N-gram**: 揭示检索序列的模式
- **分布特征**: 不同长度序列的集中度差异

### HNSW层级分析
- **高层节点占比**: 热门文档在索引结构中的位置
- **层级分布**: 索引效率与文档热度的关系

## ? 常见问题

### 1. 编码问题
如果遇到中文显示问题，确保已安装中文字体：
- Windows: SimHei、Microsoft YaHei
- Linux: 安装中文字体包

### 2. 内存不足
- 减少批次大小：修改 `EMBEDDING_BATCH_SIZE`
- 使用更小的Wikipedia子集
- 分批处理不同数据集

### 3. 模型下载
- 配置本地缓存路径避免重复下载
- 使用代理或镜像加速下载

## ? 技术栈
- **嵌入模型**: BAAI/bge-large-en-v1.5
- **向量检索**: FAISS HNSW
- **数据可视化**: Matplotlib
- **数据处理**: NumPy, Pandas
- **数据集**: HuggingFace Datasets

## ? 贡献
欢迎提交Issue和Pull Request来改进这个工具！

## ? 许可证
MIT License
```

## 📊 使用技巧和批量分析

### 1. 推荐的实验流程

#### 新手入门流程（Wikipedia3.2k）
```bash
# 步骤1: 数据准备
python dateset.py

# 步骤2: 从简单的基础分析开始
python hot.py --dataset mmlu --topk 1
python hot.py --dataset mmlu --topk 5

# 步骤3: 进阶组合分析
python hotpair.py --dataset mmlu --topk 3
python hotpair.py --dataset mmlu --topk 5

# 步骤4: 生成可视化报告
python draw_chart.py
python draw_combo_chart.py
```

#### 全面研究流程（Wikipedia100k）
```bash
# 步骤1: 数据准备
python download_datasets.py

# 步骤2: 全面分析（时间较长）
python wikipead_all.py --dataset mmlu --topk 10
python wikipead_all.py --dataset mmlu --topk 16
python wikipead_all.py --dataset mmlu --topk 32

# 步骤3: 生成综合报告
python draw_wikipead_all_chart.py --dataset mmlu --topk 10
python generate_wikipead_analysis_report.py --dataset mmlu --topk 10
```

### 2. 批量分析脚本

#### Wikipedia3.2k 数据集全量分析
```bash
# 所有数据集的基础热度分析 (Windows PowerShell)
foreach ($dataset in @('mmlu', 'nq', 'hotpotqa', 'triviaqa')) {
    foreach ($topk in @(1, 5, 10)) {
        python hot.py --dataset $dataset --topk $topk
    }
}

# 所有数据集的组合分析 (Windows PowerShell)
foreach ($dataset in @('mmlu', 'nq', 'hotpotqa', 'triviaqa')) {
    foreach ($topk in @(3, 5, 10)) {
        python hotpair.py --dataset $dataset --topk $topk
    }
}

# Linux/macOS 版本
for dataset in mmlu nq hotpotqa triviaqa; do
    for topk in 1 5 10; do
        python hot.py --dataset $dataset --topk $topk
    done
done

for dataset in mmlu nq hotpotqa triviaqa; do
    for topk in 3 5 10; do
        python hotpair.py --dataset $dataset --topk $topk
    done
done
```

#### Wikipedia100k 数据集全量分析
```bash
# 所有数据集的Wikipedia100k分析 (Windows PowerShell)
foreach ($dataset in @('mmlu', 'nq', 'hotpotqa', 'triviaqa')) {
    foreach ($topk in @(10, 16, 32)) {
        python wikipead_all.py --dataset $dataset --topk $topk
    }
}

# Linux/macOS 版本
for dataset in mmlu nq hotpotqa triviaqa; do
    for topk in 10 16 32; do
        python wikipead_all.py --dataset $dataset --topk $topk
    done
done
```

### 3. 结果整理和报告生成
```bash
# 生成所有基础对比图表
python draw_chart.py
python draw_combo_chart.py
python draw_comprehensive_chart.py
python draw_hnsw_level_chart.py

# 生成每个数据集的详细报告 (Windows PowerShell)
foreach ($dataset in @('mmlu', 'nq', 'hotpotqa', 'triviaqa')) {
    python draw_single_dataset_chart.py --dataset $dataset --topk 10
    python generate_wikipead_analysis_report.py --dataset $dataset --topk 10
}

# 生成全局综合报告
python draw_wikipead_all_chart.py --dataset all --topk all
python generate_wikipead_analysis_report.py --dataset all --topk all
```

### 4. 性能优化建议

#### 内存优化
- **小内存环境** (<8GB): 优先使用Wikipedia3.2k数据集，从小的top-k值开始
- **大内存环境** (>16GB): 可直接使用Wikipedia100k数据集进行全面分析
- **缓存优化**: Wikipedia数据和嵌入文件会被缓存，首次运行时间较长

#### 时间估算
- **Wikipedia3.2k**: 每个分析任务约5-15分钟
- **Wikipedia100k**: 每个分析任务约30-60分钟
- **首次运行**: 需要下载模型和数据集，额外增加20-30分钟

### 5. 常用组合命令

#### 快速比较不同数据集
```bash
# 比较所有数据集的top5结果
python hot.py --dataset mmlu --topk 5
python hot.py --dataset nq --topk 5
python hot.py --dataset hotpotqa --topk 5
python hot.py --dataset triviaqa --topk 5
python draw_chart.py
```

#### 深入分析单个数据集
```bash
# 对MMLU数据集进行全面分析
python hot.py --dataset mmlu --topk 1
python hot.py --dataset mmlu --topk 5
python hot.py --dataset mmlu --topk 10
python hotpair.py --dataset mmlu --topk 3
python hotpair.py --dataset mmlu --topk 5
python hotpair.py --dataset mmlu --topk 10
python hot_pair_in_seq.py --dataset mmlu --topk 10
python hotpaper_HNSWnode.py --dataset mmlu --topk 10
python draw_single_dataset_chart.py --dataset mmlu --topk 10
```

### 6. 图表定制和配置
- **修改颜色配置**: 编辑 `config.py` 中的颜色和尺寸配置
- **中文字体支持**: 图表支持中文显示（SimHei、Microsoft YaHei字体）
- **输出格式**: 所有图表默认以PNG格式输出，可通过修改代码调整
