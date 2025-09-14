# RAG文档热度分析工具 - 升级版

这个项目实现了一个基于RAG（Retrieval-Augmented Generation）系统的分析工具，用于研究在不同查询数据集下Wikipedia文档的热度分布情况，现在包含了更全面的分析功能和可视化工具。

## ? 新功能特性

### 1. 多维度分析
- **文档频率分析**: 分析热门文档的分布模式
- **N-gram序列分析**: 分析2-gram、3-gram、4-gram的检索模式
- **HNSW层级分析**: 分析热门文档在HNSW索引中的层级分布

### 2. 丰富的可视化工具
- **综合对比图表**: 跨数据集的多维度对比分析
- **单数据集详细仪表板**: 深入分析单个数据集的各项指标
- **Log-Log分布图**: 揭示长尾分布特征

### 3. 改进的文件结构
```
├── data/stats/          # 所有统计文件
├── output/charts/       # 所有图表文件
├── wikipedia_data/      # Wikipedia数据缓存
└── dataset_cache/       # 查询数据集缓存
```

## ? 快速开始

### 1. 环境配置
```bash
pip install -r requirements.txt
```

### 2. 运行示例
```bash
python run_example.py
```

### 3. 基本分析
```bash
# 对MMLU数据集进行top-5分析
python wikipead_all.py --dataset mmlu --topk 5

# 对Natural Questions数据集进行top-10分析
python wikipead_all.py --dataset nq --topk 10
```

## ? 可视化工具

### 1. 综合对比图表
```bash
python draw_comprehensive_chart.py
```
生成所有数据集的多维度对比图表，包括：
- 文档频率分布对比
- N-gram分布对比  
- HNSW高层节点分布对比
- 综合仪表板

### 2. 单数据集详细分析
```bash
python draw_single_dataset_chart.py --dataset mmlu --topk 5
```
为指定数据集生成详细的分析仪表板，包括：
- Top-10文档频率分布
- HNSW层级分布
- 各种N-gram分布
- 综合指标对比

## ? 输出文件说明

### 统计文件 (data/stats/)
- `freq_stats_{dataset}_top{k}.txt`: 文档频率统计
- `ngram_stats_n{n}_{dataset}_top{k}.txt`: N-gram统计  
- `high_level_stats_{dataset}_top{k}.txt`: 高层节点统计

### 图表文件 (output/charts/)
- `hot_docs_distribution_{dataset}_top{k}.png`: 文档频率Log-Log图
- `ngram_distribution_n{n}_{dataset}_top{k}.png`: N-gram分布图
- `high_level_distribution_{dataset}_top{k}.png`: 高层节点分布图
- `{dataset}_top{k}_dashboard.png`: 单数据集详细仪表板
- `comprehensive_dashboard.png`: 综合对比仪表板

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