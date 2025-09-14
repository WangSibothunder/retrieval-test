# WIKIPEAD_ALL.PY 数据可视化工具使用指南

本目录包含两个专门为 `wikipead_all.py` 输出数据设计的可视化工具。

## 工具概览

### 1. `draw_wikipead_all_chart.py` - 图表绘制工具
专门为 `wikipead_all.py` 的输出数据生成各种可视化图表

### 2. `generate_wikipead_analysis_report.py` - 分析报告生成器
生成详细的文本分析报告，包含统计摘要、趋势分析和优化建议

## 主要功能特性

### 图表绘制功能
- **文档频率分布对比图**: 显示Top 10%文档在不同数据集的集中度
- **N-gram序列分析图**: 分析连续2,3,4个文档对的频率分布
- **HNSW高层节点分布图**: 显示热门文档在HNSW索引中的层级分布
- **详细层级分布图**: 展示Top-10热门文档的具体层级分布
- **综合对比仪表板**: 将所有分析结果整合在一个综合图表中

### 分析报告功能
- **统计摘要**: 关键指标的数值统计
- **分布特征分析**: 集中度比值、变异系数、幂律相关性等
- **比较分析**: 跨数据集的对比洞察
- **优化建议**: 基于分析结果的实用建议

## 使用方法

### 基本使用

```bash
# 生成所有数据集和配置的图表
python draw_wikipead_all_chart.py

# 生成所有数据集和配置的分析报告
python generate_wikipead_analysis_report.py
```

### 指定数据集和配置

```bash
# 仅分析MMLU数据集的所有配置
python draw_wikipead_all_chart.py --dataset mmlu

# 仅分析所有数据集的top10配置
python draw_wikipead_all_chart.py --topk 10

# 分析特定数据集和配置的组合
python draw_wikipead_all_chart.py --dataset hotpotqa --topk 5

# 生成特定配置的分析报告
python generate_wikipead_analysis_report.py --dataset nq --topk 10
```

### 参数说明

- `--dataset`: 选择数据集
  - `mmlu`: MMLU数据集
  - `nq`: Natural Questions数据集  
  - `hotpotqa`: HotpotQA数据集
  - `triviaqa`: TriviaQA数据集
  - `all`: 所有数据集 (默认)

- `--topk`: 选择Top-K配置
  - `1`: Top-1配置
  - `5`: Top-5配置
  - `10`: Top-10配置
  - `all`: 所有可用配置 (默认)

## 输出文件

### 图表文件 (保存在 `output/charts/` 目录)
- `wikipead_frequency_distribution.png` - 文档频率分布对比图
- `wikipead_ngram_distribution.png` - N-gram分布对比图
- `wikipead_high_level_distribution.png` - HNSW高层节点分布图
- `wikipead_level_distribution_detail.png` - 详细层级分布图
- `wikipead_comprehensive_dashboard.png` - 综合对比仪表板

### 分析报告文件
- `wikipead_analysis_report.txt` - 详细的文本分析报告

### 个性化输出
当指定特定数据集或配置时，输出文件名会包含相应后缀：
- 例如: `wikipead_frequency_distribution_mmlu_top10.png`
- 例如: `wikipead_analysis_report_hotpotqa_top5.txt`

## 数据要求

这些工具需要 `wikipead_all.py` 生成的以下类型的统计文件：

### 必需文件
- `freq_stats_{dataset}_top{k}.txt` - 文档频率统计
- `ngram_stats_n{n}_{dataset}_top{k}.txt` - N-gram序列统计 (n=2,3,4)
- `high_level_stats_{dataset}_top{k}.txt` - HNSW高层节点统计

### 支持的数据集和配置组合
- 数据集: mmlu, nq, hotpotqa, triviaqa
- Top-K: 1, 5, 10

## 图表特性

### 中文支持
- 图表标题、轴标签、图例等均支持中文显示
- 配置了SimHei、Microsoft YaHei等中文字体

### 高质量输出
- 300 DPI分辨率，适合论文和报告使用
- 自动调整图表布局和大小
- 支持数值标签显示

### 智能数据处理
- 自动检测可用的统计文件
- 处理多种文件编码格式 (UTF-8, GB2312等)
- 优雅处理缺失数据

## 分析报告内容

生成的分析报告包含以下章节：

1. **执行摘要** - 关键发现概览
2. **文档频率分布分析** - 详细的频率统计
3. **N-gram序列分析** - 连续文档对的模式分析
4. **HNSW索引特征分析** - 索引层级和结构分析
5. **关键洞察** - 跨数据集的比较发现
6. **优化建议** - 基于分析的实用建议
7. **技术说明** - 指标解释和计算方法

## 故障排除

### 常见问题

1. **找不到统计文件**
   - 确保已运行 `wikipead_all.py` 生成统计文件
   - 检查文件名格式是否正确

2. **图表显示异常**
   - 检查是否安装了中文字体
   - 确保matplotlib版本兼容

3. **编码错误**
   - 工具会自动尝试多种编码格式
   - 如仍有问题，检查统计文件的编码

### 依赖要求

```bash
pip install matplotlib numpy
```

## 高级用法

### 批量处理
可以编写脚本批量生成不同配置的图表：

```bash
# 生成所有单个数据集的图表
for dataset in mmlu nq hotpotqa triviaqa; do
    python draw_wikipead_all_chart.py --dataset $dataset
done

# 生成所有配置的报告
for topk in 1 5 10; do
    python generate_wikipead_analysis_report.py --topk $topk
done
```

### 自定义输出
可以修改脚本中的文件路径和输出格式以满足特定需求。

---

这些工具专门设计用于分析 `wikipead_all.py` 的输出，提供了比原有工具更丰富的分析维度和更详细的洞察。