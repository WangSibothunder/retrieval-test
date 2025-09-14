# Wikipedia 100k 知识库实验输出

本目录包含使用Wikipedia 100k知识库进行的所有实验结果。

主要来源文件：
- `wikipead_all.py`: 主要分析脚本
- `draw_wikipead_all_chart.py`: 图表绘制脚本

实验配置：
- 知识库规模: 100,000篇Wikipedia文章
- 嵌入模型: BAAI/bge-large-en-v1.5
- 向量索引: FAISS HNSW
- Top-K配置: 10, 16, 32
