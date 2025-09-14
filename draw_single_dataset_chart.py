#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单数据集详细图表绘制工具
用于深入分析单个数据集的多维度统计结果
"""

import os
import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# 配置matplotlib中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 确保输出目录存在
os.makedirs("output/charts", exist_ok=True)

def parse_frequency_stats(file_path):
    """解析频率统计文件，提取top-10文档信息"""
    if not os.path.exists(file_path):
        return None, None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='gb2312') as f:
                content = f.read()
        except Exception as e:
            print(f"读取文件失败 {file_path}: {e}")
            return None, None
    
    # 提取top-10文档频率
    doc_frequencies = []
    lines = content.split('\n')
    for line in lines:
        if line.startswith('Rank'):
            match = re.search(r'Rank (\d+): Doc (\d+) - (\d+) 次', line)
            if match:
                rank, doc_id, freq = int(match.group(1)), int(match.group(2)), int(match.group(3))
                doc_frequencies.append((rank, doc_id, freq))
    
    # 提取top 10%占比
    percentage_match = re.search(r'Top 10% 文档占总检索的 ([\d.]+)%', content)
    percentage = float(percentage_match.group(1)) if percentage_match else None
    
    return doc_frequencies, percentage

def parse_ngram_stats(file_path):
    """解析n-gram统计文件"""
    if not os.path.exists(file_path):
        return None, None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='gb2312') as f:
                content = f.read()
        except Exception as e:
            print(f"读取文件失败 {file_path}: {e}")
            return None, None
    
    # 提取top-10 n-gram频率
    ngram_frequencies = []
    lines = content.split('\n')
    for line in lines:
        if line.startswith('Rank'):
            match = re.search(r'Rank (\d+): (\d+) 次', line)
            if match:
                rank, freq = int(match.group(1)), int(match.group(2))
                ngram_frequencies.append((rank, freq))
    
    # 提取top 10%占比
    n_value = re.search(r'(\d+)-gram', file_path)
    n = n_value.group(1) if n_value else "n"
    pattern = rf'Top 10% {n}-gram 占总访问的 ([\d.]+)%'
    percentage_match = re.search(pattern, content)
    percentage = float(percentage_match.group(1)) if percentage_match else None
    
    return ngram_frequencies, percentage

def parse_high_level_stats(file_path):
    """解析高层节点统计文件"""
    if not os.path.exists(file_path):
        return None, None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='gb2312') as f:
                content = f.read()
        except Exception as e:
            print(f"读取文件失败 {file_path}: {e}")
            return None, None
    
    # 提取top-10文档的层级信息
    doc_levels = []
    lines = content.split('\n')
    for line in lines:
        if line.startswith('Rank'):
            match = re.search(r'Rank (\d+): Doc (\d+) \(Freq (\d+)\) - 高层节点: (True|False) \(层级 (\d+)\)', line)
            if match:
                rank = int(match.group(1))
                doc_id = int(match.group(2))
                freq = int(match.group(3))
                is_high_level = match.group(4) == 'True'
                level = int(match.group(5))
                doc_levels.append((rank, doc_id, freq, is_high_level, level))
    
    # 提取高层节点占比
    percentage_match = re.search(r'Top 10% 热门文档中高层节点占比: ([\d.]+)%', content)
    percentage = float(percentage_match.group(1)) if percentage_match else None
    
    return doc_levels, percentage

def create_single_dataset_dashboard(dataset, topk):
    """为单个数据集创建综合分析仪表板"""
    print(f"创建 {dataset.upper()} Top-{topk} 数据集综合分析仪表板...")
    
    # 文件路径
    freq_file = f"freq_stats_{dataset}_top{topk}.txt"
    high_level_file = f"high_level_stats_{dataset}_top{topk}.txt"
    ngram_files = {
        2: f"ngram_stats_n2_{dataset}_top{topk}.txt",
        3: f"ngram_stats_n3_{dataset}_top{topk}.txt",
        4: f"ngram_stats_n4_{dataset}_top{topk}.txt"
    }
    
    # 解析数据
    doc_frequencies, doc_percentage = parse_frequency_stats(freq_file)
    doc_levels, high_level_percentage = parse_high_level_stats(high_level_file)
    ngram_data = {}
    for n, file_path in ngram_files.items():
        frequencies, percentage = parse_ngram_stats(file_path)
        if frequencies and percentage:
            ngram_data[n] = (frequencies, percentage)
    
    # 创建图表
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'{dataset.upper()} Top-{topk} 数据集综合分析仪表板', fontsize=16, fontweight='bold')
    
    # 1. 文档频率分布 (左上)
    if doc_frequencies:
        ax1 = plt.subplot(2, 4, 1)
        ranks = [item[0] for item in doc_frequencies]
        freqs = [item[2] for item in doc_frequencies]
        
        bars = ax1.bar(ranks, freqs, color='#FF6B6B', alpha=0.7)
        ax1.set_xlabel('文档排名')
        ax1.set_ylabel('检索频率')
        ax1.set_title('Top-10 文档频率分布')
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, freq in zip(bars, freqs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(freqs)*0.01,
                    str(freq), ha='center', va='bottom', fontsize=8)
        
        # 添加占比信息
        if doc_percentage:
            ax1.text(0.02, 0.98, f'Top 10% 文档占比: {doc_percentage:.1f}%', 
                    transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. HNSW层级分布 (右上)
    if doc_levels:
        ax2 = plt.subplot(2, 4, 2)
        levels = [item[4] for item in doc_levels]
        colors = ['#FF6B6B' if item[3] else '#4ECDC4' for item in doc_levels]
        
        ranks = [item[0] for item in doc_levels]
        bars = ax2.bar(ranks, levels, color=colors, alpha=0.7)
        ax2.set_xlabel('文档排名')
        ax2.set_ylabel('HNSW层级')
        ax2.set_title('Top-10 文档HNSW层级分布')
        ax2.grid(True, alpha=0.3)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#FF6B6B', alpha=0.7, label='高层节点 (Level > 0)'),
                          Patch(facecolor='#4ECDC4', alpha=0.7, label='底层节点 (Level = 0)')]
        ax2.legend(handles=legend_elements, loc='upper right')
        
        # 添加占比信息
        if high_level_percentage:
            ax2.text(0.02, 0.98, f'高层节点占比: {high_level_percentage:.1f}%', 
                    transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 3. N-gram分布对比 (下半部分)
    if ngram_data:
        ngram_positions = [5, 6, 7]  # 对应2, 3, 4-gram
        colors_ngram = ['#45B7D1', '#96CEB4', '#FFEAA7']
        
        for idx, (n, (frequencies, percentage)) in enumerate(ngram_data.items()):
            if idx < len(ngram_positions):
                ax = plt.subplot(2, 4, ngram_positions[idx])
                ranks = [item[0] for item in frequencies]
                freqs = [item[1] for item in frequencies]
                
                bars = ax.bar(ranks, freqs, color=colors_ngram[idx], alpha=0.7)
                ax.set_xlabel('排名')
                ax.set_ylabel('频率')
                ax.set_title(f'Top-10 {n}-gram 分布')
                ax.grid(True, alpha=0.3)
                
                # 添加数值标签
                for bar, freq in zip(bars, freqs):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(freqs)*0.01,
                           str(freq), ha='center', va='bottom', fontsize=8)
                
                # 添加占比信息
                ax.text(0.02, 0.98, f'Top 10% 占比: {percentage:.1f}%', 
                       transform=ax.transAxes, fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor=colors_ngram[idx], alpha=0.3))
    
    # 4. 综合对比条形图 (右下)
    ax4 = plt.subplot(2, 4, 8)
    categories = ['文档频率']
    percentages = []
    colors_summary = ['#FF6B6B']
    
    if doc_percentage:
        percentages.append(doc_percentage)
    
    if high_level_percentage:
        categories.append('高层节点')
        percentages.append(high_level_percentage)
        colors_summary.append('#4ECDC4')
    
    for n, (_, percentage) in ngram_data.items():
        categories.append(f'{n}-gram')
        percentages.append(percentage)
        colors_summary.append(['#45B7D1', '#96CEB4', '#FFEAA7'][n-2])
    
    if percentages:
        bars = ax4.bar(categories, percentages, color=colors_summary, alpha=0.7)
        ax4.set_ylabel('Top 10% 占比 (%)')
        ax4.set_title('各维度Top 10%占比对比')
        ax4.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, percentage in zip(bars, percentages):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(percentages)*0.01,
                    f'{percentage:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 旋转x轴标签以防重叠
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # 保存图表
    output_path = f'output/charts/{dataset}_top{topk}_dashboard.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"单数据集仪表板已保存到 {output_path}")

def create_log_log_plots(dataset, topk):
    """创建Log-Log分布图"""
    print(f"创建 {dataset.upper()} Top-{topk} Log-Log分布图...")
    
    # 查找对应的分布图文件
    freq_plot = f"hot_docs_distribution_{dataset}_top{topk}.png"
    high_level_plot = f"high_level_distribution_{dataset}_top{topk}.png"
    ngram_plots = [
        f"ngram_distribution_n2_{dataset}_top{topk}.png",
        f"ngram_distribution_n3_{dataset}_top{topk}.png",
        f"ngram_distribution_n4_{dataset}_top{topk}.png"
    ]
    
    existing_plots = []
    plot_titles = []
    
    if os.path.exists(freq_plot):
        existing_plots.append(freq_plot)
        plot_titles.append("文档频率分布")
    
    if os.path.exists(high_level_plot):
        existing_plots.append(high_level_plot)
        plot_titles.append("高层节点分布")
    
    for i, ngram_plot in enumerate(ngram_plots):
        if os.path.exists(ngram_plot):
            existing_plots.append(ngram_plot)
            plot_titles.append(f"{i+2}-gram分布")
    
    if existing_plots:
        print(f"找到 {len(existing_plots)} 个现有的分布图文件")
        for plot, title in zip(existing_plots, plot_titles):
            print(f"  - {title}: {plot}")
    else:
        print("未找到现有的分布图文件")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="单数据集详细图表绘制工具")
    parser.add_argument("--dataset", type=str, required=True, 
                       choices=["mmlu", "nq", "hotpotqa", "triviaqa"],
                       help="选择数据集")
    parser.add_argument("--topk", type=int, required=True,
                       help="指定top-k值")
    
    args = parser.parse_args()
    
    print(f"开始为 {args.dataset.upper()} Top-{args.topk} 生成详细分析图表...")
    
    # 创建综合仪表板
    create_single_dataset_dashboard(args.dataset, args.topk)
    
    # 检查Log-Log分布图
    create_log_log_plots(args.dataset, args.topk)
    
    print("图表生成完成！")

if __name__ == "__main__":
    main()