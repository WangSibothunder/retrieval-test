#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为 wikipead_all.py 输出数据绘制专门的可视化图表
包括文档频率分布、N-gram分析、HNSW节点层级分析等
"""

import os
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import argparse

# 配置matplotlib中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 确保输出目录存在
os.makedirs("output/charts", exist_ok=True)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="为wikipead_all.py输出数据绘制图表")
    parser.add_argument("--dataset", type=str, default="all", 
                       choices=["mmlu", "nq", "hotpotqa", "triviaqa", "all"],
                       help="选择数据集进行可视化 (默认: all)")
    parser.add_argument("--topk", type=str, default="all",
                       help="选择top-k配置进行可视化，如 '10' 或 'all' (默认: all)")
    return parser.parse_args()

def extract_value_from_file(file_path, pattern):
    """从统计文件中提取数值"""
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return None
    
    try:
        # 尝试多种编码格式
        encodings = ['utf-8', 'gb2312', 'gbk', 'utf-8-sig']
        content = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            print(f"无法读取文件: {file_path}")
            return None
        
        match = re.search(pattern, content)
        return float(match.group(1)) if match else None
    except Exception as e:
        print(f"读取文件失败 {file_path}: {e}")
        return None

def collect_frequency_data(datasets, topks):
    """收集文档频率分布数据"""
    data = defaultdict(dict)
    
    for dataset in datasets:
        for topk in topks:
            file_path = f"freq_stats_{dataset}_top{topk}.txt"
            if os.path.exists(file_path):
                pattern = r"Top 10% 文档占总检索的 ([\d.]+)%"
                percentage = extract_value_from_file(file_path, pattern)
                if percentage is not None:
                    data[dataset][f"top{topk}"] = percentage
                    print(f"文档频率 - {dataset} top{topk}: {percentage}%")
    
    return data

def collect_ngram_data(datasets, topks):
    """收集N-gram分布数据"""
    data = defaultdict(lambda: defaultdict(dict))
    ngram_sizes = [2, 3, 4]
    
    for n in ngram_sizes:
        for dataset in datasets:
            for topk in topks:
                file_path = f"ngram_stats_n{n}_{dataset}_top{topk}.txt"
                if os.path.exists(file_path):
                    pattern = rf"Top 10% {n}-gram 占总访问的 ([\d.]+)%"
                    percentage = extract_value_from_file(file_path, pattern)
                    if percentage is not None:
                        data[n][dataset][f"top{topk}"] = percentage
                        print(f"{n}-gram - {dataset} top{topk}: {percentage}%")
    
    return data

def collect_high_level_data(datasets, topks):
    """收集HNSW高层节点分布数据"""
    data = defaultdict(dict)
    
    for dataset in datasets:
        for topk in topks:
            file_path = f"high_level_stats_{dataset}_top{topk}.txt"
            if os.path.exists(file_path):
                pattern = r"Top 10% 热门文档中高层节点占比: ([\d.]+)%"
                percentage = extract_value_from_file(file_path, pattern)
                if percentage is not None:
                    data[dataset][f"top{topk}"] = percentage
                    print(f"高层节点 - {dataset} top{topk}: {percentage}%")
    
    return data

def collect_hnsw_level_counts(datasets, topks):
    """收集HNSW层级统计数据"""
    data = defaultdict(lambda: defaultdict(dict))
    
    for dataset in datasets:
        for topk in topks:
            file_path = f"high_level_stats_{dataset}_top{topk}.txt"
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    try:
                        with open(file_path, 'r', encoding='gb2312') as f:
                            content = f.read()
                    except Exception as e:
                        print(f"读取文件失败 {file_path}: {e}")
                        continue
                
                # 提取Top-10热门文档的层级分布
                level_pattern = r"Rank \d+: Doc \d+ \(Freq \d+\) - 高层节点: \w+ \(层级 (\d+)\)"
                levels = re.findall(level_pattern, content)
                if levels:
                    level_counts = defaultdict(int)
                    for level in levels:
                        level_counts[int(level)] += 1
                    data[dataset][f"top{topk}"] = dict(level_counts)
                    print(f"层级分布 - {dataset} top{topk}: {dict(level_counts)}")
    
    return data

def create_frequency_comparison_chart(data, output_suffix=""):
    """创建文档频率分布对比图表"""
    if not data:
        print("没有找到频率分布数据")
        return
    
    datasets = list(data.keys())
    configs = list(next(iter(data.values())).keys()) if data else []
    
    if not datasets or not configs:
        print("数据为空，无法创建图表")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(datasets))
    width = 0.25 if len(configs) <= 3 else 0.8 / len(configs)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    for i, config in enumerate(configs):
        values = [data[dataset].get(config, 0) for dataset in datasets]
        bars = ax.bar(x + i * width, values, width, 
                     label=config.upper(), color=colors[i % len(colors)], alpha=0.8)
        
        # 在柱子上添加数值标签
        for bar, value in zip(bars, values):
            if value > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('数据集', fontsize=12, fontweight='bold')
    ax.set_ylabel('Top 10% 文档占总检索的百分比 (%)', fontsize=12, fontweight='bold')
    ax.set_title('不同数据集文档频率分布对比 - 热门文档集中度分析', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(configs) - 1) / 2)
    ax.set_xticklabels([dataset.upper() for dataset in datasets])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 设置y轴范围
    max_val = max([max(data[dataset].values()) for dataset in datasets if data[dataset]]) if data else 0
    if max_val > 0:
        ax.set_ylim(0, max_val * 1.1)
    
    plt.tight_layout()
    filename = f'output/charts/wikipead_frequency_distribution{output_suffix}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"文档频率分布图已保存到 {filename}")

def create_ngram_comparison_chart(data, output_suffix=""):
    """创建N-gram分布对比图表"""
    if not data:
        print("没有找到N-gram分布数据")
        return
    
    ngram_sizes = sorted(data.keys())
    if not ngram_sizes:
        return
    
    # 获取数据集和配置
    sample_ngram = next(iter(data.values()))
    datasets = list(sample_ngram.keys()) if sample_ngram else []
    configs = list(next(iter(sample_ngram.values())).keys()) if sample_ngram and datasets else []
    
    if not datasets or not configs:
        print("N-gram数据为空，无法创建图表")
        return
    
    fig, axes = plt.subplots(1, len(ngram_sizes), figsize=(5 * len(ngram_sizes), 6))
    if len(ngram_sizes) == 1:
        axes = [axes]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    for idx, n in enumerate(ngram_sizes):
        ax = axes[idx]
        x = np.arange(len(datasets))
        width = 0.25 if len(configs) <= 3 else 0.8 / len(configs)
        
        for i, config in enumerate(configs):
            values = [data[n][dataset].get(config, 0) for dataset in datasets]
            bars = ax.bar(x + i * width, values, width, 
                         label=config.upper(), color=colors[i % len(colors)], alpha=0.8)
            
            # 在柱子上添加数值标签
            for bar, value in zip(bars, values):
                if value > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('数据集')
        ax.set_ylabel(f'Top 10% {n}-gram 占总访问的百分比 (%)')
        ax.set_title(f'{n}-gram 序列分布对比')
        ax.set_xticks(x + width * (len(configs) - 1) / 2)
        ax.set_xticklabels([dataset.upper() for dataset in datasets])
        if idx == 0:
            ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 设置y轴范围
        max_val = 0
        for dataset in datasets:
            if dataset in data[n] and data[n][dataset]:
                max_val = max(max_val, max(data[n][dataset].values()))
        if max_val > 0:
            ax.set_ylim(0, max_val * 1.1)
    
    plt.tight_layout()
    filename = f'output/charts/wikipead_ngram_distribution{output_suffix}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"N-gram分布图已保存到 {filename}")

def create_high_level_comparison_chart(data, output_suffix=""):
    """创建HNSW高层节点分布对比图表"""
    if not data:
        print("没有找到高层节点分布数据")
        return
    
    datasets = list(data.keys())
    configs = list(next(iter(data.values())).keys()) if data else []
    
    if not datasets or not configs:
        print("高层节点数据为空，无法创建图表")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(datasets))
    width = 0.25 if len(configs) <= 3 else 0.8 / len(configs)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    for i, config in enumerate(configs):
        values = [data[dataset].get(config, 0) for dataset in datasets]
        bars = ax.bar(x + i * width, values, width, 
                     label=config.upper(), color=colors[i % len(colors)], alpha=0.8)
        
        # 在柱子上添加数值标签
        for bar, value in zip(bars, values):
            if value > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('数据集', fontsize=12, fontweight='bold')
    ax.set_ylabel('Top 10% 热门文档中高层节点占比 (%)', fontsize=12, fontweight='bold')
    ax.set_title('不同数据集HNSW高层节点分布对比 - 热门文档在索引层级中的分布', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(configs) - 1) / 2)
    ax.set_xticklabels([dataset.upper() for dataset in datasets])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 设置y轴范围
    max_val = max([max(data[dataset].values()) for dataset in datasets if data[dataset]]) if data else 0
    if max_val > 0:
        ax.set_ylim(0, max_val * 1.1)
    
    plt.tight_layout()
    filename = f'output/charts/wikipead_high_level_distribution{output_suffix}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"HNSW高层节点分布图已保存到 {filename}")

def create_level_distribution_chart(data, output_suffix=""):
    """创建层级分布详细图表"""
    if not data:
        print("没有找到层级分布数据")
        return
    
    datasets = list(data.keys())
    configs = list(next(iter(data.values())).keys()) if data else []
    
    if not datasets or not configs:
        print("层级分布数据为空，无法创建图表")
        return
    
    # 获取所有可能的层级
    all_levels = set()
    for dataset in datasets:
        for config in configs:
            if config in data[dataset]:
                all_levels.update(data[dataset][config].keys())
    all_levels = sorted(all_levels)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    for idx, dataset in enumerate(datasets):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        x = np.arange(len(configs))
        width = 0.8 / len(all_levels) if all_levels else 0.8
        
        for level_idx, level in enumerate(all_levels):
            values = []
            for config in configs:
                level_data = data[dataset].get(config, {})
                values.append(level_data.get(level, 0))
            
            bars = ax.bar(x + level_idx * width, values, width, 
                         label=f'Level {level}', color=colors[level_idx % len(colors)], alpha=0.8)
            
            # 在柱子上添加数值标签
            for bar, value in zip(bars, values):
                if value > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{value}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Top-K配置')
        ax.set_ylabel('热门文档数量')
        ax.set_title(f'{dataset.upper()} - Top-10热门文档层级分布')
        ax.set_xticks(x + width * (len(all_levels) - 1) / 2)
        ax.set_xticklabels([config.upper() for config in configs])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for idx in range(len(datasets), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    filename = f'output/charts/wikipead_level_distribution_detail{output_suffix}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"详细层级分布图已保存到 {filename}")

def create_comprehensive_dashboard(freq_data, ngram_data, high_level_data, output_suffix=""):
    """创建综合对比仪表板"""
    fig = plt.figure(figsize=(20, 15))
    
    # 获取通用配置
    datasets = list(freq_data.keys()) if freq_data else []
    configs = list(next(iter(freq_data.values())).keys()) if freq_data and datasets else []
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    # 1. 文档频率分布 (左上)
    if freq_data and datasets and configs:
        ax1 = plt.subplot(2, 3, (1, 2))
        x = np.arange(len(datasets))
        width = 0.25 if len(configs) <= 3 else 0.8 / len(configs)
        
        for i, config in enumerate(configs):
            values = [freq_data[dataset].get(config, 0) for dataset in datasets]
            bars = ax1.bar(x + i * width, values, width, 
                          label=config.upper(), color=colors[i % len(colors)], alpha=0.8)
            
            for bar, value in zip(bars, values):
                if value > 0:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
        
        ax1.set_xlabel('数据集')
        ax1.set_ylabel('Top 10% 文档占总检索的百分比 (%)')
        ax1.set_title('文档频率分布对比', fontsize=12, fontweight='bold')
        ax1.set_xticks(x + width * (len(configs) - 1) / 2)
        ax1.set_xticklabels([dataset.upper() for dataset in datasets])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. 高层节点分布 (右上)
    if high_level_data and datasets and configs:
        ax2 = plt.subplot(2, 3, 3)
        x = np.arange(len(datasets))
        width = 0.25 if len(configs) <= 3 else 0.8 / len(configs)
        
        for i, config in enumerate(configs):
            values = [high_level_data[dataset].get(config, 0) for dataset in datasets]
            bars = ax2.bar(x + i * width, values, width, 
                          label=config.upper(), color=colors[i % len(colors)], alpha=0.8)
            
            for bar, value in zip(bars, values):
                if value > 0:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
        
        ax2.set_xlabel('数据集')
        ax2.set_ylabel('高层节点占比 (%)')
        ax2.set_title('HNSW高层节点分布', fontsize=12, fontweight='bold')
        ax2.set_xticks(x + width * (len(configs) - 1) / 2)
        ax2.set_xticklabels([dataset.upper() for dataset in datasets])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. N-gram分布 (下方)
    if ngram_data:
        ngram_sizes = sorted(ngram_data.keys())
        for idx, n in enumerate(ngram_sizes[:3]):  # 最多显示3个n-gram
            ax = plt.subplot(2, 3, 4 + idx)
            x = np.arange(len(datasets))
            width = 0.25 if len(configs) <= 3 else 0.8 / len(configs)
            
            for i, config in enumerate(configs):
                values = [ngram_data[n][dataset].get(config, 0) for dataset in datasets]
                bars = ax.bar(x + i * width, values, width, 
                             label=config.upper(), color=colors[i % len(colors)], alpha=0.8)
                
                for bar, value in zip(bars, values):
                    if value > 0:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                               f'{value:.1f}%', ha='center', va='bottom', fontsize=7)
            
            ax.set_xlabel('数据集')
            ax.set_ylabel(f'{n}-gram 占比 (%)')
            ax.set_title(f'{n}-gram 分布', fontsize=12, fontweight='bold')
            ax.set_xticks(x + width * (len(configs) - 1) / 2)
            ax.set_xticklabels([dataset.upper() for dataset in datasets])
            if idx == 0:
                ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'output/charts/wikipead_comprehensive_dashboard{output_suffix}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"综合对比仪表板已保存到 {filename}")

def main():
    """主函数"""
    args = parse_arguments()
    
    # 确定要处理的数据集和top-k配置
    if args.dataset == "all":
        datasets = ["mmlu", "nq", "hotpotqa", "triviaqa"]
    else:
        datasets = [args.dataset]
    
    if args.topk == "all":
        # 自动检测可用的top-k配置
        topks = []
        for k in [1, 5, 10]:
            # 检查是否存在对应的文件
            test_file = f"freq_stats_{datasets[0]}_top{k}.txt"
            if os.path.exists(test_file):
                topks.append(k)
        if not topks:
            topks = [10]  # 默认值
    else:
        try:
            topks = [int(args.topk)]
        except ValueError:
            print(f"无效的top-k值: {args.topk}")
            return
    
    output_suffix = f"_{args.dataset}_top{args.topk}" if args.dataset != "all" or args.topk != "all" else ""
    
    print(f"开始为 wikipead_all.py 生成图表...")
    print(f"数据集: {datasets}")
    print(f"Top-K配置: {topks}")
    
    # 收集数据
    print("\n=== 收集文档频率分布数据 ===")
    freq_data = collect_frequency_data(datasets, topks)
    
    print("\n=== 收集N-gram分布数据 ===")
    ngram_data = collect_ngram_data(datasets, topks)
    
    print("\n=== 收集HNSW高层节点分布数据 ===")
    high_level_data = collect_high_level_data(datasets, topks)
    
    print("\n=== 收集HNSW层级详细分布数据 ===")
    level_data = collect_hnsw_level_counts(datasets, topks)
    
    # 生成图表
    print("\n=== 生成图表 ===")
    if freq_data:
        create_frequency_comparison_chart(freq_data, output_suffix)
    
    if ngram_data:
        create_ngram_comparison_chart(ngram_data, output_suffix)
    
    if high_level_data:
        create_high_level_comparison_chart(high_level_data, output_suffix)
    
    if level_data:
        create_level_distribution_chart(level_data, output_suffix)
    
    # 生成综合仪表板
    if any([freq_data, ngram_data, high_level_data]):
        print("\n=== 生成综合仪表板 ===")
        create_comprehensive_dashboard(freq_data, ngram_data, high_level_data, output_suffix)
    
    print(f"\n所有图表生成完成！输出目录: output/charts/")

if __name__ == "__main__":
    main()