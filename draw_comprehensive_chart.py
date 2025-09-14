#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合图表绘制工具
用于可视化 wikipead_all.py 生成的各种统计数据
"""

import os
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# 配置matplotlib中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 确保输出目录存在
os.makedirs("output/charts", exist_ok=True)

def extract_percentage_from_file(file_path, pattern):
    """从统计文件中提取百分比数据"""
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='gb2312') as f:
                content = f.read()
        except Exception as e:
            print(f"读取文件失败 {file_path}: {e}")
            return None
    
    match = re.search(pattern, content)
    return float(match.group(1)) if match else None

def collect_frequency_data():
    """收集所有数据集的文档频率分布数据"""
    databases = ["mmlu", "nq", "hotpotqa", "triviaqa"]
    top_configs = ["top1", "top5", "top10"]
    
    data = defaultdict(dict)
    
    for db in databases:
        for config in top_configs:
            file_path = f"freq_stats_{db}_{config}.txt"
            if os.path.exists(file_path):
                pattern = r"Top 10% 文档占总检索的 ([\d.]+)%"
                percentage = extract_percentage_from_file(file_path, pattern)
                if percentage is not None:
                    data[db][config] = percentage
                    print(f"提取 {db} {config}: {percentage}%")
    
    return data

def collect_ngram_data():
    """收集所有数据集的n-gram分布数据"""
    databases = ["mmlu", "nq", "hotpotqa", "triviaqa"]
    top_configs = ["top1", "top5", "top10"]
    ngram_sizes = [2, 3, 4]
    
    data = defaultdict(lambda: defaultdict(dict))
    
    for db in databases:
        for config in top_configs:
            for n in ngram_sizes:
                file_path = f"ngram_stats_n{n}_{db}_{config}.txt"
                if os.path.exists(file_path):
                    pattern = rf"Top 10% {n}-gram 占总访问的 ([\d.]+)%"
                    percentage = extract_percentage_from_file(file_path, pattern)
                    if percentage is not None:
                        data[n][db][config] = percentage
                        print(f"提取 {n}-gram {db} {config}: {percentage}%")
    
    return data

def collect_high_level_data():
    """收集高层节点分布数据"""
    databases = ["mmlu", "nq", "hotpotqa", "triviaqa"]
    top_configs = ["top1", "top5", "top10"]
    
    data = defaultdict(dict)
    
    for db in databases:
        for config in top_configs:
            file_path = f"high_level_stats_{db}_{config}.txt"
            if os.path.exists(file_path):
                pattern = r"Top 10% 热门文档中高层节点占比: ([\d.]+)%"
                percentage = extract_percentage_from_file(file_path, pattern)
                if percentage is not None:
                    data[db][config] = percentage
                    print(f"提取高层节点比例 {db} {config}: {percentage}%")
    
    return data

def create_frequency_comparison_chart(data):
    """创建文档频率分布对比图表"""
    if not data:
        print("没有找到频率分布数据")
        return
    
    databases = list(data.keys())
    top_configs = ["top1", "top5", "top10"]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(databases))
    width = 0.25
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, config in enumerate(top_configs):
        values = []
        for db in databases:
            values.append(data[db].get(config, 0))
        
        bars = ax.bar(x + i * width, values, width, 
                     label=config.upper(), color=colors[i], alpha=0.8)
        
        # 在柱子上添加数值标签
        for bar, value in zip(bars, values):
            if value > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{value:.1f}%', ha='center', va='bottom')
    
    ax.set_xlabel('数据集')
    ax.set_ylabel('Top 10% 文档占总检索的百分比 (%)')
    ax.set_title('不同数据集文档频率分布对比 - 热门文档集中度分析')
    ax.set_xticks(x + width)
    ax.set_xticklabels([db.upper() for db in databases])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max([max(data[db].values()) for db in databases if data[db]]) * 1.1)
    
    plt.tight_layout()
    plt.savefig('output/charts/frequency_distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("文档频率分布对比图已保存到 output/charts/frequency_distribution_comparison.png")

def create_ngram_comparison_chart(data):
    """创建n-gram分布对比图表"""
    if not data:
        print("没有找到n-gram分布数据")
        return
    
    ngram_sizes = sorted(data.keys())
    databases = ["mmlu", "nq", "hotpotqa", "triviaqa"]
    top_configs = ["top1", "top5", "top10"]
    
    fig, axes = plt.subplots(1, len(ngram_sizes), figsize=(15, 6))
    if len(ngram_sizes) == 1:
        axes = [axes]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for idx, n in enumerate(ngram_sizes):
        ax = axes[idx]
        x = np.arange(len(databases))
        width = 0.25
        
        for i, config in enumerate(top_configs):
            values = []
            for db in databases:
                values.append(data[n][db].get(config, 0))
            
            bars = ax.bar(x + i * width, values, width, 
                         label=config.upper(), color=colors[i], alpha=0.8)
            
            # 在柱子上添加数值标签
            for bar, value in zip(bars, values):
                if value > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('数据集')
        ax.set_ylabel('Top 10% n-gram 占总访问的百分比 (%)')
        ax.set_title(f'{n}-gram 分布对比')
        ax.set_xticks(x + width)
        ax.set_xticklabels([db.upper() for db in databases])
        if idx == 0:
            ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 设置y轴范围
        max_val = 0
        for db in databases:
            if db in data[n] and data[n][db]:
                max_val = max(max_val, max(data[n][db].values()))
        if max_val > 0:
            ax.set_ylim(0, max_val * 1.1)
    
    plt.tight_layout()
    plt.savefig('output/charts/ngram_distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("N-gram分布对比图已保存到 output/charts/ngram_distribution_comparison.png")

def create_high_level_comparison_chart(data):
    """创建高层节点分布对比图表"""
    if not data:
        print("没有找到高层节点分布数据")
        return
    
    databases = list(data.keys())
    top_configs = ["top1", "top5", "top10"]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(databases))
    width = 0.25
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, config in enumerate(top_configs):
        values = []
        for db in databases:
            values.append(data[db].get(config, 0))
        
        bars = ax.bar(x + i * width, values, width, 
                     label=config.upper(), color=colors[i], alpha=0.8)
        
        # 在柱子上添加数值标签
        for bar, value in zip(bars, values):
            if value > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{value:.1f}%', ha='center', va='bottom')
    
    ax.set_xlabel('数据集')
    ax.set_ylabel('Top 10% 热门文档中高层节点占比 (%)')
    ax.set_title('不同数据集HNSW高层节点分布对比 - 热门文档在索引中的层级分析')
    ax.set_xticks(x + width)
    ax.set_xticklabels([db.upper() for db in databases])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 设置y轴范围
    max_val = max([max(data[db].values()) for db in databases if data[db]]) if data else 0
    if max_val > 0:
        ax.set_ylim(0, max_val * 1.1)
    
    plt.tight_layout()
    plt.savefig('output/charts/high_level_distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("高层节点分布对比图已保存到 output/charts/high_level_distribution_comparison.png")

def create_comprehensive_dashboard():
    """创建综合对比仪表板"""
    print("开始收集综合数据...")
    
    freq_data = collect_frequency_data()
    ngram_data = collect_ngram_data()
    high_level_data = collect_high_level_data()
    
    if not any([freq_data, ngram_data, high_level_data]):
        print("没有找到任何统计数据文件")
        return
    
    # 创建综合对比图
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 文档频率分布 (左上)
    if freq_data:
        ax1 = plt.subplot(2, 3, (1, 2))
        databases = list(freq_data.keys())
        top_configs = ["top1", "top5", "top10"]
        x = np.arange(len(databases))
        width = 0.25
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, config in enumerate(top_configs):
            values = [freq_data[db].get(config, 0) for db in databases]
            bars = ax1.bar(x + i * width, values, width, 
                          label=config.upper(), color=colors[i], alpha=0.8)
            
            for bar, value in zip(bars, values):
                if value > 0:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
        
        ax1.set_xlabel('数据集')
        ax1.set_ylabel('Top 10% 文档占总检索的百分比 (%)')
        ax1.set_title('文档频率分布对比')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels([db.upper() for db in databases])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. 高层节点分布 (右上)
    if high_level_data:
        ax2 = plt.subplot(2, 3, 3)
        databases = list(high_level_data.keys())
        top_configs = ["top1", "top5", "top10"]  # 确保变量定义
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # 确保变量定义
        x = np.arange(len(databases))
        width = 0.25
        
        for i, config in enumerate(top_configs):
            values = [high_level_data[db].get(config, 0) for db in databases]
            bars = ax2.bar(x + i * width, values, width, 
                          label=config.upper(), color=colors[i], alpha=0.8)
            
            for bar, value in zip(bars, values):
                if value > 0:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
        
        ax2.set_xlabel('数据集')
        ax2.set_ylabel('高层节点占比 (%)')
        ax2.set_title('HNSW高层节点分布')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels([db.upper() for db in databases])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. N-gram分布 (下方)
    if ngram_data:
        ngram_sizes = sorted(ngram_data.keys())
        for idx, n in enumerate(ngram_sizes):
            ax = plt.subplot(2, 3, 4 + idx)
            databases = ["mmlu", "nq", "hotpotqa", "triviaqa"]
            top_configs = ["top1", "top5", "top10"]  # 确保变量定义
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # 确保变量定义
            x = np.arange(len(databases))
            width = 0.25
            
            for i, config in enumerate(top_configs):
                values = [ngram_data[n][db].get(config, 0) for db in databases]
                bars = ax.bar(x + i * width, values, width, 
                             label=config.upper(), color=colors[i], alpha=0.8)
                
                for bar, value in zip(bars, values):
                    if value > 0:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                               f'{value:.1f}%', ha='center', va='bottom', fontsize=7)
            
            ax.set_xlabel('数据集')
            ax.set_ylabel(f'{n}-gram 占比 (%)')
            ax.set_title(f'{n}-gram 分布')
            ax.set_xticks(x + width)
            ax.set_xticklabels([db.upper() for db in databases])
            if idx == 0:
                ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/charts/comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("综合对比仪表板已保存到 output/charts/comprehensive_dashboard.png")

def main():
    """主函数"""
    print("开始生成综合图表...")
    
    # 收集数据
    print("\n=== 收集文档频率分布数据 ===")
    freq_data = collect_frequency_data()
    
    print("\n=== 收集N-gram分布数据 ===")
    ngram_data = collect_ngram_data()
    
    print("\n=== 收集高层节点分布数据 ===")
    high_level_data = collect_high_level_data()
    
    # 生成图表
    print("\n=== 生成图表 ===")
    if freq_data:
        create_frequency_comparison_chart(freq_data)
    
    if ngram_data:
        create_ngram_comparison_chart(ngram_data)
    
    if high_level_data:
        create_high_level_comparison_chart(high_level_data)
    
    # 生成综合仪表板
    print("\n=== 生成综合仪表板 ===")
    create_comprehensive_dashboard()
    
    print("\n所有图表生成完成！")

if __name__ == "__main__":
    main()