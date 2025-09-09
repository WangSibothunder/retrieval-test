#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import re
import os
from collections import defaultdict, Counter

# Set Chinese font support
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def extract_hnsw_level_data(file_path):
    """Extract HNSW level data from high level statistics file"""
    try:
        encodings = ['gb2312', 'gbk', 'utf-8', 'utf-8-sig']
        content = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            print(f"Error: Cannot read file {file_path} with any encoding")
            return [], 0.0
        
        # Extract level distribution data
        level_data = []
        lines = content.strip().split('\n')
        
        for line in lines:
            if 'Rank' in line and 'Doc' in line and 'Freq' in line:
                # Parse line like: "Rank 1: Doc 2877 (Freq 1148) - 高层节点: True (层级 1)"
                try:
                    # Extract level number
                    level_match = re.search(r'层级\s+(\d+)', line)
                    if level_match:
                        level = int(level_match.group(1))
                        level_data.append(level)
                except:
                    continue
        
        # Extract high level percentage
        percentage_match = re.search(r'高层节点占比:\s*(\d+\.\d+)%', content)
        high_level_percentage = float(percentage_match.group(1)) if percentage_match else 0.0
        
        return level_data, high_level_percentage
        
    except FileNotFoundError:
        print(f"Error: File not found {file_path}")
        return [], 0.0
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return [], 0.0

def collect_hnsw_data():
    """Collect all HNSW level data"""
    databases = ['hotpotqa', 'mmlu', 'nq', 'triviaqa']
    
    data = {}
    
    for db in databases:
        file_path = f'high_level_stats_{db}_top10.txt'
        level_data, high_level_percentage = extract_hnsw_level_data(file_path)
        
        # Count level distribution
        level_counts = Counter(level_data)
        total_docs = len(level_data)
        
        data[db] = {
            'level_counts': level_counts,
            'total_docs': total_docs,
            'high_level_percentage': high_level_percentage,
            'level_data': level_data
        }
        
        print(f"{db}: Level distribution: {dict(level_counts)}, High level: {high_level_percentage}%")
    
    return data

def create_level_distribution_chart(data):
    """Create HNSW level distribution chart"""
    databases = ['hotpotqa', 'mmlu', 'nq', 'triviaqa']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    axes = [ax1, ax2, ax3, ax4]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for i, db in enumerate(databases):
        ax = axes[i]
        level_counts = data[db]['level_counts']
        total_docs = data[db]['total_docs']
        
        if level_counts:
            levels = sorted(level_counts.keys())
            counts = [level_counts[level] for level in levels]
            percentages = [count/total_docs*100 for count in counts]
            
            bars = ax.bar(levels, percentages, color=colors[i], alpha=0.7, edgecolor='black')
            
            # Add value labels on bars
            for j, (level, percentage) in enumerate(zip(levels, percentages)):
                ax.text(level, percentage + 1, f'{percentage:.1f}%\n({counts[j]})', 
                       ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.set_title(f'{db.upper()} - HNSW层级分布', fontsize=14, fontweight='bold')
        ax.set_xlabel('HNSW层级', fontsize=12)
        ax.set_ylabel('占比 (%)', fontsize=12)
        ax.set_ylim(0, 110)
        ax.grid(True, axis='y', alpha=0.3)
        
        # Set integer ticks for x-axis
        if level_counts:
            ax.set_xticks(sorted(level_counts.keys()))
    
    plt.suptitle('四个数据集热门文章的HNSW层级分布 (Top-10配置)', 
                 fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    
    # Save chart
    os.makedirs('output/charts', exist_ok=True)
    output_file = 'output/charts/hnsw_level_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"HNSW层级分布图已保存为: {output_file}")
    
    plt.show()

def create_high_level_comparison_chart(data):
    """Create high level percentage comparison chart"""
    databases = ['hotpotqa', 'mmlu', 'nq', 'triviaqa']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate high level percentages for each database
    high_level_percentages = []
    level_1_percentages = []
    level_2_plus_percentages = []
    
    for db in databases:
        level_counts = data[db]['level_counts']
        total_docs = data[db]['total_docs']
        
        level_1_count = level_counts.get(1, 0)
        level_2_plus_count = sum(count for level, count in level_counts.items() if level >= 2)
        
        level_1_pct = (level_1_count / total_docs * 100) if total_docs > 0 else 0
        level_2_plus_pct = (level_2_plus_count / total_docs * 100) if total_docs > 0 else 0
        
        level_1_percentages.append(level_1_pct)
        level_2_plus_percentages.append(level_2_plus_pct)
        high_level_percentages.append(data[db]['high_level_percentage'])
    
    x_positions = np.arange(len(databases))
    bar_width = 0.35
    
    # Create stacked bar chart
    bars1 = ax.bar(x_positions, level_1_percentages, bar_width, 
                   label='Level 1', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x_positions, level_2_plus_percentages, bar_width, 
                   bottom=level_1_percentages, label='Level 2+', color='#e74c3c', alpha=0.8)
    
    # Add percentage labels
    for i, (l1, l2) in enumerate(zip(level_1_percentages, level_2_plus_percentages)):
        if l1 > 0:
            ax.text(i, l1/2, f'{l1:.1f}%', ha='center', va='center', 
                   fontweight='bold', color='white', fontsize=11)
        if l2 > 0:
            ax.text(i, l1 + l2/2, f'{l2:.1f}%', ha='center', va='center', 
                   fontweight='bold', color='white', fontsize=11)
    
    ax.set_title('四个数据集热门文章的HNSW层级分布对比', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('数据库', fontsize=14, fontweight='bold')
    ax.set_ylabel('占比 (%)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([db.upper() for db in databases], fontsize=12)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save chart
    os.makedirs('output/charts', exist_ok=True)
    output_file = 'output/charts/hnsw_level_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"HNSW层级对比图已保存为: {output_file}")
    
    plt.show()

def create_frequency_vs_level_chart(data):
    """Create frequency vs HNSW level scatter plot"""
    databases = ['hotpotqa', 'mmlu', 'nq', 'triviaqa']
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    markers = ['o', 's', '^', 'D']
    
    for i, db in enumerate(databases):
        file_path = f'high_level_stats_{db}_top10.txt'
        
        # Extract frequency and level data from file
        frequencies = []
        levels = []
        
        try:
            encodings = ['gb2312', 'gbk', 'utf-8', 'utf-8-sig']
            content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content:
                lines = content.strip().split('\n')
                for line in lines:
                    if 'Rank' in line and 'Doc' in line and 'Freq' in line:
                        try:
                            # Extract frequency
                            freq_match = re.search(r'Freq\s+(\d+)', line)
                            level_match = re.search(r'层级\s+(\d+)', line)
                            
                            if freq_match and level_match:
                                freq = int(freq_match.group(1))
                                level = int(level_match.group(1))
                                frequencies.append(freq)
                                levels.append(level)
                        except:
                            continue
        except:
            continue
        
        if frequencies and levels:
            ax.scatter(levels, frequencies, c=colors[i], marker=markers[i], 
                      s=100, alpha=0.7, label=db.upper(), edgecolors='black')
    
    ax.set_title('热门文章频率与HNSW层级关系', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('HNSW层级', fontsize=14, fontweight='bold')
    ax.set_ylabel('检索频率', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, title='数据库', title_fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Set integer ticks for x-axis
    ax.set_xticks(range(1, 4))
    
    plt.tight_layout()
    
    # Save chart
    os.makedirs('output/charts', exist_ok=True)
    output_file = 'output/charts/frequency_vs_hnsw_level.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"频率vs层级散点图已保存为: {output_file}")
    
    plt.show()

def main():
    """Main function"""
    print("开始收集HNSW层级数据...")
    data = collect_hnsw_data()
    
    print("\n开始绘制HNSW层级分布图...")
    create_level_distribution_chart(data)
    
    print("\n开始绘制HNSW层级对比图...")
    create_high_level_comparison_chart(data)
    
    print("\n开始绘制频率vs层级散点图...")
    create_frequency_vs_level_chart(data)
    
    print("HNSW层级图表绘制完成!")

if __name__ == "__main__":
    main()