#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
�����ݼ���ϸͼ����ƹ���
������������������ݼ��Ķ�ά��ͳ�ƽ��
"""

import os
import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# ����matplotlib��������֧��
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # ��ȷ��ʾ����

# ȷ�����Ŀ¼����
os.makedirs("output/charts", exist_ok=True)

def parse_frequency_stats(file_path):
    """����Ƶ��ͳ���ļ�����ȡtop-10�ĵ���Ϣ"""
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
            print(f"��ȡ�ļ�ʧ�� {file_path}: {e}")
            return None, None
    
    # ��ȡtop-10�ĵ�Ƶ��
    doc_frequencies = []
    lines = content.split('\n')
    for line in lines:
        if line.startswith('Rank'):
            match = re.search(r'Rank (\d+): Doc (\d+) - (\d+) ��', line)
            if match:
                rank, doc_id, freq = int(match.group(1)), int(match.group(2)), int(match.group(3))
                doc_frequencies.append((rank, doc_id, freq))
    
    # ��ȡtop 10%ռ��
    percentage_match = re.search(r'Top 10% �ĵ�ռ�ܼ����� ([\d.]+)%', content)
    percentage = float(percentage_match.group(1)) if percentage_match else None
    
    return doc_frequencies, percentage

def parse_ngram_stats(file_path):
    """����n-gramͳ���ļ�"""
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
            print(f"��ȡ�ļ�ʧ�� {file_path}: {e}")
            return None, None
    
    # ��ȡtop-10 n-gramƵ��
    ngram_frequencies = []
    lines = content.split('\n')
    for line in lines:
        if line.startswith('Rank'):
            match = re.search(r'Rank (\d+): (\d+) ��', line)
            if match:
                rank, freq = int(match.group(1)), int(match.group(2))
                ngram_frequencies.append((rank, freq))
    
    # ��ȡtop 10%ռ��
    n_value = re.search(r'(\d+)-gram', file_path)
    n = n_value.group(1) if n_value else "n"
    pattern = rf'Top 10% {n}-gram ռ�ܷ��ʵ� ([\d.]+)%'
    percentage_match = re.search(pattern, content)
    percentage = float(percentage_match.group(1)) if percentage_match else None
    
    return ngram_frequencies, percentage

def parse_high_level_stats(file_path):
    """�����߲�ڵ�ͳ���ļ�"""
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
            print(f"��ȡ�ļ�ʧ�� {file_path}: {e}")
            return None, None
    
    # ��ȡtop-10�ĵ��Ĳ㼶��Ϣ
    doc_levels = []
    lines = content.split('\n')
    for line in lines:
        if line.startswith('Rank'):
            match = re.search(r'Rank (\d+): Doc (\d+) \(Freq (\d+)\) - �߲�ڵ�: (True|False) \(�㼶 (\d+)\)', line)
            if match:
                rank = int(match.group(1))
                doc_id = int(match.group(2))
                freq = int(match.group(3))
                is_high_level = match.group(4) == 'True'
                level = int(match.group(5))
                doc_levels.append((rank, doc_id, freq, is_high_level, level))
    
    # ��ȡ�߲�ڵ�ռ��
    percentage_match = re.search(r'Top 10% �����ĵ��и߲�ڵ�ռ��: ([\d.]+)%', content)
    percentage = float(percentage_match.group(1)) if percentage_match else None
    
    return doc_levels, percentage

def create_single_dataset_dashboard(dataset, topk):
    """Ϊ�������ݼ������ۺϷ����Ǳ��"""
    print(f"���� {dataset.upper()} Top-{topk} ���ݼ��ۺϷ����Ǳ��...")
    
    # �ļ�·��
    freq_file = f"freq_stats_{dataset}_top{topk}.txt"
    high_level_file = f"high_level_stats_{dataset}_top{topk}.txt"
    ngram_files = {
        2: f"ngram_stats_n2_{dataset}_top{topk}.txt",
        3: f"ngram_stats_n3_{dataset}_top{topk}.txt",
        4: f"ngram_stats_n4_{dataset}_top{topk}.txt"
    }
    
    # ��������
    doc_frequencies, doc_percentage = parse_frequency_stats(freq_file)
    doc_levels, high_level_percentage = parse_high_level_stats(high_level_file)
    ngram_data = {}
    for n, file_path in ngram_files.items():
        frequencies, percentage = parse_ngram_stats(file_path)
        if frequencies and percentage:
            ngram_data[n] = (frequencies, percentage)
    
    # ����ͼ��
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'{dataset.upper()} Top-{topk} ���ݼ��ۺϷ����Ǳ��', fontsize=16, fontweight='bold')
    
    # 1. �ĵ�Ƶ�ʷֲ� (����)
    if doc_frequencies:
        ax1 = plt.subplot(2, 4, 1)
        ranks = [item[0] for item in doc_frequencies]
        freqs = [item[2] for item in doc_frequencies]
        
        bars = ax1.bar(ranks, freqs, color='#FF6B6B', alpha=0.7)
        ax1.set_xlabel('�ĵ�����')
        ax1.set_ylabel('����Ƶ��')
        ax1.set_title('Top-10 �ĵ�Ƶ�ʷֲ�')
        ax1.grid(True, alpha=0.3)
        
        # �����ֵ��ǩ
        for bar, freq in zip(bars, freqs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(freqs)*0.01,
                    str(freq), ha='center', va='bottom', fontsize=8)
        
        # ���ռ����Ϣ
        if doc_percentage:
            ax1.text(0.02, 0.98, f'Top 10% �ĵ�ռ��: {doc_percentage:.1f}%', 
                    transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. HNSW�㼶�ֲ� (����)
    if doc_levels:
        ax2 = plt.subplot(2, 4, 2)
        levels = [item[4] for item in doc_levels]
        colors = ['#FF6B6B' if item[3] else '#4ECDC4' for item in doc_levels]
        
        ranks = [item[0] for item in doc_levels]
        bars = ax2.bar(ranks, levels, color=colors, alpha=0.7)
        ax2.set_xlabel('�ĵ�����')
        ax2.set_ylabel('HNSW�㼶')
        ax2.set_title('Top-10 �ĵ�HNSW�㼶�ֲ�')
        ax2.grid(True, alpha=0.3)
        
        # ���ͼ��
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#FF6B6B', alpha=0.7, label='�߲�ڵ� (Level > 0)'),
                          Patch(facecolor='#4ECDC4', alpha=0.7, label='�ײ�ڵ� (Level = 0)')]
        ax2.legend(handles=legend_elements, loc='upper right')
        
        # ���ռ����Ϣ
        if high_level_percentage:
            ax2.text(0.02, 0.98, f'�߲�ڵ�ռ��: {high_level_percentage:.1f}%', 
                    transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 3. N-gram�ֲ��Ա� (�°벿��)
    if ngram_data:
        ngram_positions = [5, 6, 7]  # ��Ӧ2, 3, 4-gram
        colors_ngram = ['#45B7D1', '#96CEB4', '#FFEAA7']
        
        for idx, (n, (frequencies, percentage)) in enumerate(ngram_data.items()):
            if idx < len(ngram_positions):
                ax = plt.subplot(2, 4, ngram_positions[idx])
                ranks = [item[0] for item in frequencies]
                freqs = [item[1] for item in frequencies]
                
                bars = ax.bar(ranks, freqs, color=colors_ngram[idx], alpha=0.7)
                ax.set_xlabel('����')
                ax.set_ylabel('Ƶ��')
                ax.set_title(f'Top-10 {n}-gram �ֲ�')
                ax.grid(True, alpha=0.3)
                
                # �����ֵ��ǩ
                for bar, freq in zip(bars, freqs):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(freqs)*0.01,
                           str(freq), ha='center', va='bottom', fontsize=8)
                
                # ���ռ����Ϣ
                ax.text(0.02, 0.98, f'Top 10% ռ��: {percentage:.1f}%', 
                       transform=ax.transAxes, fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor=colors_ngram[idx], alpha=0.3))
    
    # 4. �ۺ϶Ա�����ͼ (����)
    ax4 = plt.subplot(2, 4, 8)
    categories = ['�ĵ�Ƶ��']
    percentages = []
    colors_summary = ['#FF6B6B']
    
    if doc_percentage:
        percentages.append(doc_percentage)
    
    if high_level_percentage:
        categories.append('�߲�ڵ�')
        percentages.append(high_level_percentage)
        colors_summary.append('#4ECDC4')
    
    for n, (_, percentage) in ngram_data.items():
        categories.append(f'{n}-gram')
        percentages.append(percentage)
        colors_summary.append(['#45B7D1', '#96CEB4', '#FFEAA7'][n-2])
    
    if percentages:
        bars = ax4.bar(categories, percentages, color=colors_summary, alpha=0.7)
        ax4.set_ylabel('Top 10% ռ�� (%)')
        ax4.set_title('��ά��Top 10%ռ�ȶԱ�')
        ax4.grid(True, alpha=0.3)
        
        # �����ֵ��ǩ
        for bar, percentage in zip(bars, percentages):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(percentages)*0.01,
                    f'{percentage:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # ��תx���ǩ�Է��ص�
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # ����ͼ��
    output_path = f'output/charts/{dataset}_top{topk}_dashboard.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"�����ݼ��Ǳ���ѱ��浽 {output_path}")

def create_log_log_plots(dataset, topk):
    """����Log-Log�ֲ�ͼ"""
    print(f"���� {dataset.upper()} Top-{topk} Log-Log�ֲ�ͼ...")
    
    # ���Ҷ�Ӧ�ķֲ�ͼ�ļ�
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
        plot_titles.append("�ĵ�Ƶ�ʷֲ�")
    
    if os.path.exists(high_level_plot):
        existing_plots.append(high_level_plot)
        plot_titles.append("�߲�ڵ�ֲ�")
    
    for i, ngram_plot in enumerate(ngram_plots):
        if os.path.exists(ngram_plot):
            existing_plots.append(ngram_plot)
            plot_titles.append(f"{i+2}-gram�ֲ�")
    
    if existing_plots:
        print(f"�ҵ� {len(existing_plots)} �����еķֲ�ͼ�ļ�")
        for plot, title in zip(existing_plots, plot_titles):
            print(f"  - {title}: {plot}")
    else:
        print("δ�ҵ����еķֲ�ͼ�ļ�")

def main():
    """������"""
    parser = argparse.ArgumentParser(description="�����ݼ���ϸͼ����ƹ���")
    parser.add_argument("--dataset", type=str, required=True, 
                       choices=["mmlu", "nq", "hotpotqa", "triviaqa"],
                       help="ѡ�����ݼ�")
    parser.add_argument("--topk", type=int, required=True,
                       help="ָ��top-kֵ")
    
    args = parser.parse_args()
    
    print(f"��ʼΪ {args.dataset.upper()} Top-{args.topk} ������ϸ����ͼ��...")
    
    # �����ۺ��Ǳ��
    create_single_dataset_dashboard(args.dataset, args.topk)
    
    # ���Log-Log�ֲ�ͼ
    create_log_log_plots(args.dataset, args.topk)
    
    print("ͼ��������ɣ�")

if __name__ == "__main__":
    main()