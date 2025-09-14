#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
�ۺ�ͼ����ƹ���
���ڿ��ӻ� wikipead_all.py ���ɵĸ���ͳ������
"""

import os
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# ����matplotlib��������֧��
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # ��ȷ��ʾ����

# ȷ�����Ŀ¼����
os.makedirs("output/charts", exist_ok=True)

def extract_percentage_from_file(file_path, pattern):
    """��ͳ���ļ�����ȡ�ٷֱ�����"""
    if not os.path.exists(file_path):
        print(f"�ļ�������: {file_path}")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='gb2312') as f:
                content = f.read()
        except Exception as e:
            print(f"��ȡ�ļ�ʧ�� {file_path}: {e}")
            return None
    
    match = re.search(pattern, content)
    return float(match.group(1)) if match else None

def collect_frequency_data():
    """�ռ��������ݼ����ĵ�Ƶ�ʷֲ�����"""
    databases = ["mmlu", "nq", "hotpotqa", "triviaqa"]
    top_configs = ["top1", "top5", "top10"]
    
    data = defaultdict(dict)
    
    for db in databases:
        for config in top_configs:
            file_path = f"freq_stats_{db}_{config}.txt"
            if os.path.exists(file_path):
                pattern = r"Top 10% �ĵ�ռ�ܼ����� ([\d.]+)%"
                percentage = extract_percentage_from_file(file_path, pattern)
                if percentage is not None:
                    data[db][config] = percentage
                    print(f"��ȡ {db} {config}: {percentage}%")
    
    return data

def collect_ngram_data():
    """�ռ��������ݼ���n-gram�ֲ�����"""
    databases = ["mmlu", "nq", "hotpotqa", "triviaqa"]
    top_configs = ["top1", "top5", "top10"]
    ngram_sizes = [2, 3, 4]
    
    data = defaultdict(lambda: defaultdict(dict))
    
    for db in databases:
        for config in top_configs:
            for n in ngram_sizes:
                file_path = f"ngram_stats_n{n}_{db}_{config}.txt"
                if os.path.exists(file_path):
                    pattern = rf"Top 10% {n}-gram ռ�ܷ��ʵ� ([\d.]+)%"
                    percentage = extract_percentage_from_file(file_path, pattern)
                    if percentage is not None:
                        data[n][db][config] = percentage
                        print(f"��ȡ {n}-gram {db} {config}: {percentage}%")
    
    return data

def collect_high_level_data():
    """�ռ��߲�ڵ�ֲ�����"""
    databases = ["mmlu", "nq", "hotpotqa", "triviaqa"]
    top_configs = ["top1", "top5", "top10"]
    
    data = defaultdict(dict)
    
    for db in databases:
        for config in top_configs:
            file_path = f"high_level_stats_{db}_{config}.txt"
            if os.path.exists(file_path):
                pattern = r"Top 10% �����ĵ��и߲�ڵ�ռ��: ([\d.]+)%"
                percentage = extract_percentage_from_file(file_path, pattern)
                if percentage is not None:
                    data[db][config] = percentage
                    print(f"��ȡ�߲�ڵ���� {db} {config}: {percentage}%")
    
    return data

def create_frequency_comparison_chart(data):
    """�����ĵ�Ƶ�ʷֲ��Ա�ͼ��"""
    if not data:
        print("û���ҵ�Ƶ�ʷֲ�����")
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
        
        # �������������ֵ��ǩ
        for bar, value in zip(bars, values):
            if value > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{value:.1f}%', ha='center', va='bottom')
    
    ax.set_xlabel('���ݼ�')
    ax.set_ylabel('Top 10% �ĵ�ռ�ܼ����İٷֱ� (%)')
    ax.set_title('��ͬ���ݼ��ĵ�Ƶ�ʷֲ��Ա� - �����ĵ����жȷ���')
    ax.set_xticks(x + width)
    ax.set_xticklabels([db.upper() for db in databases])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max([max(data[db].values()) for db in databases if data[db]]) * 1.1)
    
    plt.tight_layout()
    plt.savefig('output/charts/frequency_distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("�ĵ�Ƶ�ʷֲ��Ա�ͼ�ѱ��浽 output/charts/frequency_distribution_comparison.png")

def create_ngram_comparison_chart(data):
    """����n-gram�ֲ��Ա�ͼ��"""
    if not data:
        print("û���ҵ�n-gram�ֲ�����")
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
            
            # �������������ֵ��ǩ
            for bar, value in zip(bars, values):
                if value > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('���ݼ�')
        ax.set_ylabel('Top 10% n-gram ռ�ܷ��ʵİٷֱ� (%)')
        ax.set_title(f'{n}-gram �ֲ��Ա�')
        ax.set_xticks(x + width)
        ax.set_xticklabels([db.upper() for db in databases])
        if idx == 0:
            ax.legend()
        ax.grid(True, alpha=0.3)
        
        # ����y�᷶Χ
        max_val = 0
        for db in databases:
            if db in data[n] and data[n][db]:
                max_val = max(max_val, max(data[n][db].values()))
        if max_val > 0:
            ax.set_ylim(0, max_val * 1.1)
    
    plt.tight_layout()
    plt.savefig('output/charts/ngram_distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("N-gram�ֲ��Ա�ͼ�ѱ��浽 output/charts/ngram_distribution_comparison.png")

def create_high_level_comparison_chart(data):
    """�����߲�ڵ�ֲ��Ա�ͼ��"""
    if not data:
        print("û���ҵ��߲�ڵ�ֲ�����")
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
        
        # �������������ֵ��ǩ
        for bar, value in zip(bars, values):
            if value > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{value:.1f}%', ha='center', va='bottom')
    
    ax.set_xlabel('���ݼ�')
    ax.set_ylabel('Top 10% �����ĵ��и߲�ڵ�ռ�� (%)')
    ax.set_title('��ͬ���ݼ�HNSW�߲�ڵ�ֲ��Ա� - �����ĵ��������еĲ㼶����')
    ax.set_xticks(x + width)
    ax.set_xticklabels([db.upper() for db in databases])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ����y�᷶Χ
    max_val = max([max(data[db].values()) for db in databases if data[db]]) if data else 0
    if max_val > 0:
        ax.set_ylim(0, max_val * 1.1)
    
    plt.tight_layout()
    plt.savefig('output/charts/high_level_distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("�߲�ڵ�ֲ��Ա�ͼ�ѱ��浽 output/charts/high_level_distribution_comparison.png")

def create_comprehensive_dashboard():
    """�����ۺ϶Ա��Ǳ��"""
    print("��ʼ�ռ��ۺ�����...")
    
    freq_data = collect_frequency_data()
    ngram_data = collect_ngram_data()
    high_level_data = collect_high_level_data()
    
    if not any([freq_data, ngram_data, high_level_data]):
        print("û���ҵ��κ�ͳ�������ļ�")
        return
    
    # �����ۺ϶Ա�ͼ
    fig = plt.figure(figsize=(20, 12))
    
    # 1. �ĵ�Ƶ�ʷֲ� (����)
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
        
        ax1.set_xlabel('���ݼ�')
        ax1.set_ylabel('Top 10% �ĵ�ռ�ܼ����İٷֱ� (%)')
        ax1.set_title('�ĵ�Ƶ�ʷֲ��Ա�')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels([db.upper() for db in databases])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. �߲�ڵ�ֲ� (����)
    if high_level_data:
        ax2 = plt.subplot(2, 3, 3)
        databases = list(high_level_data.keys())
        top_configs = ["top1", "top5", "top10"]  # ȷ����������
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # ȷ����������
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
        
        ax2.set_xlabel('���ݼ�')
        ax2.set_ylabel('�߲�ڵ�ռ�� (%)')
        ax2.set_title('HNSW�߲�ڵ�ֲ�')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels([db.upper() for db in databases])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. N-gram�ֲ� (�·�)
    if ngram_data:
        ngram_sizes = sorted(ngram_data.keys())
        for idx, n in enumerate(ngram_sizes):
            ax = plt.subplot(2, 3, 4 + idx)
            databases = ["mmlu", "nq", "hotpotqa", "triviaqa"]
            top_configs = ["top1", "top5", "top10"]  # ȷ����������
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # ȷ����������
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
            
            ax.set_xlabel('���ݼ�')
            ax.set_ylabel(f'{n}-gram ռ�� (%)')
            ax.set_title(f'{n}-gram �ֲ�')
            ax.set_xticks(x + width)
            ax.set_xticklabels([db.upper() for db in databases])
            if idx == 0:
                ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/charts/comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("�ۺ϶Ա��Ǳ���ѱ��浽 output/charts/comprehensive_dashboard.png")

def main():
    """������"""
    print("��ʼ�����ۺ�ͼ��...")
    
    # �ռ�����
    print("\n=== �ռ��ĵ�Ƶ�ʷֲ����� ===")
    freq_data = collect_frequency_data()
    
    print("\n=== �ռ�N-gram�ֲ����� ===")
    ngram_data = collect_ngram_data()
    
    print("\n=== �ռ��߲�ڵ�ֲ����� ===")
    high_level_data = collect_high_level_data()
    
    # ����ͼ��
    print("\n=== ����ͼ�� ===")
    if freq_data:
        create_frequency_comparison_chart(freq_data)
    
    if ngram_data:
        create_ngram_comparison_chart(ngram_data)
    
    if high_level_data:
        create_high_level_comparison_chart(high_level_data)
    
    # �����ۺ��Ǳ��
    print("\n=== �����ۺ��Ǳ�� ===")
    create_comprehensive_dashboard()
    
    print("\n����ͼ��������ɣ�")

if __name__ == "__main__":
    main()