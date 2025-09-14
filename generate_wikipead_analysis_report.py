#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ϊ wikipead_all.py ��������������ϸ�ķ�������
����ͳ��ժҪ�����Ʒ������쳣������
"""

import os
import re
import numpy as np
from collections import defaultdict
from datetime import datetime
import argparse

def parse_arguments():
    """���������в���"""
    parser = argparse.ArgumentParser(description="Ϊwikipead_all.py�����������ɷ�������")
    parser.add_argument("--dataset", type=str, default="all", 
                       choices=["mmlu", "nq", "hotpotqa", "triviaqa", "all"],
                       help="ѡ�����ݼ����з��� (Ĭ��: all)")
    parser.add_argument("--topk", type=str, default="all",
                       help="ѡ��top-k���ý��з������� '10' �� 'all' (Ĭ��: all)")
    return parser.parse_args()

def extract_value_from_file(file_path, pattern):
    """��ͳ���ļ�����ȡ��ֵ"""
    if not os.path.exists(file_path):
        return None
    
    try:
        # ���Զ��ֱ�����ʽ
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
            return None
        
        match = re.search(pattern, content)
        return float(match.group(1)) if match else None
    except Exception:
        return None

def extract_top_frequencies(file_path, top_n=10):
    """��Ƶ��ͳ���ļ�����ȡǰN���ĵ�Ƶ��"""
    if not os.path.exists(file_path):
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='gb2312') as f:
                content = f.read()
        except Exception:
            return []
    
    # ��ȡ�������ĵ�ID��Ƶ��
    pattern = r"Rank (\d+): Doc (\d+) - (\d+) ��"
    matches = re.findall(pattern, content)
    
    frequencies = []
    for rank, doc_id, freq in matches[:top_n]:
        frequencies.append({
            'rank': int(rank),
            'doc_id': int(doc_id),
            'frequency': int(freq)
        })
    
    return frequencies

def analyze_frequency_distribution(datasets, topks):
    """�����ĵ�Ƶ�ʷֲ�����"""
    analysis = {}
    
    for dataset in datasets:
        analysis[dataset] = {}
        for topk in topks:
            file_path = f"freq_stats_{dataset}_top{topk}.txt"
            
            # ��ȡTop 10%ռ��
            pattern = r"Top 10% �ĵ�ռ�ܼ����� ([\d.]+)%"
            top10_percent = extract_value_from_file(file_path, pattern)
            
            # ��ȡǰ10���ĵ�Ƶ��
            top_frequencies = extract_top_frequencies(file_path, 10)
            
            if top10_percent is not None and top_frequencies:
                total_freq = sum([f['frequency'] for f in top_frequencies])
                
                # �����ֲ�����
                freqs = [f['frequency'] for f in top_frequencies]
                if len(freqs) > 1:
                    # ���㼯�ж�ָ��
                    max_freq = max(freqs)
                    min_freq = min(freqs)
                    concentration_ratio = max_freq / min_freq if min_freq > 0 else float('inf')
                    
                    # ��������ϵ��
                    cv = np.std(freqs) / np.mean(freqs) if np.mean(freqs) > 0 else 0
                    
                    # �����Ƿ��������ɷֲ�
                    log_ranks = np.log(range(1, len(freqs) + 1))
                    log_freqs = np.log(freqs)
                    correlation = np.corrcoef(log_ranks, log_freqs)[0, 1]
                    
                    analysis[dataset][f"top{topk}"] = {
                        'top10_percent': top10_percent,
                        'total_top10_freq': total_freq,
                        'max_frequency': max_freq,
                        'min_frequency': min_freq,
                        'concentration_ratio': concentration_ratio,
                        'coefficient_variation': cv,
                        'power_law_correlation': correlation,
                        'top_frequencies': top_frequencies
                    }
    
    return analysis

def analyze_ngram_patterns(datasets, topks):
    """����N-gramģʽ����"""
    analysis = defaultdict(lambda: defaultdict(dict))
    ngram_sizes = [2, 3, 4]
    
    for n in ngram_sizes:
        for dataset in datasets:
            for topk in topks:
                file_path = f"ngram_stats_n{n}_{dataset}_top{topk}.txt"
                pattern = rf"Top 10% {n}-gram ռ�ܷ��ʵ� ([\d.]+)%"
                percentage = extract_value_from_file(file_path, pattern)
                
                if percentage is not None:
                    analysis[n][dataset][f"top{topk}"] = percentage
    
    return analysis

def analyze_hnsw_characteristics(datasets, topks):
    """����HNSW��������"""
    analysis = {}
    
    for dataset in datasets:
        analysis[dataset] = {}
        for topk in topks:
            file_path = f"high_level_stats_{dataset}_top{topk}.txt"
            
            # ��ȡ�߲��ڵ�ռ��
            pattern = r"Top 10% �����ĵ��и߲��ڵ�ռ��: ([\d.]+)%"
            high_level_percent = extract_value_from_file(file_path, pattern)
            
            # ������ϸ�㼶�ֲ�
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    try:
                        with open(file_path, 'r', encoding='gb2312') as f:
                            content = f.read()
                    except Exception:
                        continue
                
                # ��ȡ�㼶��Ϣ
                level_pattern = r"Rank \d+: Doc \d+ \(Freq \d+\) - �߲��ڵ�: (\w+) \(�㼶 (\d+)\)"
                level_matches = re.findall(level_pattern, content)
                
                if level_matches and high_level_percent is not None:
                    levels = [int(level) for is_high, level in level_matches]
                    high_level_count = sum(1 for is_high, level in level_matches if is_high == "True")
                    
                    analysis[dataset][f"top{topk}"] = {
                        'high_level_percent': high_level_percent,
                        'high_level_count': high_level_count,
                        'avg_level': np.mean(levels) if levels else 0,
                        'max_level': max(levels) if levels else 0,
                        'level_distribution': dict(zip(*np.unique(levels, return_counts=True))) if levels else {}
                    }
    
    return analysis

def generate_comparative_insights(freq_analysis, ngram_analysis, hnsw_analysis):
    """���ɱȽϷ�������"""
    insights = []
    
    # 1. �����ݼ��ȶȼ��жȶԱ�
    if freq_analysis:
        datasets = list(freq_analysis.keys())
        top10_values = {}
        for dataset in datasets:
            if freq_analysis[dataset]:
                config_key = list(freq_analysis[dataset].keys())[0]  # ȡ��һ������
                top10_values[dataset] = freq_analysis[dataset][config_key]['top10_percent']
        
        if top10_values:
            max_dataset = max(top10_values.keys(), key=lambda k: top10_values[k])
            min_dataset = min(top10_values.keys(), key=lambda k: top10_values[k])
            insights.append(f"�ȶȼ��ж����ߵ����ݼ�: {max_dataset.upper()} ({top10_values[max_dataset]:.1f}%)")
            insights.append(f"�ȶȷֲ������ȵ����ݼ�: {min_dataset.upper()} ({top10_values[min_dataset]:.1f}%)")
            
            avg_concentration = np.mean(list(top10_values.values()))
            insights.append(f"ƽ���ȶȼ��ж�: {avg_concentration:.1f}%")
    
    # 2. N-gramģʽ����
    if ngram_analysis:
        for n in sorted(ngram_analysis.keys()):
            ngram_data = ngram_analysis[n]
            if ngram_data:
                dataset_avgs = {}
                for dataset in ngram_data:
                    if ngram_data[dataset]:
                        values = list(ngram_data[dataset].values())
                        dataset_avgs[dataset] = np.mean(values)
                
                if dataset_avgs:
                    max_dataset = max(dataset_avgs.keys(), key=lambda k: dataset_avgs[k])
                    insights.append(f"{n}-gram���м��ж�����: {max_dataset.upper()} ({dataset_avgs[max_dataset]:.1f}%)")
    
    # 3. HNSW�㼶��������
    if hnsw_analysis:
        high_level_ratios = {}
        avg_levels = {}
        for dataset in hnsw_analysis:
            if hnsw_analysis[dataset]:
                config_key = list(hnsw_analysis[dataset].keys())[0]
                high_level_ratios[dataset] = hnsw_analysis[dataset][config_key]['high_level_percent']
                avg_levels[dataset] = hnsw_analysis[dataset][config_key]['avg_level']
        
        if high_level_ratios:
            max_high_level = max(high_level_ratios.keys(), key=lambda k: high_level_ratios[k])
            insights.append(f"�߲��ڵ���������: {max_high_level.upper()} ({high_level_ratios[max_high_level]:.1f}%)")
        
        if avg_levels:
            max_avg_level = max(avg_levels.keys(), key=lambda k: avg_levels[k])
            insights.append(f"ƽ���㼶����: {max_avg_level.upper()} ({avg_levels[max_avg_level]:.2f})")
    
    return insights

def generate_recommendations(freq_analysis, ngram_analysis, hnsw_analysis):
    """���ڷ������������Ż�����"""
    recommendations = []
    
    # ����Ƶ�ʷֲ��Ľ���
    if freq_analysis:
        high_concentration_datasets = []
        for dataset in freq_analysis:
            if freq_analysis[dataset]:
                config_key = list(freq_analysis[dataset].keys())[0]
                if freq_analysis[dataset][config_key]['top10_percent'] > 50:
                    high_concentration_datasets.append(dataset)
        
        if high_concentration_datasets:
            recommendations.append(f"�߶ȼ��е����ݼ� ({', '.join(high_concentration_datasets)}) ��Ҫ����:")
            recommendations.append("  - ���Ӽ��������Բ���")
            recommendations.append("  - ʵʩ�ĵ�ȥ�ػ�ȥƫ����")
            recommendations.append("  - �Ż�embeddingģ�����������ƶȼ��㾫��")
    
    # ����N-gramģʽ�Ľ���
    if ngram_analysis:
        for n in sorted(ngram_analysis.keys()):
            high_ngram_datasets = []
            for dataset in ngram_analysis[n]:
                if ngram_analysis[n][dataset]:
                    avg_percent = np.mean(list(ngram_analysis[n][dataset].values()))
                    if avg_percent > 30:  # ����30%Ϊ��ֵ
                        high_ngram_datasets.append(dataset)
            
            if high_ngram_datasets:
                recommendations.append(f"{n}-gram���и߶��ظ������ݼ���Ҫ:")
                recommendations.append("  - ������ѯģʽ�Ƿ���������")
                recommendations.append("  - �������Ӳ�ѯ������")
    
    # ����HNSW�����Ľ���
    if hnsw_analysis:
        low_high_level_datasets = []
        for dataset in hnsw_analysis:
            if hnsw_analysis[dataset]:
                config_key = list(hnsw_analysis[dataset].keys())[0]
                if hnsw_analysis[dataset][config_key]['high_level_percent'] < 10:
                    low_high_level_datasets.append(dataset)
        
        if low_high_level_datasets:
            recommendations.append(f"�߲��ڵ������ϵ͵����ݼ� ({', '.join(low_high_level_datasets)}) ���ܱ���:")
            recommendations.append("  - �����ĵ���HNSW�����в���ͻ��")
            recommendations.append("  - ���Կ��ǵ���HNSW��������")
            recommendations.append("  - ��������������Ҫ�Ż�")
    
    return recommendations

def generate_analysis_report(datasets, topks, output_file="output/charts/wikipead_analysis_report.txt"):
    """���������ķ�������"""
    # �ռ���������
    freq_analysis = analyze_frequency_distribution(datasets, topks)
    ngram_analysis = analyze_ngram_patterns(datasets, topks)
    hnsw_analysis = analyze_hnsw_characteristics(datasets, topks)
    
    # ���ɶ����ͽ���
    insights = generate_comparative_insights(freq_analysis, ngram_analysis, hnsw_analysis)
    recommendations = generate_recommendations(freq_analysis, ngram_analysis, hnsw_analysis)
    
    # ȷ������Ŀ¼����
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # ���ɱ���
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("WIKIPEAD_ALL.PY ���ݷ�������\n")
        f.write("=" * 80 + "\n")
        f.write(f"����ʱ��: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"�������ݼ�: {', '.join([d.upper() for d in datasets])}\n")
        f.write(f"Top-K����: {', '.join([f'top{k}' for k in topks])}\n")
        f.write("\n")
        
        # 1. ִ��ժҪ
        f.write("1. ִ��ժҪ\n")
        f.write("-" * 40 + "\n")
        if insights:
            for insight in insights[:5]:  # ǰ5���ؼ�����
                f.write(f"? {insight}\n")
        f.write("\n")
        
        # 2. �ĵ�Ƶ�ʷֲ�����
        if freq_analysis:
            f.write("2. �ĵ�Ƶ�ʷֲ�����\n")
            f.write("-" * 40 + "\n")
            for dataset in datasets:
                if dataset in freq_analysis and freq_analysis[dataset]:
                    f.write(f"\n{dataset.upper()}���ݼ�:\n")
                    for config in freq_analysis[dataset]:
                        data = freq_analysis[dataset][config]
                        f.write(f"  {config}:\n")
                        f.write(f"    Top 10%ռ��: {data['top10_percent']:.1f}%\n")
                        f.write(f"    ����Ƶ��: {data['max_frequency']}\n")
                        f.write(f"    ���жȱ�ֵ: {data['concentration_ratio']:.2f}\n")
                        f.write(f"    ����ϵ��: {data['coefficient_variation']:.3f}\n")
                        f.write(f"    ����������: {data['power_law_correlation']:.3f}\n")
        f.write("\n")
        
        # 3. N-gram���з���
        if ngram_analysis:
            f.write("3. N-gram���з���\n")
            f.write("-" * 40 + "\n")
            for n in sorted(ngram_analysis.keys()):
                f.write(f"\n{n}-gram����:\n")
                for dataset in datasets:
                    if dataset in ngram_analysis[n] and ngram_analysis[n][dataset]:
                        f.write(f"  {dataset.upper()}: ")
                        values = list(ngram_analysis[n][dataset].values())
                        f.write(f"ƽ�� {np.mean(values):.1f}% (��Χ: {min(values):.1f}%-{max(values):.1f}%)\n")
        f.write("\n")
        
        # 4. HNSW������������
        if hnsw_analysis:
            f.write("4. HNSW������������\n")
            f.write("-" * 40 + "\n")
            for dataset in datasets:
                if dataset in hnsw_analysis and hnsw_analysis[dataset]:
                    f.write(f"\n{dataset.upper()}���ݼ�:\n")
                    for config in hnsw_analysis[dataset]:
                        data = hnsw_analysis[dataset][config]
                        f.write(f"  {config}:\n")
                        f.write(f"    �߲��ڵ�ռ��: {data['high_level_percent']:.1f}%\n")
                        f.write(f"    �߲��ڵ�����: {data['high_level_count']}\n")
                        f.write(f"    ƽ���㼶: {data['avg_level']:.2f}\n")
                        f.write(f"    ���߲㼶: {data['max_level']}\n")
        f.write("\n")
        
        # 5. �ؼ�����
        f.write("5. �ؼ�����\n")
        f.write("-" * 40 + "\n")
        for insight in insights:
            f.write(f"? {insight}\n")
        f.write("\n")
        
        # 6. �Ż�����
        f.write("6. �Ż�����\n")
        f.write("-" * 40 + "\n")
        for recommendation in recommendations:
            f.write(f"{recommendation}\n")
        f.write("\n")
        
        # 7. ����˵��
        f.write("7. ����˵��\n")
        f.write("-" * 40 + "\n")
        f.write("? ���жȱ�ֵ: ����Ƶ��/����Ƶ�ʣ�ֵԽ����ʾ�ֲ�Խ������\n")
        f.write("? ����ϵ��: ��׼��/��ֵ������Ƶ�ʷֲ������Ա����̶�\n")
        f.write("? ����������: log(rank)��log(frequency)������ϵ�����ӽ�-1��ʾ�������ɷֲ�\n")
        f.write("? �߲��ڵ�: HNSW�����в㼶>0�Ľڵ㣬ͨ������Ҫ��hub�ڵ�\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("��������\n")
        f.write("=" * 80 + "\n")
    
    print(f"���������ѱ��浽: {output_file}")
    return output_file

def main():
    """������"""
    args = parse_arguments()
    
    # ȷ��Ҫ���������ݼ���top-k����
    if args.dataset == "all":
        datasets = ["mmlu", "nq", "hotpotqa", "triviaqa"]
    else:
        datasets = [args.dataset]
    
    if args.topk == "all":
        # �Զ��������õ�top-k����
        topks = []
        for k in [1, 5, 10]:
            test_file = f"freq_stats_{datasets[0]}_top{k}.txt"
            if os.path.exists(test_file):
                topks.append(k)
        if not topks:
            topks = [10]  # Ĭ��ֵ
    else:
        try:
            topks = [int(args.topk)]
        except ValueError:
            print(f"��Ч��top-kֵ: {args.topk}")
            return
    
    print(f"��ʼΪ wikipead_all.py ���ɷ�������...")
    print(f"���ݼ�: {datasets}")
    print(f"Top-K����: {topks}")
    
    # ���ɷ�������
    output_suffix = f"_{args.dataset}_top{args.topk}" if args.dataset != "all" or args.topk != "all" else ""
    output_file = f"output/charts/wikipead_analysis_report{output_suffix}.txt"
    
    generate_analysis_report(datasets, topks, output_file)
    
    print("���������������ɣ�")

if __name__ == "__main__":
    main()