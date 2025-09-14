#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整理实验输出文件到分组文件夹结构
按照知识库数据集、QA数据集、实验类型、Top-K配置进行分层组织
"""

import os
import shutil
import glob
import re

def create_directory_structure():
    """创建目录结构"""
    datasets = ["mmlu", "nq", "hotpotqa", "triviaqa"]
    
    # 创建Wikipedia100k输出结构
    for dataset in datasets:
        base_path = f"生成输出/wikipedia100k输出/{dataset}数据集"
        
        # 文档热度分析
        for topk in [10, 16, 32]:
            os.makedirs(f"{base_path}/文档热度分析/top{topk}实验", exist_ok=True)
        
        # N-gram序列分析
        for topk in [10, 16, 32]:
            os.makedirs(f"{base_path}/Ngram序列分析/top{topk}实验", exist_ok=True)
        
        # HNSW索引分析
        for topk in [10, 16, 32]:
            os.makedirs(f"{base_path}/HNSW索引分析/top{topk}实验", exist_ok=True)
        
        # 综合对比分析
        os.makedirs(f"{base_path}/综合对比分析", exist_ok=True)
    
    # 创建Wikipedia3.2k输出结构
    for dataset in datasets:
        base_path = f"生成输出/wikipedia3.2k输出/{dataset}数据集"
        
        # 文档热度分析
        for topk in [1, 5, 10]:
            os.makedirs(f"{base_path}/文档热度分析/top{topk}实验", exist_ok=True)
        
        # 文档对组合分析
        for topk in [3, 5, 10]:
            os.makedirs(f"{base_path}/文档对组合分析/top{topk}实验", exist_ok=True)
        
        # 综合对比分析
        os.makedirs(f"{base_path}/综合对比分析", exist_ok=True)
    
    print("目录结构创建完成")

def get_friendly_filename(filename):
    """根据文件类型重命名文件"""
    filename_lower = filename.lower()
    
    # 热门文档相关
    if "hot_docs" in filename_lower or "freq_stats" in filename_lower:
        if filename.endswith(".txt"):
            return "热门文档统计.txt"
        else:
            return "热门文档分布图.png"
    
    # N-gram相关  
    elif "ngram" in filename_lower:
        if "n2" in filename_lower:
            return "2-gram分析.png" if filename.endswith(".png") else "2-gram统计.txt"
        elif "n3" in filename_lower:
            return "3-gram分析.png" if filename.endswith(".png") else "3-gram统计.txt"
        elif "n4" in filename_lower:
            return "4-gram分析.png" if filename.endswith(".png") else "4-gram统计.txt"
        else:
            return "N-gram分析.png" if filename.endswith(".png") else "N-gram统计.txt"
    
    # HNSW高层节点相关
    elif "high_level" in filename_lower:
        if filename.endswith(".txt"):
            return "HNSW高层节点统计.txt"
        else:
            return "HNSW高层节点分布图.png"
    
    # 文档对组合相关
    elif "ordered_combo" in filename_lower:
        return "有序文档对统计.txt" if filename.endswith(".txt") else "有序文档对分布图.png"
    elif "unordered_combo" in filename_lower:
        return "无序文档对统计.txt" if filename.endswith(".txt") else "无序文档对分布图.png"
    
    # 综合对比相关
    elif "comprehensive" in filename_lower or "dashboard" in filename_lower:
        return "综合对比仪表板.png"
    elif "frequency" in filename_lower and "distribution" in filename_lower:
        return "频率分布对比图.png"
    elif "comparison" in filename_lower:
        return "对比分析图.png"
    
    # 保持原始文件名
    return filename

def copy_file_with_info(src, dst):
    """复制文件并显示信息"""
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  复制: {src} -> {dst}")
        return True
    except Exception as e:
        print(f"  错误: 复制文件失败 {src}: {e}")
        return False

def organize_wikipedia100k_files():
    """整理Wikipedia100k相关文件"""
    datasets = ["mmlu", "nq", "hotpotqa", "triviaqa"]
    
    print("=== 整理Wikipedia100k输出文件 ===")
    
    for dataset in datasets:
        print(f"处理{dataset}数据集...")
        
        # 处理文档热度分析文件
        for topk in [10, 16, 32]:
            target_dir = f"生成输出/wikipedia100k输出/{dataset}数据集/文档热度分析/top{topk}实验"
            
            files_to_copy = [
                f"hot_docs_distribution_{dataset}_top{topk}.png",
                f"freq_stats_{dataset}_top{topk}.txt"
            ]
            
            for file_path in files_to_copy:
                if os.path.exists(file_path):
                    new_name = get_friendly_filename(os.path.basename(file_path))
                    target_path = os.path.join(target_dir, new_name)
                    copy_file_with_info(file_path, target_path)
        
        # 处理N-gram序列分析文件
        for topk in [10, 16, 32]:
            target_dir = f"生成输出/wikipedia100k输出/{dataset}数据集/Ngram序列分析/top{topk}实验"
            
            for n in [2, 3, 4]:
                files_to_copy = [
                    f"ngram_stats_n{n}_{dataset}_top{topk}.txt",
                    f"ngram_distribution_n{n}_{dataset}_top{topk}.png"
                ]
                
                for file_path in files_to_copy:
                    if os.path.exists(file_path):
                        new_name = get_friendly_filename(os.path.basename(file_path))
                        target_path = os.path.join(target_dir, new_name)
                        copy_file_with_info(file_path, target_path)
        
        # 处理HNSW索引分析文件
        for topk in [10, 16, 32]:
            target_dir = f"生成输出/wikipedia100k输出/{dataset}数据集/HNSW索引分析/top{topk}实验"
            
            files_to_copy = [
                f"high_level_stats_{dataset}_top{topk}.txt",
                f"high_level_distribution_{dataset}_top{topk}.png"
            ]
            
            for file_path in files_to_copy:
                if os.path.exists(file_path):
                    new_name = get_friendly_filename(os.path.basename(file_path))
                    target_path = os.path.join(target_dir, new_name)
                    copy_file_with_info(file_path, target_path)
        
        # 处理综合对比分析文件
        target_dir = f"生成输出/wikipedia100k输出/{dataset}数据集/综合对比分析"
        
        # 查找output/charts/下的wikipead_开头的文件
        chart_files = glob.glob("output/charts/wikipead_*.png")
        for file_path in chart_files:
            if dataset in file_path or "all" in file_path:
                new_name = get_friendly_filename(os.path.basename(file_path))
                target_path = os.path.join(target_dir, new_name)
                copy_file_with_info(file_path, target_path)

def organize_wikipedia32k_files():
    """整理Wikipedia3.2k相关文件"""
    datasets = ["mmlu", "nq", "hotpotqa", "triviaqa"]
    
    print("=== 整理Wikipedia3.2k输出文件 ===")
    
    for dataset in datasets:
        print(f"处理{dataset}数据集...")
        
        # 处理文档热度分析文件
        for topk in [1, 5, 10]:
            target_dir = f"生成输出/wikipedia3.2k输出/{dataset}数据集/文档热度分析/top{topk}实验"
            
            # data/stats中的文件
            stats_files = [f"data/stats/freq_stats_{dataset}_top{topk}.txt"]
            # output/charts中的文件
            chart_files = [f"output/charts/hot_docs_distribution_{dataset}_top{topk}.png"]
            
            all_files = stats_files + chart_files
            
            for file_path in all_files:
                if os.path.exists(file_path):
                    new_name = get_friendly_filename(os.path.basename(file_path))
                    target_path = os.path.join(target_dir, new_name)
                    copy_file_with_info(file_path, target_path)
        
        # 处理文档对组合分析文件
        for topk in [3, 5, 10]:
            target_dir = f"生成输出/wikipedia3.2k输出/{dataset}数据集/文档对组合分析/top{topk}实验"
            
            # data/stats中的文件
            stats_files = [
                f"data/stats/ordered_combo_stats_{dataset}_top{topk}.txt",
                f"data/stats/unordered_combo_stats_{dataset}_top{topk}.txt"
            ]
            # output/charts中的文件
            chart_files = [
                f"output/charts/ordered_combo_distribution_{dataset}_top{topk}.png",
                f"output/charts/unordered_combo_distribution_{dataset}_top{topk}.png"
            ]
            
            all_files = stats_files + chart_files
            
            for file_path in all_files:
                if os.path.exists(file_path):
                    new_name = get_friendly_filename(os.path.basename(file_path))
                    target_path = os.path.join(target_dir, new_name)
                    copy_file_with_info(file_path, target_path)
        
        # 处理综合对比分析文件
        target_dir = f"生成输出/wikipedia3.2k输出/{dataset}数据集/综合对比分析"
        
        # 通用对比图表
        general_files = [
            "combo_comparison_chart.png",
            "combo_side_by_side_chart.png",
            "database_comparison_chart.png",
            "ngram_comparison_chart.png",
            "output/charts/combo_comparison_chart.png",
            "output/charts/combo_side_by_side_chart.png",
            "output/charts/database_comparison_chart.png",
            "output/charts/ngram_heatmap.png",
            "output/charts/ngram_trend_chart.png"
        ]
        
        for file_path in general_files:
            if os.path.exists(file_path):
                new_name = get_friendly_filename(os.path.basename(file_path))
                target_path = os.path.join(target_dir, new_name)
                copy_file_with_info(file_path, target_path)

def main():
    """主函数"""
    print("开始整理实验输出文件...")
    
    # 1. 创建目录结构
    create_directory_structure()
    
    # 2. 整理Wikipedia100k文件
    organize_wikipedia100k_files()
    
    # 3. 整理Wikipedia3.2k文件
    organize_wikipedia32k_files()
    
    print("\n文件整理完成！")
    print("目录结构:")
    print("生成输出/")
    print("├── wikipedia100k输出/")
    print("│   └── [数据集]/")
    print("│       ├── 文档热度分析/")
    print("│       │   ├── top10实验/")
    print("│       │   ├── top16实验/")
    print("│       │   └── top32实验/")
    print("│       ├── Ngram序列分析/")
    print("│       ├── HNSW索引分析/")
    print("│       └── 综合对比分析/")
    print("└── wikipedia3.2k输出/")
    print("    └── [数据集]/")
    print("        ├── 文档热度分析/")
    print("        │   ├── top1实验/")
    print("        │   ├── top5实验/")
    print("        │   └── top10实验/")
    print("        ├── 文档对组合分析/")
    print("        └── 综合对比分析/")

if __name__ == "__main__":
    main()