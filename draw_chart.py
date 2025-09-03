#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制四个数据库（hotpotqa/mmlu/nq/triviaqa）在不同top配置下的前10%文档占比图表
"""

import matplotlib.pyplot as plt
import numpy as np
import re
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def extract_top10_percentage(file_path):
    """
    从频率统计文件中提取前10%文档占总检索的占比
    
    Args:
        file_path (str): 文件路径
    
    Returns:
        float: 前10%文档占比（百分比数值）
    """
    try:
        # 尝试多种编码格式
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
            print(f"错误: 无法用任何编码格式读取文件 {file_path}")
            return 0.0
        
        # 使用正则表达式提取百分比数值
        pattern = r'Top 10% 文档占总检索的 (\d+\.\d+)%'
        match = re.search(pattern, content)
        
        if match:
            return float(match.group(1))
        else:
            print(f"警告: 无法在文件 {file_path} 中找到Top 10%占比信息")
            return 0.0
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return 0.0
    except Exception as e:
        print(f"错误: 读取文件 {file_path} 时发生异常: {e}")
        return 0.0

def collect_data():
    """
    收集所有数据库和top配置的数据
    
    Returns:
        dict: 包含所有数据的字典
    """
    databases = ['hotpotqa', 'mmlu', 'nq', 'triviaqa']
    top_configs = ['top1', 'top5', 'top10']
    
    data = {}
    
    for db in databases:
        data[db] = {}
        for top in top_configs:
            file_path = f'data/stats/freq_stats_{db}_{top}.txt'
            percentage = extract_top10_percentage(file_path)
            data[db][top] = percentage
            print(f"{db} {top}: {percentage}%")
    
    return data

def create_chart(data):
    """
    创建并绘制图表
    
    Args:
        data (dict): 包含所有数据的字典
    """
    databases = ['hotpotqa', 'mmlu', 'nq', 'triviaqa']
    top_configs = ['top1', 'top5', 'top10']
    
    # 设置图表大小和样式
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 设置柱状图的宽度和位置
    bar_width = 0.25
    x_positions = np.arange(len(databases))
    
    # 颜色设置
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝色、橙色、绿色
    
    # 绘制柱状图
    for i, top in enumerate(top_configs):
        values = [data[db][top] for db in databases]
        bars = ax.bar(x_positions + i * bar_width, values, bar_width, 
                     label=top.upper(), color=colors[i], alpha=0.8)
        
        # 在柱子上添加数值标签
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 设置图表标题和标签
    ax.set_title('四个数据库在不同Top配置下的前10%文档占比对比', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('数据库', fontsize=14, fontweight='bold')
    ax.set_ylabel('前10%文档占总检索的占比 (%)', fontsize=14, fontweight='bold')
    
    # 设置x轴标签
    ax.set_xticks(x_positions + bar_width)
    ax.set_xticklabels(databases, fontsize=12)
    
    # 设置y轴范围和网格
    ax.set_ylim(0, max([max(data[db].values()) for db in databases]) + 5)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # 设置图例
    ax.legend(title='Top配置', fontsize=11, title_fontsize=12, loc='upper right')
    
    # 调整布局
    plt.tight_layout()
    
    # 确保输出目录存在
    os.makedirs('output/charts', exist_ok=True)
    
    # 保存图表
    output_file = 'output/charts/database_comparison_chart.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"图表已保存为: {output_file}")
    
    # 显示图表
    plt.show()

def main():
    """
    主函数
    """
    print("开始收集数据...")
    data = collect_data()
    
    print("\n收集到的数据:")
    for db in data:
        print(f"{db}: {data[db]}")
    
    print("\n开始绘制图表...")
    create_chart(data)
    
    print("图表绘制完成!")

if __name__ == "__main__":
    main()