#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制四个数据库（hotpotqa/mmlu/nq/triviaqa）在不同top配置下的
unordered combo 和 ordered combo 的前10%组合占比图表
"""

import matplotlib.pyplot as plt
import numpy as np
import re
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def extract_top10_percentage(file_path):
    """
    从combo统计文件中提取前10%组合占总频率的占比
    
    Args:
        file_path (str): 文件路径
    
    Returns:
        float: 前10%组合占比（百分比数值）
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
        # 匹配形如 "Top 10% 组合出现占总频率的 XX.XX%" 的模式
        patterns = [
            r'Top 10% .*?(\d+\.\d+)%',
            r'前10%.*?(\d+\.\d+)%',
            r'(\d+\.\d+)%'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                return float(match.group(1))
        
        print(f"警告: 无法在文件 {file_path} 中找到Top 10%占比信息")
        return 0.0
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return 0.0
    except Exception as e:
        print(f"错误: 读取文件 {file_path} 时发生异常: {e}")
        return 0.0

def collect_combo_data():
    """
    收集所有数据库和top配置的combo数据
    
    Returns:
        dict: 包含ordered和unordered combo数据的字典
    """
    databases = ['hotpotqa', 'mmlu', 'nq', 'triviaqa']
    top_configs = ['top3', 'top5', 'top10']  # combo文件使用top3而不是top1
    combo_types = ['ordered', 'unordered']
    
    data = {}
    
    for combo_type in combo_types:
        data[combo_type] = {}
        for db in databases:
            data[combo_type][db] = {}
            for top in top_configs:
                file_path = f'data/stats/{combo_type}_combo_stats_{db}_{top}.txt'
                percentage = extract_top10_percentage(file_path)
                data[combo_type][db][top] = percentage
                print(f"{combo_type} {db} {top}: {percentage}%")
    
    return data

def create_combo_chart(data):
    """
    创建并绘制combo对比图表
    
    Args:
        data (dict): 包含所有combo数据的字典
    """
    databases = ['hotpotqa', 'mmlu', 'nq', 'triviaqa']
    top_configs = ['top3', 'top5', 'top10']
    combo_types = ['ordered', 'unordered']
    
    # 创建子图 - 上下两个图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # 设置柱状图的宽度和位置
    bar_width = 0.25
    x_positions = np.arange(len(databases))
    
    # 颜色设置
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝色、橙色、绿色
    
    # 绘制ordered combo图表
    ax1.set_title('Ordered Combo - 四个数据库在不同Top配置下的前10%组合占比', 
                  fontsize=16, fontweight='bold', pad=20)
    
    for i, top in enumerate(top_configs):
        values = [data['ordered'][db][top] for db in databases]
        bars = ax1.bar(x_positions + i * bar_width, values, bar_width, 
                      label=top.upper(), color=colors[i], alpha=0.8)
        
        # 在柱子上添加数值标签
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('数据库', fontsize=12, fontweight='bold')
    ax1.set_ylabel('前10%组合占总频率的占比 (%)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_positions + bar_width)
    ax1.set_xticklabels(databases, fontsize=11)
    max_ordered = max([max(data['ordered'][db].values()) for db in databases])
    ax1.set_ylim(0, max_ordered + 3)
    ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax1.legend(title='Top配置', fontsize=10, title_fontsize=11, loc='upper right')
    
    # 绘制unordered combo图表
    ax2.set_title('Unordered Combo - 四个数据库在不同Top配置下的前10%组合占比', 
                  fontsize=16, fontweight='bold', pad=20)
    
    for i, top in enumerate(top_configs):
        values = [data['unordered'][db][top] for db in databases]
        bars = ax2.bar(x_positions + i * bar_width, values, bar_width, 
                      label=top.upper(), color=colors[i], alpha=0.8)
        
        # 在柱子上添加数值标签
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax2.set_xlabel('数据库', fontsize=12, fontweight='bold')
    ax2.set_ylabel('前10%组合占总频率的占比 (%)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_positions + bar_width)
    ax2.set_xticklabels(databases, fontsize=11)
    max_unordered = max([max(data['unordered'][db].values()) for db in databases])
    ax2.set_ylim(0, max_unordered + 3)
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax2.legend(title='Top配置', fontsize=10, title_fontsize=11, loc='upper right')
    
    # 调整布局
    plt.tight_layout()
    
    # 确保输出目录存在
    os.makedirs('output/charts', exist_ok=True)
    
    # 保存图表
    output_file = 'output/charts/combo_comparison_chart.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Combo对比图表已保存为: {output_file}")
    
    # 显示图表
    plt.show()

def create_side_by_side_chart(data):
    """
    创建并绘制左右对比的combo图表
    
    Args:
        data (dict): 包含所有combo数据的字典
    """
    databases = ['hotpotqa', 'mmlu', 'nq', 'triviaqa']
    top_configs = ['top3', 'top5', 'top10']
    
    # 创建子图 - 左右两个图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # 设置柱状图的宽度和位置
    bar_width = 0.25
    x_positions = np.arange(len(databases))
    
    # 颜色设置
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝色、橙色、绿色
    
    # 绘制ordered combo图表
    ax1.set_title('Ordered Combo\n前10%组合占比对比', 
                  fontsize=14, fontweight='bold', pad=20)
    
    for i, top in enumerate(top_configs):
        values = [data['ordered'][db][top] for db in databases]
        bars = ax1.bar(x_positions + i * bar_width, values, bar_width, 
                      label=top.upper(), color=colors[i], alpha=0.8)
        
        # 在柱子上添加数值标签
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('数据库', fontsize=12, fontweight='bold')
    ax1.set_ylabel('前10%组合占总频率的占比 (%)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_positions + bar_width)
    ax1.set_xticklabels(databases, fontsize=11)
    max_ordered = max([max(data['ordered'][db].values()) for db in databases])
    ax1.set_ylim(0, max_ordered + 3)
    ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax1.legend(title='Top配置', fontsize=10, title_fontsize=11, loc='upper right')
    
    # 绘制unordered combo图表
    ax2.set_title('Unordered Combo\n前10%组合占比对比', 
                  fontsize=14, fontweight='bold', pad=20)
    
    for i, top in enumerate(top_configs):
        values = [data['unordered'][db][top] for db in databases]
        bars = ax2.bar(x_positions + i * bar_width, values, bar_width, 
                      label=top.upper(), color=colors[i], alpha=0.8)
        
        # 在柱子上添加数值标签
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax2.set_xlabel('数据库', fontsize=12, fontweight='bold')
    ax2.set_ylabel('前10%组合占总频率的占比 (%)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_positions + bar_width)
    ax2.set_xticklabels(databases, fontsize=11)
    max_unordered = max([max(data['unordered'][db].values()) for db in databases])
    ax2.set_ylim(0, max_unordered + 3)
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax2.legend(title='Top配置', fontsize=10, title_fontsize=11, loc='upper right')
    
    # 调整布局
    plt.tight_layout()
    
    # 确保输出目录存在
    os.makedirs('output/charts', exist_ok=True)
    
    # 保存图表
    output_file = 'output/charts/combo_side_by_side_chart.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Combo左右对比图表已保存为: {output_file}")
    
    # 显示图表
    plt.show()

def main():
    """
    主函数
    """
    print("开始收集combo数据...")
    data = collect_combo_data()
    
    print("\n收集到的数据:")
    for combo_type in data:
        print(f"\n{combo_type.capitalize()} Combo:")
        for db in data[combo_type]:
            print(f"  {db}: {data[combo_type][db]}")
    
    print("\n开始绘制上下对比图表...")
    create_combo_chart(data)
    
    print("\n开始绘制左右对比图表...")
    create_side_by_side_chart(data)
    
    print("Combo图表绘制完成!")

if __name__ == "__main__":
    main()