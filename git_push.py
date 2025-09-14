#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Git操作脚本 - 推送整理后的文件到GitHub
"""

import subprocess
import os

def run_command(cmd):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8')
        print(f"命令: {cmd}")
        print(f"返回码: {result.returncode}")
        if result.stdout:
            print(f"输出: {result.stdout}")
        if result.stderr:
            print(f"错误: {result.stderr}")
        print("-" * 50)
        return result.returncode == 0
    except Exception as e:
        print(f"执行命令失败: {cmd}, 错误: {e}")
        return False

def main():
    """主函数"""
    print("开始Git操作...")
    
    # 1. 检查当前Git状态
    print("=== 检查Git状态 ===")
    run_command("git status --porcelain")
    
    # 2. 添加所有文件
    print("=== 添加文件到Git ===")
    run_command("git add .")
    
    # 3. 检查添加后的状态
    print("=== 检查添加后状态 ===")
    run_command("git status --short")
    
    # 4. 创建提交
    print("=== 创建提交 ===")
    commit_message = """整理实验输出文件：按数据集、实验类型、Top-K配置分层组织

- 创建完整的文件夹结构分类体系
- Wikipedia100k输出：包含文档热度、N-gram、HNSW索引分析
- Wikipedia3.2k输出：包含文档热度、文档对组合分析  
- 按Top-K配置进一步细分实验结果
- 添加友好的中文文件名和README说明
- 整理脚本：organize_experiment_outputs.py"""
    
    run_command(f'git commit -m "{commit_message}"')
    
    # 5. 检查远程仓库
    print("=== 检查远程仓库 ===")
    run_command("git remote -v")
    
    # 6. 推送到GitHub
    print("=== 推送到GitHub ===")
    run_command("git push origin main")
    
    # 7. 检查最终状态
    print("=== 检查最终状态 ===")
    run_command("git log --oneline -3")

if __name__ == "__main__":
    main()