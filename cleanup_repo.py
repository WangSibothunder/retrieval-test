#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理Git仓库脚本 - 移除不需要的PNG和TXT文件
只保留"生成输出"文件夹和核心代码文件
"""

import subprocess
import os
import glob

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

def get_files_to_remove():
    """获取需要从Git中移除的文件列表"""
    files_to_remove = []
    
    # 获取根目录中的所有PNG文件
    png_files = glob.glob("*.png")
    files_to_remove.extend(png_files)
    
    # 获取根目录中的所有TXT文件（但排除README等重要文件）
    txt_files = glob.glob("*.txt")
    # 排除一些重要的配置文件
    important_files = ['requirements.txt', 'commands.txt']
    txt_files = [f for f in txt_files if f not in important_files]
    files_to_remove.extend(txt_files)
    
    return files_to_remove

def create_gitignore():
    """创建.gitignore文件"""
    gitignore_content = """# 忽略根目录中的PNG和TXT统计文件
/*.png
/*.txt
!requirements.txt
!commands.txt

# 忽略临时文件
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
.pytest_cache/

# 忽略IDE文件
.vscode/
.idea/
*.swp
*.swo

# 忽略系统文件
.DS_Store
Thumbs.db
"""
    
    with open('.gitignore', 'w', encoding='utf-8') as f:
        f.write(gitignore_content)
    print("创建了.gitignore文件")

def main():
    """主函数"""
    print("开始清理Git仓库...")
    
    # 1. 检查当前状态
    print("=== 当前Git状态 ===")
    run_command("git status --short")
    
    # 2. 获取需要移除的文件
    files_to_remove = get_files_to_remove()
    
    if files_to_remove:
        print(f"=== 发现需要移除的文件 ({len(files_to_remove)}个) ===")
        for file in files_to_remove:
            print(f"  - {file}")
        
        # 3. 从Git中移除这些文件
        print("=== 从Git仓库中移除文件 ===")
        for file in files_to_remove:
            if os.path.exists(file):
                run_command(f"git rm --cached {file}")
    else:
        print("没有发现需要移除的PNG/TXT文件")
    
    # 4. 创建.gitignore文件
    print("=== 创建.gitignore文件 ===")
    create_gitignore()
    
    # 5. 添加.gitignore到Git
    print("=== 添加.gitignore到Git ===")
    run_command("git add .gitignore")
    
    # 6. 确保"生成输出"文件夹被跟踪
    print("=== 确保生成输出文件夹被跟踪 ===")
    run_command('git add "生成输出"')
    
    # 7. 添加核心Python文件
    python_files = [
        "*.py", "README.md", "README_NEW.md", "WIKIPEAD_VISUALIZATION_GUIDE.md",
        "requirements.txt", "commands.txt", "config.py", "data", "output"
    ]
    
    print("=== 添加核心文件 ===")
    for pattern in python_files:
        run_command(f"git add {pattern}")
    
    # 8. 检查暂存状态
    print("=== 检查暂存状态 ===")
    run_command("git status --short")
    
    # 9. 创建提交
    print("=== 创建提交 ===")
    commit_message = """清理仓库：移除根目录PNG/TXT文件，只保留生成输出文件夹

- 从Git仓库中移除根目录中的PNG和TXT统计文件
- 添加.gitignore防止这些文件再次被提交
- 保留核心Python代码文件和requirements.txt等配置文件
- 确保"生成输出"文件夹完整保留
- 仓库现在只包含必要的代码和整理好的实验结果"""
    
    run_command(f'git commit -m "{commit_message}"')
    
    # 10. 推送到GitHub
    print("=== 推送到GitHub ===")
    run_command("git push origin main")
    
    # 11. 检查最终状态
    print("=== 最终Git状态 ===")
    run_command("git status")
    run_command("git log --oneline -2")

if __name__ == "__main__":
    main()