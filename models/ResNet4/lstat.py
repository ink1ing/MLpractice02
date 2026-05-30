#!/usr/bin/env python
# 文件系统检查脚本：检查数据目录和文件的权限，确保所有文件都是可读的

import os
import sys
import stat
import argparse
import random

def check_permissions(path, sample_size=10):
    """
    检查目录和文件的权限
    
    参数:
        path: 要检查的目录路径
        sample_size: 每个目录随机抽样检查的文件数
    """
    if not os.path.exists(path):
        print(f"错误: 路径不存在: {path}")
        return False
    
    if os.path.isfile(path):
        # 检查单个文件
        try:
            mode = os.stat(path).st_mode
            is_readable = bool(mode & stat.S_IRUSR)
            print(f"文件: {path}")
            print(f"  - 可读: {'是' if is_readable else '否'}")
            print(f"  - 权限: {stat.filemode(mode)}")
            
            # 尝试读取文件
            try:
                with open(path, 'rb') as f:
                    _ = f.read(1)
                print(f"  - 读取测试: 成功")
            except Exception as e:
                print(f"  - 读取测试: 失败 - {str(e)}")
                
            return is_readable
        except Exception as e:
            print(f"检查文件权限时出错: {path} - {str(e)}")
            return False
    
    # 如果是目录，递归检查
    print(f"\n目录: {path}")
    try:
        mode = os.stat(path).st_mode
        is_readable = bool(mode & stat.S_IRUSR)
        is_executable = bool(mode & stat.S_IXUSR)  # 目录需要可执行权限才能列出内容
        
        print(f"  - 可读: {'是' if is_readable else '否'}")
        print(f"  - 可执行: {'是' if is_executable else '否'}")
        print(f"  - 权限: {stat.filemode(mode)}")
        
        if not (is_readable and is_executable):
            print(f"  - 警告: 目录缺少必要权限，可能无法访问其中的文件")
            return False
    except Exception as e:
        print(f"检查目录权限时出错: {path} - {str(e)}")
        return False
    
    # 获取目录下的所有项目
    try:
        items = os.listdir(path)
        print(f"  - 包含 {len(items)} 个项目")
        
        # 检查子目录
        subdirs = [os.path.join(path, item) for item in items if os.path.isdir(os.path.join(path, item))]
        if subdirs:
            print(f"  - 包含 {len(subdirs)} 个子目录")
            for subdir in subdirs:
                check_permissions(subdir, sample_size)
        
        # 随机抽样检查文件
        files = [os.path.join(path, item) for item in items if os.path.isfile(os.path.join(path, item))]
        if files:
            print(f"  - 包含 {len(files)} 个文件")
            # 随机抽样
            if len(files) > sample_size:
                sampled_files = random.sample(files, sample_size)
                print(f"  - 随机抽样 {sample_size} 个文件进行检查:")
            else:
                sampled_files = files
                print(f"  - 检查所有 {len(files)} 个文件:")
            
            # 检查抽样文件
            for file in sampled_files:
                file_name = os.path.basename(file)
                try:
                    file_mode = os.stat(file).st_mode
                    file_readable = bool(file_mode & stat.S_IRUSR)
                    print(f"    - {file_name}: {'可读' if file_readable else '不可读'} ({stat.filemode(file_mode)})")
                    
                    # 尝试读取文件
                    if file_readable:
                        try:
                            with open(file, 'rb') as f:
                                _ = f.read(1)
                        except Exception as e:
                            print(f"      警告: 文件标记为可读，但读取失败: {str(e)}")
                except Exception as e:
                    print(f"    - {file_name}: 检查出错 - {str(e)}")
        
        return True
    except Exception as e:
        print(f"列出目录内容时出错: {path} - {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="检查数据目录和文件的权限")
    parser.add_argument("path", help="要检查的目录或文件路径")
    parser.add_argument("--samples", type=int, default=5, help="每个目录随机抽样检查的文件数量")
    
    args = parser.parse_args()
    
    print("=== 文件权限检查 ===")
    print(f"检查路径: {os.path.abspath(args.path)}")
    print(f"抽样数量: {args.samples}")
    
    result = check_permissions(args.path, args.samples)
    
    print("\n=== 检查结果 ===")
    if result:
        print("权限检查通过，未发现明显问题")
    else:
        print("权限检查发现问题，请查看上方日志")
    
if __name__ == "__main__":
    main() 