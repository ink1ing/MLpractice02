#!/usr/bin/env python
# 数据集检查脚本：用于验证数据集结构和路径是否正确

import os
import argparse
import glob

def check_directory(path, name):
    """检查目录是否存在并输出其内容"""
    print(f"\n===== 检查{name}：{path} =====")
    
    if not os.path.exists(path):
        print(f"错误：{name}不存在！")
        return False
        
    if not os.path.isdir(path):
        print(f"错误：{path}不是一个目录！")
        return False
        
    items = os.listdir(path)
    print(f"目录包含 {len(items)} 个项目")
    
    # 列出前10个项目
    for i, item in enumerate(items[:10]):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            subitem_count = len(os.listdir(item_path))
            print(f"  {item}/ (目录，包含 {subitem_count} 个项目)")
        else:
            print(f"  {item} (文件，大小 {os.path.getsize(item_path)/1024:.1f} KB)")
    
    if len(items) > 10:
        print(f"  ... 以及 {len(items)-10} 个其他项目")
    
    return True

def check_tif_json_pairs(data_root, json_root):
    """检查TIF和JSON文件的配对情况"""
    print("\n===== 检查TIF和JSON文件配对 =====")
    
    # 获取叶片目录列表
    leaf_dirs = []
    for pattern in ['9*', '14*', '19*']:
        matching_dirs = glob.glob(os.path.join(data_root, pattern))
        leaf_dirs.extend(matching_dirs)
    
    if not leaf_dirs:
        print("警告：未找到任何叶片目录！")
        return
    
    print(f"找到了 {len(leaf_dirs)} 个叶片目录")
    
    total_tif = 0
    total_json = 0
    total_pairs = 0
    
    for leaf_dir in leaf_dirs:
        dir_name = os.path.basename(leaf_dir)
        print(f"\n检查目录：{dir_name}")
        
        # 对应的JSON目录
        json_subdir = dir_name + '_json'
        json_dir = os.path.join(json_root, json_subdir)
        
        if not os.path.exists(json_dir):
            print(f"  错误：对应的JSON目录不存在：{json_dir}")
            continue
        
        # 统计TIF文件
        tif_files = [f for f in os.listdir(leaf_dir) if f.endswith('.tif')]
        tif_count = len(tif_files)
        total_tif += tif_count
        
        # 统计JSON文件
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        json_count = len(json_files)
        total_json += json_count
        
        # 检查配对
        pair_count = 0
        for tif_file in tif_files[:5]:  # 只检查前5个用于示例
            json_file = tif_file.replace('.tif', '.json')
            json_path = os.path.join(json_dir, json_file)
            
            if os.path.exists(json_path):
                pair_count += 1
                print(f"  找到配对：{tif_file} -> {json_file}")
            else:
                print(f"  未找到配对：{tif_file} -> {json_file}")
        
        # 统计所有配对
        all_pairs = 0
        for tif_file in tif_files:
            json_file = tif_file.replace('.tif', '.json')
            json_path = os.path.join(json_dir, json_file)
            
            if os.path.exists(json_path):
                all_pairs += 1
        
        total_pairs += all_pairs
        print(f"  目录 {dir_name} 统计：{tif_count} TIF文件，{json_count} JSON文件，{all_pairs} 个有效配对")
    
    print(f"\n总计：{total_tif} TIF文件，{total_json} JSON文件，{total_pairs} 个有效配对")
    
    if total_pairs == 0:
        print("\n警告：未找到任何有效的TIF-JSON配对！请检查文件命名格式是否匹配")

def main():
    parser = argparse.ArgumentParser(description='玉米南方锈病数据集检查工具')
    
    parser.add_argument('--data_root', type=str, default='./guanceng-bit',
                        help='数据根目录路径')
    parser.add_argument('--json_root', type=str, default='./biaozhu_json',
                        help='JSON标注根目录路径')
    
    args = parser.parse_args()
    
    print("====== 玉米南方锈病数据集检查 ======")
    print(f"数据根目录：{os.path.abspath(args.data_root)}")
    print(f"JSON标注目录：{os.path.abspath(args.json_root)}")
    
    # 检查两个主目录
    data_ok = check_directory(args.data_root, "数据根目录")
    json_ok = check_directory(args.json_root, "JSON标注目录")
    
    if data_ok and json_ok:
        check_tif_json_pairs(args.data_root, args.json_root)
    
    print("\n====== 数据集检查完成 ======")

if __name__ == "__main__":
    main() 