#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查FCN训练状态
显示训练日志和模型文件
"""
import os
import glob
import time
import sys
from datetime import datetime

def check_training_status(output_dir='./output_fcn'):
    """检查训练状态并显示最新日志"""
    print(f"\n===== FCN训练状态检查 =====")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查输出目录是否存在
    if not os.path.exists(output_dir):
        print(f"错误: 输出目录 {output_dir} 不存在")
        return
    
    # 检查模型文件
    model_files = glob.glob(os.path.join(output_dir, '*.pth'))
    if model_files:
        print(f"\n发现 {len(model_files)} 个模型文件:")
        for model_file in sorted(model_files):
            file_size = os.path.getsize(model_file) / (1024 * 1024)  # MB
            mod_time = datetime.fromtimestamp(os.path.getmtime(model_file))
            print(f"  - {os.path.basename(model_file)}: {file_size:.2f} MB, 修改时间: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("\n未发现模型文件，训练可能尚未完成第一个轮次")
    
    # 检查训练日志
    log_files = glob.glob(os.path.join(output_dir, '*.log')) + glob.glob(os.path.join(output_dir, '*.txt'))
    if log_files:
        print(f"\n发现 {len(log_files)} 个日志文件:")
        for log_file in sorted(log_files):
            file_size = os.path.getsize(log_file) / 1024  # KB
            mod_time = datetime.fromtimestamp(os.path.getmtime(log_file))
            print(f"  - {os.path.basename(log_file)}: {file_size:.2f} KB, 修改时间: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 显示最新的日志内容
            if file_size > 0:
                print("\n最后10行日志内容:")
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        for line in lines[-10:]:
                            print(f"    {line.strip()}")
                except Exception as e:
                    print(f"    无法读取日志文件: {e}")
    else:
        print("\n未发现日志文件")
    
    # 检查其他输出文件
    image_files = glob.glob(os.path.join(output_dir, '*.png')) + glob.glob(os.path.join(output_dir, '*.jpg'))
    if image_files:
        print(f"\n发现 {len(image_files)} 个图像文件:")
        for image_file in sorted(image_files)[-5:]:  # 只显示最新的5个
            file_size = os.path.getsize(image_file) / 1024  # KB
            mod_time = datetime.fromtimestamp(os.path.getmtime(image_file))
            print(f"  - {os.path.basename(image_file)}: {file_size:.2f} KB, 修改时间: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查进程状态
    import psutil
    fcn_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] == 'python.exe' and any('run_fcn.py' in cmd for cmd in proc.info['cmdline'] if cmd):
                fcn_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    if fcn_processes:
        print(f"\n发现 {len(fcn_processes)} 个FCN训练进程正在运行:")
        for proc in fcn_processes:
            cpu_percent = proc.cpu_percent(interval=0.1)
            memory_info = proc.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            running_time = time.time() - proc.create_time()
            hours, remainder = divmod(running_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"  - PID: {proc.pid}, CPU: {cpu_percent:.1f}%, 内存: {memory_mb:.1f} MB, 运行时间: {int(hours)}小时{int(minutes)}分钟{int(seconds)}秒")
    else:
        print("\n未发现正在运行的FCN训练进程")
    
    print("\n===== 检查完成 =====")

if __name__ == "__main__":
    output_dir = './output_fcn'
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    
    check_training_status(output_dir) 