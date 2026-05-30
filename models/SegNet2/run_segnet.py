#!/usr/bin/env python
# 核心文件，完整运行项目不可缺少
# 运行SegNet模型训练的脚本
# 使用示例: python run_segnet.py --data_root ./guanceng-bit --json_root ./biaozhu_json --batch_size 16 --model_type segnet+

import os
import subprocess
import argparse
import torch
import platform
import shutil
from datetime import datetime
import sys

def setup_env():
    """设置虚拟环境和必要的依赖"""
    # 检查是否存在虚拟环境
    if not os.path.exists("venv"):
        print("正在创建虚拟环境...")
        subprocess.run(["python", "-m", "venv", "venv"])
    
    # 根据操作系统激活虚拟环境并安装依赖
    if platform.system() == "Windows":
        activate_cmd = "venv\\Scripts\\activate"
    else:
        activate_cmd = "source venv/bin/activate"
    
    # 检查GPU是否可用
    gpu_available = torch.cuda.is_available()
    gpu_info = f"GPU: {torch.cuda.get_device_name(0)}" if gpu_available else "GPU: 不可用，将使用CPU"
    print(gpu_info)
    
    # 安装依赖
    req_install_cmd = f"{activate_cmd} && pip install torch torchvision rasterio scikit-image matplotlib tqdm seaborn numpy pillow scipy"
    print("正在安装依赖...")
    subprocess.run(req_install_cmd, shell=True)
    
    return gpu_available

def backup_code(output_dir):
    """备份代码到输出目录，组织更加有序"""
    backup_dir = os.path.join(output_dir, "code_backup")
    os.makedirs(backup_dir, exist_ok=True)
    
    # 按模块组织备份
    model_dir = os.path.join(backup_dir, "model")
    data_dir = os.path.join(backup_dir, "data")
    utils_dir = os.path.join(backup_dir, "utils")
    scripts_dir = os.path.join(backup_dir, "scripts")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(utils_dir, exist_ok=True)
    os.makedirs(scripts_dir, exist_ok=True)
    
    # 模型相关文件
    if os.path.exists("model.py"):
        shutil.copy("model.py", os.path.join(model_dir, "model.py"))
    
    # 数据相关文件
    if os.path.exists("dataset.py"):
        shutil.copy("dataset.py", os.path.join(data_dir, "dataset.py"))
    
    # 工具相关文件
    if os.path.exists("utils.py"):
        shutil.copy("utils.py", os.path.join(utils_dir, "utils.py"))
    
    # 训练和脚本文件
    if os.path.exists("train.py"):
        shutil.copy("train.py", os.path.join(scripts_dir, "train.py"))
    if os.path.exists("run_segnet.py"):
        shutil.copy("run_segnet.py", os.path.join(scripts_dir, "run_segnet.py"))
    if os.path.exists("test_segnet.py"):
        shutil.copy("test_segnet.py", os.path.join(scripts_dir, "test_segnet.py"))
    
    # 保存当前命令和环境信息
    with open(os.path.join(backup_dir, "run_info.txt"), "w") as f:
        f.write(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"命令行: {' '.join(sys.argv)}\n")
        f.write(f"Python版本: {sys.version}\n")
        f.write(f"PyTorch版本: {torch.__version__}\n")
        f.write(f"CUDA是否可用: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"CUDA版本: {torch.version.cuda}\n")
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
    
    print(f"代码已备份到 {backup_dir}")
    return backup_dir

def run_training(args):
    """运行训练过程"""
    # 创建命令行
    cmd = [
        "python", "train.py",
        "--data_root", args.data_root,
        "--json_root", args.json_root,
        "--model_type", args.model_type,
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--output_dir", args.output_dir,
        "--num_workers", str(args.num_workers),
        "--task_weights", args.task_weights,
        "--optimizer", args.optimizer,
        "--lr_scheduler", args.lr_scheduler
    ]
    
    # 添加可选参数
    if args.amp:
        cmd.append("--amp")
    
    if args.loss_type:
        cmd.extend(["--loss_type", args.loss_type])
    
    if args.load_weights:
        cmd.extend(["--load_weights", args.load_weights])
    
    if args.resume:
        cmd.extend(["--resume", args.resume])
    
    if args.use_hybrid_loss:
        cmd.append("--use_hybrid_loss")
    
    if args.adaptive_weights:
        cmd.append("--adaptive_weights")
    
    # 执行训练命令
    print(f"开始训练：{' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    # 检查是否成功
    if result.returncode != 0:
        print("训练过程失败，请检查错误信息")
        return False
    return True

def run_testing(args):
    """运行测试过程"""
    if not args.run_test:
        return True
    
    # 查找最佳模型路径
    model_path = os.path.join(args.output_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(args.output_dir, 'last_model.pth')
    
    if not os.path.exists(model_path):
        print("无法找到训练好的模型文件，跳过测试")
        return False
    
    # 创建测试命令行
    test_cmd = [
        "python", "test_segnet.py",
        "--data_root", args.data_root,
        "--json_root", args.json_root,
        "--model_path", model_path,
        "--model_type", args.model_type,
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--output_dir", os.path.join(args.output_dir, "test_results")
    ]
    
    if args.use_hybrid_loss:
        test_cmd.append("--use_hybrid_loss")
    
    # 执行测试命令
    print(f"开始测试：{' '.join(test_cmd)}")
    result = subprocess.run(test_cmd)
    
    if result.returncode != 0:
        print("测试过程失败，请检查错误信息")
        return False
    
    print(f"测试结果已保存到 {os.path.join(args.output_dir, 'test_results')}")
    return True

def main():
    parser = argparse.ArgumentParser(description='玉米南方锈病SegNet模型训练脚本')
    
    # 必要参数
    parser.add_argument('--data_root', type=str, required=True,
                        help='数据根目录路径')
    parser.add_argument('--json_root', type=str, required=True, 
                        help='JSON标注根目录路径')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='segnet+',
                        choices=['segnet', 'segnet+'],
                        help='SegNet模型类型，默认使用带注意力的增强版')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小，根据GPU内存调整')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='学习率')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--task_weights', type=str, default='0.5,0.3,0.2',
                        help='任务权重 [位置,等级,分割]')
    parser.add_argument('--loss_type', type=str, default='focal',
                        choices=['ce', 'focal'],
                        help='损失函数类型')
    
    # 优化器和学习率策略
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw', 'sgd'],
                        help='优化器类型')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=['plateau', 'cosine', 'step', 'none'],
                        help='学习率调度器类型')
    
    # 增强训练策略
    parser.add_argument('--use_hybrid_loss', action='store_true',
                        help='使用混合损失函数')
    parser.add_argument('--adaptive_weights', action='store_true',
                        help='使用自适应任务权重')
    
    # 其他参数
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录，默认为./output_segnet_YYYY-MM-DD_HH-MM')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='启用混合精度训练')
    parser.add_argument('--load_weights', type=str, default=None,
                        help='从权重文件加载模型')
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练')
    parser.add_argument('--run_test', action='store_true',
                        help='训练完成后自动运行测试')
    parser.add_argument('--skip_backup', action='store_true',
                        help='跳过代码备份')
    
    args = parser.parse_args()
    
    # 设置输出目录
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        args.output_dir = f"./output_segnet_{timestamp}"
    
    # 检查并创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置环境
    gpu_available = setup_env()
    if not gpu_available:
        print("警告：未检测到GPU，训练速度可能会很慢")
        if args.batch_size > 8:
            print(f"自动减小批次大小从 {args.batch_size} 到 8")
            args.batch_size = 8
    
    # 备份代码
    if not args.skip_backup:
        backup_dir = backup_code(args.output_dir)
    
    # 运行训练
    training_success = run_training(args)
    
    # 如果训练成功且需要测试，运行测试
    if training_success and args.run_test:
        run_testing(args)
    
    print(f"全部流程完成！结果保存在 {args.output_dir}")
    print(f"最佳模型路径: {os.path.join(args.output_dir, 'best_model.pth')}")
    print(f"最后模型路径: {os.path.join(args.output_dir, 'last_model.pth')}")

if __name__ == "__main__":
    main() 