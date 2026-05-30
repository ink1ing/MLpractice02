# [核心文件] UNet+Attention模型训练启动文件：负责配置和启动模型训练过程，设置训练参数如批量大小、学习率等
import os
import subprocess
import argparse
import torch

def main(args):
    """
    运行UNet+Attention模型训练，设置优化的参数
    """
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 构建命令行参数
    cmd = [
        "python", "train_unet.py",
        "--data_root", args.data_root,
        "--json_root", args.json_root,
        "--img_size", str(args.img_size),
        "--in_channels", str(args.in_channels),
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--optimizer", args.optimizer,
        "--lr_scheduler", args.lr_scheduler,
        "--output_dir", args.output_dir,
        "--num_workers", str(args.num_workers),
        "--seg_weight", str(args.seg_weight),
        "--position_weight", str(args.position_weight),
        "--grade_weight", str(args.grade_weight),
    ]
    
    # 添加可选标志
    if args.amp:
        cmd.append("--amp")
    
    if args.no_cuda:
        cmd.append("--no_cuda")
    
    # 打印完整命令
    print("运行命令:")
    print(" ".join(cmd))
    
    # 执行命令
    subprocess.run(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="玉米南方锈病UNet+Attention模型训练启动脚本")
    
    # 数据参数
    parser.add_argument('--data_root', type=str, default='./guanceng-bit',
                        help='数据根目录路径')
    parser.add_argument('--json_root', type=str, default='./biaozhu_json',
                        help='JSON标注根目录路径')
    parser.add_argument('--img_size', type=int, default=128,
                        help='图像大小')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='输入图像通道数')
    
    # 模型训练参数
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='学习率')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'],
                        help='优化器类型')
    parser.add_argument('--lr_scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine', 'step', 'none'],
                        help='学习率调度器类型')
    
    # 任务权重参数
    parser.add_argument('--seg_weight', type=float, default=0.4,
                        help='分割任务的损失权重')
    parser.add_argument('--position_weight', type=float, default=0.3,
                        help='位置分类任务的损失权重')
    parser.add_argument('--grade_weight', type=float, default=0.3,
                        help='等级回归任务的损失权重')
    
    # 系统参数
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--no_cuda', action='store_true',
                        help='不使用CUDA')
    parser.add_argument('--amp', action='store_true',
                        help='是否启用混合精度训练')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./unet_output',
                        help='输出目录路径')
    
    args = parser.parse_args()
    
    # 检查CUDA可用性
    if not args.no_cuda and torch.cuda.is_available():
        print(f"已检测到CUDA设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        if args.batch_size > 32 and torch.cuda.get_device_properties(0).total_memory < 16 * (1024**3):
            print(f"警告: 批次大小 {args.batch_size} 可能对于当前GPU过大，考虑减小批次大小")
    
    main(args) 