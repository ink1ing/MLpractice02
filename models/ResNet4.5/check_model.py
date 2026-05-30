# 模型检查脚本：用于检查训练好的模型文件，分析模型结构、训练指标和性能参数
import torch
import os
import time

def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}小时{int(minutes)}分{int(seconds)}秒"

def main():
    model_path = 'output/best_model.pth'
    print(f"检查模型文件: {model_path}")
    
    try:
        # 加载模型（设置weights_only=False）
        # 某些情况下，即使weights_only=False也可能无法加载完整的模型，这可能是纯权重文件
        try:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
            is_full_checkpoint = True
        except Exception as e:
            print(f"尝试加载完整检查点失败: {str(e)}")
            print("尝试作为权重文件加载...")
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
            is_full_checkpoint = False
        
        print("模型加载成功！")
        print("-" * 40)
        
        # 检查是否为完整检查点还是仅权重
        if is_full_checkpoint and isinstance(checkpoint, dict):
            print("模型检查点信息:")
            
            # 打印所有键
            print(f"检查点包含的键: {list(checkpoint.keys())}")
            
            # 基本信息
            if 'epoch' in checkpoint:
                print(f"训练轮次: {checkpoint['epoch']}")
            
            if 'training_time' in checkpoint:
                training_time = checkpoint['training_time']
                print(f"训练用时: {format_time(training_time)}")
            
            # 检查metrics键
            if 'metrics' in checkpoint:
                metrics = checkpoint['metrics']
                print("\nmetrics键内容:")
                if isinstance(metrics, dict):
                    for key, value in metrics.items():
                        if isinstance(value, list) and len(value) > 0:
                            print(f"  - {key}: 最后一个值 = {value[-1]}")
                        else:
                            print(f"  - {key}: {value}")
                else:
                    print(f"  metrics类型: {type(metrics)}")
            
            # 性能指标
            if 'best_val_loss' in checkpoint:
                print(f"最佳验证损失: {checkpoint['best_val_loss']:.4f}")
            
            if 'best_f1' in checkpoint:
                print(f"最佳F1分数: {checkpoint['best_f1']:.4f}")
            
            # 检查metrics_history
            if 'metrics_history' in checkpoint:
                metrics = checkpoint['metrics_history']
                
                if 'train' in metrics and metrics['train']:
                    train_epochs = len(metrics['train'])
                    print(f"训练总轮数: {train_epochs}")
                
                if 'val' in metrics and metrics['val']:
                    last_metrics = metrics['val'][-1]
                    print("\n最后一轮验证指标:")
                    if 'loss' in last_metrics:
                        print(f"  - 损失: {last_metrics['loss']:.4f}")
                    if 'position_accuracy' in last_metrics:
                        print(f"  - 位置准确率: {last_metrics['position_accuracy']:.4f}")
                    if 'position_f1' in last_metrics:
                        print(f"  - 位置F1: {last_metrics['position_f1']:.4f}")
                    if 'grade_mae' in last_metrics:
                        print(f"  - 等级MAE: {last_metrics['grade_mae']:.4f}")
                
            # 检查scheduler_state_dict
            if 'scheduler_state_dict' in checkpoint:
                print("\n学习率调度器信息:")
                scheduler = checkpoint['scheduler_state_dict']
                if 'last_epoch' in scheduler:
                    print(f"  - 最后训练轮次: {scheduler['last_epoch']}")
                if '_last_lr' in scheduler:
                    print(f"  - 最终学习率: {scheduler['_last_lr']}")
                
        else:
            print("模型文件仅包含权重，没有训练指标信息")
            
        # 输出模型大小
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"\n模型文件大小: {size_mb:.1f} MB")
    
    except Exception as e:
        print(f"检查模型文件时出错: {str(e)}")

if __name__ == "__main__":
    main() 