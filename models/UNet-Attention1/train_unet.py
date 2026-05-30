# [核心文件] UNet+Attention模型训练文件：包含训练和评估功能，使用Dice Loss进行图像分割和组合损失函数进行多任务学习
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from torch.amp import autocast, GradScaler
import time

# 导入自定义模块
from dataset import CornRustDataset, get_dataloaders
from unet_model import get_unet_model
from utils import save_checkpoint, load_checkpoint, calculate_metrics, plot_metrics, FocalLoss, calculate_class_weights

# 定义数据增强变换
def get_data_transforms(train=True):
    """获取数据增强变换"""
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
        ])
    else:
        return None

# 定义Dice Loss，用于分割任务
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        # 扁平化预测和目标
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # 计算交集
        intersection = (pred_flat * target_flat).sum()
        
        # 计算Dice系数
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        # 返回Dice损失
        return 1 - dice

# 定义组合损失函数
class CombinedLoss(nn.Module):
    def __init__(self, seg_weight=0.4, position_weight=0.3, grade_weight=0.3, focal_gamma=2.0):
        super(CombinedLoss, self).__init__()
        self.seg_weight = seg_weight
        self.position_weight = position_weight
        self.grade_weight = grade_weight
        
        self.seg_criterion = DiceLoss()
        self.position_criterion = FocalLoss(gamma=focal_gamma)
        self.grade_criterion = nn.MSELoss()
        
    def forward(self, preds, targets):
        # 解包预测和目标
        seg_pred, position_logits, grade_values = preds
        seg_target, position_labels, grade_labels = targets
        
        # 计算各个任务的损失
        seg_loss = self.seg_criterion(seg_pred, seg_target)
        position_loss = self.position_criterion(position_logits, position_labels)
        grade_loss = self.grade_criterion(grade_values, grade_labels)
        
        # 组合损失
        total_loss = (self.seg_weight * seg_loss + 
                      self.position_weight * position_loss + 
                      self.grade_weight * grade_loss)
                      
        return total_loss, seg_loss, position_loss, grade_loss

def train_one_epoch(model, train_loader, optimizer, criterion, device, scaler=None):
    """训练模型一个epoch"""
    model.train()
    total_loss = 0.0
    seg_loss_sum = 0.0
    position_loss_sum = 0.0
    grade_loss_sum = 0.0
    position_correct = 0
    total_samples = 0
    grade_mae_sum = 0.0
    
    progress_bar = tqdm(train_loader, desc="训练中")
    
    for images, position_labels, grade_labels in progress_bar:
        # 将数据移动到指定设备
        images = images.to(device)
        position_labels = position_labels.to(device).view(-1).long()
        grade_labels = grade_labels.float().unsqueeze(1).to(device)
        
        # 创建分割标签（这里假设分割标签是基于位置标签生成的临时掩码）
        # 在实际应用中，应该从数据集中读取真实的分割标签
        batch_size, channels, height, width = images.shape
        seg_labels = torch.zeros((batch_size, 1, height, width), device=device)
        
        # 为示例生成简单的分割掩码 - 在实际应用中替换为真实掩码
        for i, grade in enumerate(grade_labels):
            # 如果等级 > 0，则创建简单的圆形掩码作为示例
            if grade.item() > 0:
                center_y, center_x = height // 2, width // 2
                radius = int(min(height, width) * 0.3)  # 使用图像尺寸的30%作为半径
                
                y, x = torch.meshgrid(torch.arange(height, device=device), 
                                      torch.arange(width, device=device), indexing='ij')
                mask = ((y - center_y)**2 + (x - center_x)**2 <= radius**2).float()
                seg_labels[i, 0] = mask
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 使用混合精度训练
        if scaler is not None:
            with autocast(device_type='cuda'):
                # 前向传播
                seg_pred, position_logits, grade_values = model(images)
                
                # 计算损失
                loss, seg_loss, position_loss, grade_loss = criterion(
                    (seg_pred, position_logits, grade_values),
                    (seg_labels, position_labels, grade_labels)
                )
            
            # 使用scaler进行反向传播和参数更新
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 常规训练流程
            seg_pred, position_logits, grade_values = model(images)
            
            loss, seg_loss, position_loss, grade_loss = criterion(
                (seg_pred, position_logits, grade_values),
                (seg_labels, position_labels, grade_labels)
            )
            
            loss.backward()
            optimizer.step()
        
        # 统计指标
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        seg_loss_sum += seg_loss.item() * batch_size
        position_loss_sum += position_loss.item() * batch_size
        grade_loss_sum += grade_loss.item() * batch_size
        
        # 计算位置分类准确率
        _, position_preds = torch.max(position_logits, 1)
        position_correct += (position_preds == position_labels).sum().item()
        
        # 计算等级预测MAE
        grade_mae = torch.abs(grade_values - grade_labels).mean().item()
        grade_mae_sum += grade_mae * batch_size
        
        total_samples += batch_size
        
        # 更新进度条显示当前性能指标
        progress_bar.set_postfix({
            'loss': loss.item(),
            'pos_acc': position_correct / total_samples,
            'grade_mae': grade_mae_sum / total_samples
        })
    
    # 计算整个epoch的平均指标
    avg_loss = total_loss / total_samples
    avg_seg_loss = seg_loss_sum / total_samples
    avg_position_loss = position_loss_sum / total_samples
    avg_grade_loss = grade_loss_sum / total_samples
    position_accuracy = position_correct / total_samples
    grade_mae = grade_mae_sum / total_samples
    
    # 返回训练指标
    return {
        'loss': avg_loss,
        'seg_loss': avg_seg_loss,
        'position_loss': avg_position_loss,
        'grade_loss': avg_grade_loss,
        'position_accuracy': position_accuracy,
        'grade_mae': grade_mae
    }

def evaluate(model, val_loader, criterion, device):
    """评估模型在验证集上的性能"""
    model.eval()
    total_loss = 0.0
    seg_loss_sum = 0.0
    position_loss_sum = 0.0
    grade_loss_sum = 0.0
    
    # 收集所有预测和真实标签
    position_preds_all = []
    position_labels_all = []
    grade_values_all = []
    grade_labels_all = []
    
    with torch.no_grad():
        for images, position_labels, grade_labels in val_loader:
            # 将数据移动到指定设备
            images = images.to(device)
            position_labels = position_labels.to(device).view(-1).long()
            grade_labels = grade_labels.float().unsqueeze(1).to(device)
            
            # 创建分割标签（这里假设分割标签是基于位置标签生成的临时掩码）
            batch_size, channels, height, width = images.shape
            seg_labels = torch.zeros((batch_size, 1, height, width), device=device)
            
            # 为示例生成简单的分割掩码
            for i, grade in enumerate(grade_labels):
                if grade.item() > 0:
                    center_y, center_x = height // 2, width // 2
                    radius = int(min(height, width) * 0.3)
                    
                    y, x = torch.meshgrid(torch.arange(height, device=device), 
                                          torch.arange(width, device=device), indexing='ij')
                    mask = ((y - center_y)**2 + (x - center_x)**2 <= radius**2).float()
                    seg_labels[i, 0] = mask
            
            # 使用混合精度计算
            with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                # 前向传播
                seg_pred, position_logits, grade_values = model(images)
                
                # 计算损失
                loss, seg_loss, position_loss, grade_loss = criterion(
                    (seg_pred, position_logits, grade_values),
                    (seg_labels, position_labels, grade_labels)
                )
            
            # 统计指标
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            seg_loss_sum += seg_loss.item() * batch_size
            position_loss_sum += position_loss.item() * batch_size
            grade_loss_sum += grade_loss.item() * batch_size
            
            # 获取位置预测类别
            _, position_preds = torch.max(position_logits, 1)
            
            # 收集预测和标签
            position_preds_all.extend(position_preds.cpu().numpy())
            position_labels_all.extend(position_labels.cpu().numpy())
            grade_values_all.extend(grade_values.cpu().numpy())
            grade_labels_all.extend(grade_labels.cpu().numpy())
    
    # 计算平均指标
    total_samples = len(val_loader.dataset)
    avg_loss = total_loss / total_samples
    avg_seg_loss = seg_loss_sum / total_samples
    avg_position_loss = position_loss_sum / total_samples
    avg_grade_loss = grade_loss_sum / total_samples
    
    # 计算位置分类详细指标
    position_accuracy = accuracy_score(position_labels_all, position_preds_all)
    position_f1 = f1_score(position_labels_all, position_preds_all, average='macro')
    position_f1_per_class = f1_score(position_labels_all, position_preds_all, average=None)
    position_cm = confusion_matrix(position_labels_all, position_preds_all)
    position_precision = precision_score(position_labels_all, position_preds_all, average='macro')
    position_recall = recall_score(position_labels_all, position_preds_all, average='macro')
    
    # 计算等级回归指标
    grade_values_all = np.array(grade_values_all)
    grade_labels_all = np.array(grade_labels_all)
    grade_mae = np.mean(np.abs(grade_values_all - grade_labels_all))
    
    # 计算±2误差容忍率
    tolerance = 2.0
    grade_tolerance_accuracy = np.mean(np.abs(grade_values_all - grade_labels_all) <= tolerance)
    
    # 返回包含所有评估指标的字典
    return {
        'loss': avg_loss,
        'seg_loss': avg_seg_loss,
        'position_loss': avg_position_loss,
        'grade_loss': avg_grade_loss,
        'position_accuracy': position_accuracy,
        'position_f1': position_f1,
        'position_f1_per_class': position_f1_per_class,
        'position_precision': position_precision,
        'position_recall': position_recall,
        'position_cm': position_cm,
        'grade_mae': grade_mae,
        'grade_tolerance_accuracy': grade_tolerance_accuracy
    }

def main(args):
    """主训练函数，处理训练和验证流程"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"使用设备: {device}")
    
    # GPU优化配置
    if torch.cuda.is_available() and not args.no_cuda:
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        
        # 打印GPU信息
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        gpu_mem = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
        print(f"\nGPU信息: {gpu_name}, 总内存: {gpu_mem:.2f}GB")
    
    # 设置混合精度训练
    use_amp = args.amp and torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("启用混合精度训练 (AMP)")
    
    # 设置随机种子确保可重复性
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据集
    print(f"\n数据配置:")
    print(f"数据根目录: {args.data_root}")
    print(f"JSON标注目录: {args.json_root}")
    print(f"图像大小: {args.img_size}x{args.img_size}")
    print(f"训练集比例: {args.train_ratio}")
    
    # 获取训练和验证数据加载器
    train_loader, val_loader = get_dataloaders(
        data_root=args.data_root,
        json_root=args.json_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        train_ratio=args.train_ratio,
        aug_prob=args.aug_prob
    )
    
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    
    # 创建UNet+Attention模型
    model = get_unet_model(in_channels=args.in_channels, img_size=args.img_size)
    model = model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型信息:")
    print(f"模型类型: UNet+Attention")
    print(f"输入通道数: {args.in_channels}")
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    # 定义组合损失函数
    criterion = CombinedLoss(
        seg_weight=args.seg_weight,
        position_weight=args.position_weight,
        grade_weight=args.grade_weight,
        focal_gamma=args.focal_gamma
    )
    
    # 定义优化器
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
    # 定义学习率调度器
    scheduler = None
    if args.lr_scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=args.min_lr
        )
    elif args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr
        )
    elif args.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.5
        )
    
    # 跟踪最佳性能
    best_val_loss = float('inf')
    best_f1 = 0.0
    
    # 初始化指标历史记录
    metrics_history = {
        'train': [],
        'val': [],
        'train_loss': [],
        'val_loss': [],
        'train_seg_loss': [],
        'val_seg_loss': [],
        'train_position_accuracy': [],
        'val_position_accuracy': [],
        'val_position_f1': [],
        'train_grade_mae': [],
        'val_grade_mae': [],
        'val_grade_tolerance': []
    }
    
    # 开始训练
    print(f"\n开始训练，共 {args.epochs} 轮...")
    for epoch in range(args.epochs):
        print(f"\n轮次 {epoch+1}/{args.epochs}")
        
        # 训练一个轮次
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler=scaler)
        
        # 验证
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # 保存指标历史
        metrics_history['train'].append(train_metrics)
        metrics_history['val'].append(val_metrics)
        
        # 更新历史记录
        metrics_history['train_loss'].append(train_metrics['loss'])
        metrics_history['val_loss'].append(val_metrics['loss'])
        metrics_history['train_seg_loss'].append(train_metrics['seg_loss'])
        metrics_history['val_seg_loss'].append(val_metrics['seg_loss'])
        metrics_history['train_position_accuracy'].append(train_metrics['position_accuracy'])
        metrics_history['val_position_accuracy'].append(val_metrics['position_accuracy'])
        metrics_history['val_position_f1'].append(val_metrics['position_f1'])
        metrics_history['train_grade_mae'].append(train_metrics['grade_mae'])
        metrics_history['val_grade_mae'].append(val_metrics['grade_mae'])
        metrics_history['val_grade_tolerance'].append(val_metrics['grade_tolerance_accuracy'])
        
        # 打印当前性能
        print(f"训练: 损失={train_metrics['loss']:.4f}, 位置准确率={train_metrics['position_accuracy']:.4f}, 等级MAE={train_metrics['grade_mae']:.4f}")
        print(f"验证: 损失={val_metrics['loss']:.4f}, 位置准确率={val_metrics['position_accuracy']:.4f}, 位置F1={val_metrics['position_f1']:.4f}, 等级MAE={val_metrics['grade_mae']:.4f}")
        
        # 更新学习率调度器
        if scheduler is not None:
            if args.lr_scheduler == 'plateau':
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
        
        # 保存最佳模型
        current_val_loss = val_metrics['loss']
        current_f1 = val_metrics['position_f1']
        
        is_best = False
        
        # 根据验证损失或F1分数确定最佳模型
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            is_best = True
        elif current_f1 > best_f1:
            best_f1 = current_f1
            is_best = True
        
        # 保存检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'best_f1': best_f1,
            'metrics_history': metrics_history
        }
        
        # 保存最后一轮模型
        try:
            torch.save(checkpoint, os.path.join(args.output_dir, 'last_model.pth'))
        except Exception as e:
            print(f"保存最后一轮模型时出错: {e}")
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'last_model_weights.pth'))
        
        # 保存最佳模型
        if is_best:
            try:
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
                print("发现最佳模型，已保存!")
            except Exception as e:
                print(f"保存最佳模型时出错: {e}")
    
    # 绘制训练过程指标
    plt.figure(figsize=(16, 12))
    
    # 绘制损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(metrics_history['train_loss'], label='训练损失')
    plt.plot(metrics_history['val_loss'], label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练和验证损失')
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 绘制位置准确率和F1曲线
    plt.subplot(2, 2, 2)
    plt.plot(metrics_history['train_position_accuracy'], label='训练位置准确率')
    plt.plot(metrics_history['val_position_accuracy'], label='验证位置准确率')
    plt.plot(metrics_history['val_position_f1'], label='验证位置F1')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy / F1')
    plt.title('位置分类性能')
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 绘制等级MAE曲线
    plt.subplot(2, 2, 3)
    plt.plot(metrics_history['train_grade_mae'], label='训练等级MAE')
    plt.plot(metrics_history['val_grade_mae'], label='验证等级MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('等级预测平均绝对误差')
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 绘制分割损失曲线
    plt.subplot(2, 2, 4)
    plt.plot(metrics_history['train_seg_loss'], label='训练分割损失')
    plt.plot(metrics_history['val_seg_loss'], label='验证分割损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('分割任务损失')
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 调整子图布局并保存
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_metrics.png'))
    plt.close()
    
    # 打印最终结果
    print("\n训练完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"最佳位置F1分数: {best_f1:.4f}")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='玉米南方锈病UNet+Attention模型训练')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, default='./guanceng-bit',
                        help='数据根目录路径')
    parser.add_argument('--json_root', type=str, default='./biaozhu_json',
                        help='JSON标注根目录路径')
    parser.add_argument('--img_size', type=int, default=128,
                        help='图像大小')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='输入图像通道数')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例')
    parser.add_argument('--aug_prob', type=float, default=0.7,
                        help='数据增强应用概率')
    
    # 损失函数参数
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss的gamma参数')
    parser.add_argument('--seg_weight', type=float, default=0.4,
                        help='分割任务的损失权重')
    parser.add_argument('--position_weight', type=float, default=0.3,
                        help='位置分类任务的损失权重')
    parser.add_argument('--grade_weight', type=float, default=0.3,
                        help='等级回归任务的损失权重')
    
    # 优化器参数
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'],
                        help='优化器类型')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='学习率')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='最小学习率，用于学习率调度')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减系数')
    parser.add_argument('--lr_scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine', 'step', 'none'],
                        help='学习率调度器类型')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--no_cuda', action='store_true',
                        help='不使用CUDA')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--amp', action='store_true',
                        help='是否启用混合精度训练')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./unet_output',
                        help='输出目录路径')
    
    args = parser.parse_args()
    main(args) 