# 核心文件，完整运行项目不可缺少
# 训练脚本文件：实现模型训练、验证和测试的主要流程，包括数据加载、损失函数定义、优化器配置、训练循环、模型评估和结果保存等功能
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
from torch.amp import autocast, GradScaler  # 从torch.amp导入而不是torch.cuda.amp
import time
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import math

# 导入自定义模块
from dataset import CornRustDataset, get_dataloaders
from model import get_model
from utils import save_checkpoint, load_checkpoint, calculate_metrics, plot_metrics, download_bigearthnet_mini, FocalLoss, calculate_class_weights

# 定义数据增强变换
def get_data_transforms(train=True):
    """
    获取数据增强变换
    
    参数:
        train: 是否为训练模式，训练时应用数据增强，验证/测试时不应用
    
    返回:
        transforms: 数据增强变换组合
    """
    if train:
        # 训练时使用多种数据增强方法提高模型泛化能力
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),  # 随机水平翻转，增加样本多样性
            transforms.RandomVerticalFlip(),    # 随机垂直翻转
            transforms.RandomRotation(15),      # 随机旋转，角度范围为±15度
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)  # 随机颜色变化
        ])
    else:
        # 验证/测试时不进行数据增强，保持原始图像
        return None

# 添加混合损失函数类
class HybridLoss(nn.Module):
    """
    混合损失函数，结合多种损失函数以更好地处理玉米锈病特征
    
    组合：
    1. 位置分类：Focal Loss + Label Smoothing
    2. 等级回归：MSE + L1 Loss (平滑L1损失)
    3. 分割任务：BCE + Dice Loss (边界敏感)
    """
    def __init__(self, gamma=2.0, alpha=None, beta=0.25, smooth=1e-5, label_smoothing=0.1):
        super(HybridLoss, self).__init__()
        self.gamma = gamma  # Focal Loss参数
        self.alpha = alpha  # 类别权重
        self.beta = beta    # 平滑L1损失参数
        self.smooth = smooth  # Dice Loss平滑因子
        self.label_smoothing = label_smoothing  # 标签平滑参数
        
        # 分类损失 - 带标签平滑的Focal Loss
        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha)
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # 回归损失 - MSE + L1
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # 分割损失 - BCE + Dice
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def dice_loss(self, pred, target):
        """计算Dice损失，对边界更敏感"""
        pred = torch.sigmoid(pred)
        smooth = self.smooth
        
        # 展平预测和目标
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # 计算交集
        intersection = (pred_flat * target_flat).sum()
        
        # 计算Dice系数
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        
        return 1 - dice
    
    def smooth_l1_loss(self, pred, target):
        """平滑L1损失，对异常值更鲁棒"""
        beta = self.beta
        diff = torch.abs(pred - target)
        loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
        return loss.mean()
    
    def position_loss(self, pred, target):
        """位置分类的混合损失"""
        focal = self.focal_loss(pred, target)
        ce = self.ce_loss(pred, target)
        return 0.5 * focal + 0.5 * ce
    
    def grade_loss(self, pred, target):
        """等级回归的混合损失"""
        mse = self.mse_loss(pred, target)
        smooth_l1 = self.smooth_l1_loss(pred, target)
        return 0.6 * mse + 0.4 * smooth_l1
    
    def segmentation_loss(self, pred, target):
        """分割的混合损失"""
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return 0.5 * bce + 0.5 * dice

# 添加自适应任务权重平衡类
class AdaptiveTaskWeights:
    """
    自适应任务权重平衡机制，根据训练过程中各任务的相对难度动态调整权重
    
    基于各任务损失的梯度范数和学习进展来调整权重
    """
    def __init__(self, num_tasks=3, init_weights=None, gamma=0.5, beta=2.0):
        """
        初始化自适应任务权重
        
        参数:
            num_tasks: 任务数量
            init_weights: 初始权重，若不提供则平均分配
            gamma: 平滑参数，控制权重调整速度
            beta: 温度参数，控制任务间差异敏感度
        """
        self.num_tasks = num_tasks
        self.gamma = gamma
        self.beta = beta
        
        # 初始化权重
        if init_weights is None:
            self.weights = torch.ones(num_tasks) / num_tasks
        else:
            self.weights = torch.tensor(init_weights)
            # 确保权重和为1
            self.weights = self.weights / self.weights.sum()
        
        # 任务损失历史
        self.loss_history = []
        self.running_weights = self.weights.clone()
    
    def update(self, task_losses):
        """
        更新任务权重
        
        参数:
            task_losses: 各任务的当前损失值列表
        
        返回:
            更新后的权重
        """
        # 记录损失历史
        self.loss_history.append(task_losses)
        
        # 只在有足够历史数据时更新
        if len(self.loss_history) < 5:
            return self.weights
        
        # 计算最近5个批次的损失变化率
        recent_losses = torch.tensor(self.loss_history[-5:])
        loss_changes = (recent_losses[-1] / (recent_losses[0] + 1e-8))
        
        # 计算权重比例，变化大的任务给更大权重
        weight_factors = torch.exp(self.beta * loss_changes)
        new_weights = weight_factors / weight_factors.sum()
        
        # 平滑更新权重
        self.running_weights = (1 - self.gamma) * self.running_weights + self.gamma * new_weights
        self.running_weights = self.running_weights / self.running_weights.sum()
        
        return self.running_weights
    
    def get_weights(self):
        """获取当前权重"""
        return self.running_weights.tolist()

# 修改train_one_epoch函数，添加课程学习和自适应权重
def train_one_epoch(model, train_loader, optimizer, position_criterion, grade_criterion, device, 
                    task_weights=[0.7, 0.3, 0.0], scaler=None, segmentation_criterion=None,
                    epoch=0, total_epochs=50, adaptive_weights=None, hybrid_loss=None):
    """
    训练模型一个epoch，支持课程学习和自适应任务权重
    
    参数:
        model: 模型实例
        train_loader: 训练数据加载器
        optimizer: 优化器实例
        position_criterion: 位置分类的损失函数
        grade_criterion: 等级预测的损失函数
        device: 计算设备(CPU/GPU)
        task_weights: 任务权重，默认[0.7, 0.3, 0.0]
        scaler: 混合精度训练的GradScaler实例
        segmentation_criterion: 分割任务的损失函数
        epoch: 当前epoch
        total_epochs: 总epoch数
        adaptive_weights: 自适应权重实例
        hybrid_loss: 混合损失实例
        
    返回:
        dict: 包含训练指标的字典
    """
    print("进入train_one_epoch函数")
    model.train()  # 设置模型为训练模式
    
    # 初始化指标累积变量
    total_loss = 0.0
    position_loss_sum = 0.0
    grade_loss_sum = 0.0
    segmentation_loss_sum = 0.0
    position_correct = 0
    total_samples = 0
    grade_mae_sum = 0.0
    
    # 检查模型类型
    is_segnet = "SegNet" in model.__class__.__name__
    
    # 课程学习策略 - 分阶段训练
    # 第一阶段(前10%epoch): 主要训练分类
    # 第二阶段(10%-30%epoch): 分类+回归
    # 第三阶段(30%-100%epoch): 分类+回归+分割
    progress_ratio = epoch / total_epochs
    
    if progress_ratio < 0.1:  # 前10%的epoch
        curr_task_weights = [1.0, 0.0, 0.0]  # 只训练位置分类
        print(f"课程学习阶段1: 位置分类阶段 (权重={curr_task_weights})")
    elif progress_ratio < 0.3:  # 10%-30%的epoch
        curr_task_weights = [0.7, 0.3, 0.0]  # 位置分类 + 等级回归
        print(f"课程学习阶段2: 分类+回归阶段 (权重={curr_task_weights})")
    else:  # 30%-100%的epoch
        if adaptive_weights:
            curr_task_weights = adaptive_weights.get_weights()
            print(f"课程学习阶段3: 自适应权重阶段 (权重={curr_task_weights})")
        else:
            curr_task_weights = task_weights
            print(f"课程学习阶段3: 完整训练阶段 (权重={curr_task_weights})")
    
    # 使用tqdm显示进度条
    print(f"开始训练循环，训练集大小: {len(train_loader.dataset)}，批次大小: {train_loader.batch_size}")
    progress_bar = tqdm(train_loader, desc="训练中")
    
    # 遍历训练数据批次
    print("开始遍历数据批次...")
    batch_idx = 0
    for images, position_labels, grade_labels in progress_bar:
        print(f"处理批次 {batch_idx}，数据形状: 图像={images.shape}，位置={position_labels.shape}，等级={grade_labels.shape}")
        batch_idx += 1
        
        # 将数据移动到指定设备
        images = images.to(device)
        position_labels = position_labels.to(device).view(-1).long()
        grade_labels = grade_labels.float().unsqueeze(1).to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 使用混合精度训练
        if scaler is not None:
            with autocast(device_type='cuda'):
                # 前向传播 - 根据模型类型处理输出
                if is_segnet:
                    # SegNet模型输出三部分
                    position_logits, grade_values, segmentation = model(images)
                    
                    # 创建粗略分割标签
                    seg_targets = (torch.sum(images, dim=1, keepdim=True) > 0.1).float()
                    
                    # 使用混合损失或标准损失
                    if hybrid_loss:
                        # 使用混合损失
                        loss_position = hybrid_loss.position_loss(position_logits, position_labels)
                        loss_grade = hybrid_loss.grade_loss(grade_values, grade_labels)
                        loss_segmentation = hybrid_loss.segmentation_loss(segmentation, seg_targets)
                    else:
                        # 使用标准损失
                        loss_position = position_criterion(position_logits, position_labels)
                        loss_grade = grade_criterion(grade_values, grade_labels)
                        loss_segmentation = segmentation_criterion(segmentation, seg_targets) if segmentation_criterion else torch.tensor(0.0, device=device)
                    
                    # 使用任务权重组合损失
                    loss = (curr_task_weights[0] * loss_position + 
                            curr_task_weights[1] * loss_grade + 
                            curr_task_weights[2] * loss_segmentation)
                    
                    # 如果使用自适应权重，收集各任务损失用于更新
                    if adaptive_weights and progress_ratio >= 0.3:
                        task_losses = [loss_position.item(), loss_grade.item(), loss_segmentation.item()]
                        adaptive_weights.update(task_losses)
                else:
                    # 常规模型只输出两部分
                    position_logits, grade_values = model(images)
                    
                    # 使用混合损失或标准损失
                    if hybrid_loss:
                        loss_position = hybrid_loss.position_loss(position_logits, position_labels)
                        loss_grade = hybrid_loss.grade_loss(grade_values, grade_labels)
                    else:
                        loss_position = position_criterion(position_logits, position_labels)
                        loss_grade = grade_criterion(grade_values, grade_labels)
                    
                    # 使用任务权重组合损失
                    loss = curr_task_weights[0] * loss_position + curr_task_weights[1] * loss_grade
            
            # 使用scaler进行反向传播和参数更新
            scaler.scale(loss).backward()
            
            # 梯度裁剪，防止梯度爆炸
            if epoch < 5:  # 前5个epoch使用更严格的裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            # 常规训练流程（不使用混合精度）
            # 前向传播
            if is_segnet:
                position_logits, grade_values, segmentation = model(images)
                
                # 创建粗略分割标签
                seg_targets = (torch.sum(images, dim=1, keepdim=True) > 0.1).float()
                
                # 使用混合损失或标准损失
                if hybrid_loss:
                    loss_position = hybrid_loss.position_loss(position_logits, position_labels)
                    loss_grade = hybrid_loss.grade_loss(grade_values, grade_labels)
                    loss_segmentation = hybrid_loss.segmentation_loss(segmentation, seg_targets)
                else:
                    loss_position = position_criterion(position_logits, position_labels)
                    loss_grade = grade_criterion(grade_values, grade_labels)
                    loss_segmentation = segmentation_criterion(segmentation, seg_targets) if segmentation_criterion else torch.tensor(0.0, device=device)
                
                # 使用任务权重组合损失
                loss = (curr_task_weights[0] * loss_position + 
                        curr_task_weights[1] * loss_grade + 
                        curr_task_weights[2] * loss_segmentation)
                
                # 如果使用自适应权重，收集各任务损失用于更新
                if adaptive_weights and progress_ratio >= 0.3:
                    task_losses = [loss_position.item(), loss_grade.item(), loss_segmentation.item()]
                    adaptive_weights.update(task_losses)
            else:
                position_logits, grade_values = model(images)
                
                # 使用混合损失或标准损失
                if hybrid_loss:
                    loss_position = hybrid_loss.position_loss(position_logits, position_labels)
                    loss_grade = hybrid_loss.grade_loss(grade_values, grade_labels)
                else:
                    loss_position = position_criterion(position_logits, position_labels)
                    loss_grade = grade_criterion(grade_values, grade_labels)
                
                # 使用任务权重组合损失
                loss = curr_task_weights[0] * loss_position + curr_task_weights[1] * loss_grade
            
            # 反向传播和优化
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            if epoch < 5:  # 前5个epoch使用更严格的裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
            optimizer.step()
        
        # 统计指标
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        position_loss_sum += loss_position.item() * batch_size
        grade_loss_sum += loss_grade.item() * batch_size
        
        # 如果是SegNet模型且有分割损失，则累加分割损失
        if is_segnet and curr_task_weights[2] > 0:
            segmentation_loss_sum += loss_segmentation.item() * batch_size
        
        # 计算位置分类准确率
        _, position_preds = torch.max(position_logits, 1)
        position_correct += (position_preds == position_labels).sum().item()
        
        # 计算等级预测MAE
        grade_mae = torch.abs(grade_values - grade_labels).mean().item()
        grade_mae_sum += grade_mae * batch_size
        
        total_samples += batch_size
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': loss.item(),
            'pos_acc': position_correct / total_samples,
            'grade_mae': grade_mae_sum / total_samples
        })
    
    # 计算整个epoch的平均指标
    avg_loss = total_loss / total_samples
    avg_position_loss = position_loss_sum / total_samples
    avg_grade_loss = grade_loss_sum / total_samples
    position_accuracy = position_correct / total_samples
    grade_mae = grade_mae_sum / total_samples
    
    # 构建返回的指标字典
    metrics = {
        'loss': avg_loss,
        'position_loss': avg_position_loss,
        'grade_loss': avg_grade_loss,
        'position_accuracy': position_accuracy,
        'grade_mae': grade_mae
    }
    
    # 如果是SegNet模型且训练了分割任务，添加分割损失指标
    if is_segnet and curr_task_weights[2] > 0:
        avg_segmentation_loss = segmentation_loss_sum / total_samples
        metrics['segmentation_loss'] = avg_segmentation_loss
    
    # 返回包含所有训练指标的字典
    return metrics

def evaluate(model, val_loader, position_criterion, grade_criterion, device, task_weights=[0.7, 0.3, 0.0], segmentation_criterion=None):
    """
    评估模型在验证集上的性能
    
    参数:
        model: 模型实例
        val_loader: 验证数据加载器
        position_criterion: 位置分类的损失函数
        grade_criterion: 等级预测的损失函数（回归损失）
        device: 计算设备(CPU/GPU)
        task_weights: 任务权重，默认[0.7, 0.3, 0.0]表示位置任务占70%，等级任务占30%，分割任务0%（非SegNet）
        segmentation_criterion: 分割任务的损失函数（仅SegNet模型使用）
        
    返回:
        dict: 包含详细评估指标的字典，包括多种性能指标
    """
    model.eval()  # 设置模型为评估模式，禁用Dropout
    total_loss = 0.0  # 累计总损失
    position_loss_sum = 0.0  # 累计位置分类损失
    grade_loss_sum = 0.0  # 累计等级回归损失
    segmentation_loss_sum = 0.0  # 累计分割损失（仅SegNet模型）
    
    # 收集所有预测和真实标签，用于计算整体指标
    position_preds_all = []  # 所有位置预测
    position_labels_all = []  # 所有位置真实标签
    grade_values_all = []  # 所有等级预测
    grade_labels_all = []  # 所有等级真实标签
    
    # 检查模型类型，确定是否是SegNet模型
    is_segnet = "SegNet" in model.__class__.__name__
    
    with torch.no_grad():  # 关闭梯度计算，减少内存占用
        for images, position_labels, grade_labels in val_loader:
            # 将数据移动到指定设备
            images = images.to(device)
            position_labels = position_labels.to(device)
            
            # 确保位置标签是一维整数张量 (batch_size,)
            position_labels = position_labels.view(-1).long()
            
            # 将等级标签转换为float类型并添加维度，用于回归
            grade_labels = grade_labels.float().unsqueeze(1).to(device)
            
            # 使用混合精度计算（但不进行梯度计算，因为是验证阶段）
            with autocast(device_type='cuda', enabled=torch.cuda.is_available()):  # 指定设备类型为'cuda'
                # 前向传播 - 根据模型类型处理输出
                if is_segnet:
                    # SegNet模型输出三部分
                    position_logits, grade_values, segmentation = model(images)
                    
                    # 计算损失
                    loss_position = position_criterion(position_logits, position_labels)
                    loss_grade = grade_criterion(grade_values, grade_labels)
                    
                    # 计算分割损失（如果有分割标准）
                    if segmentation_criterion is not None:
                        # 创建粗略分割标签
                        seg_targets = (torch.sum(images, dim=1, keepdim=True) > 0.1).float()
                        loss_segmentation = segmentation_criterion(segmentation, seg_targets)
                    else:
                        loss_segmentation = torch.tensor(0.0, device=device)
                    
                    # 使用任务权重组合损失
                    loss = (task_weights[0] * loss_position + 
                            task_weights[1] * loss_grade + 
                            task_weights[2] * loss_segmentation)
                else:
                    # 常规模型只输出两部分
                    position_logits, grade_values = model(images)
                    
                    # 计算损失
                    loss_position = position_criterion(position_logits, position_labels)
                    loss_grade = grade_criterion(grade_values, grade_labels)
                    
                    # 使用任务权重组合损失
                    loss = task_weights[0] * loss_position + task_weights[1] * loss_grade
            
            # 统计指标
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            position_loss_sum += loss_position.item() * batch_size
            grade_loss_sum += loss_grade.item() * batch_size
            
            # 如果是SegNet模型且有分割损失，则累加分割损失
            if is_segnet and segmentation_criterion is not None:
                segmentation_loss_sum += loss_segmentation.item() * batch_size
            
            # 获取位置预测类别
            _, position_preds = torch.max(position_logits, 1)
            
            # 收集预测和标签用于计算整体指标
            position_preds_all.extend(position_preds.cpu().numpy())
            position_labels_all.extend(position_labels.cpu().numpy())
            grade_values_all.extend(grade_values.cpu().numpy())
            grade_labels_all.extend(grade_labels.cpu().numpy())
    
    # 计算平均指标
    total_samples = len(val_loader.dataset)
    avg_loss = total_loss / total_samples
    avg_position_loss = position_loss_sum / total_samples
    avg_grade_loss = grade_loss_sum / total_samples
    
    # 计算位置分类详细指标
    position_accuracy = accuracy_score(position_labels_all, position_preds_all)  # 准确率
    position_f1 = f1_score(position_labels_all, position_preds_all, average='macro')  # 宏平均F1
    position_f1_per_class = f1_score(position_labels_all, position_preds_all, average=None)  # 每类F1
    position_cm = confusion_matrix(position_labels_all, position_preds_all)  # 混淆矩阵
    position_precision = precision_score(position_labels_all, position_preds_all, average='macro')  # 宏平均精确率
    position_recall = recall_score(position_labels_all, position_preds_all, average='macro')  # 宏平均召回率
    
    # 转换等级预测和标签为numpy数组
    grade_values_all = np.array(grade_values_all).flatten()
    grade_labels_all = np.array(grade_labels_all).flatten()
    
    # 计算等级预测MAE和容忍误差内准确率
    grade_mae = np.mean(np.abs(grade_values_all - grade_labels_all))  # 平均绝对误差
    
    # 构建返回的指标字典
    metrics = {
        'loss': avg_loss,
        'position_loss': avg_position_loss,
        'grade_loss': avg_grade_loss,
        'position_accuracy': position_accuracy,
        'position_f1': position_f1,
        'position_precision': position_precision,
        'position_recall': position_recall,
        'position_f1_per_class': position_f1_per_class.tolist(),  # 转换为Python列表以便存储
        'position_cm': position_cm.tolist(),  # 转换为Python列表以便存储
        'grade_mae': grade_mae
    }
    
    # 如果是SegNet模型，添加分割损失指标
    if is_segnet and segmentation_criterion is not None:
        avg_segmentation_loss = segmentation_loss_sum / total_samples
        metrics['segmentation_loss'] = avg_segmentation_loss
    
    return metrics

def plot_confusion_matrix(cm, class_names, title, save_path=None):
    """
    绘制混淆矩阵可视化图
    
    参数:
        cm: 混淆矩阵
        class_names: 类别名称列表
        title: 图表标题
        save_path: 保存路径，如果提供则保存图像
    """
    plt.figure(figsize=(10, 8))  # 设置图像大小
    
    # 使用seaborn绘制热图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    # 添加标签和标题
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title(title)
    
    # 保存图像
    if save_path:
        plt.savefig(save_path)
    plt.close()  # 关闭图形，释放内存

def plot_metrics(metrics_history, save_dir):
    """
    绘制训练过程中的指标变化曲线
    
    参数:
        metrics_history: 包含各指标历史记录的字典，格式为 {'train': [train_metrics列表], 'val': [val_metrics列表]}
        save_dir: 图像保存目录
    """
    # 确保有训练历史数据
    if len(metrics_history['train']) == 0 or len(metrics_history['val']) == 0:
        print("警告：没有足够的训练历史数据进行绘图")
        return
    
    # 创建一个2x2的图表布局，显示4种主要指标
    plt.figure(figsize=(16, 12))
    
    # 准备x轴数据
    epochs = range(1, len(metrics_history['train']) + 1)
    
    # 绘制损失曲线 - 左上角
    plt.subplot(2, 2, 1)
    plt.plot(epochs, [m['loss'] for m in metrics_history['train']], label='训练损失')
    plt.plot(epochs, [m['loss'] for m in metrics_history['val']], label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练和验证损失')
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # 确保x轴只显示整数
    
    # 绘制位置准确率和F1曲线 - 右上角
    plt.subplot(2, 2, 2)
    plt.plot(epochs, [m['position_accuracy'] for m in metrics_history['train']], label='训练位置准确率')
    plt.plot(epochs, [m['position_accuracy'] for m in metrics_history['val']], label='验证位置准确率')
    plt.plot(epochs, [m['position_f1'] if 'position_f1' in m else 0 for m in metrics_history['val']], label='验证位置F1')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy / F1')
    plt.title('位置分类性能')
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 绘制等级MAE曲线 - 左下角
    plt.subplot(2, 2, 3)
    plt.plot(epochs, [m['grade_mae'] for m in metrics_history['train']], label='训练等级MAE')
    plt.plot(epochs, [m['grade_mae'] for m in metrics_history['val']], label='验证等级MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('等级预测平均绝对误差')
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 绘制分割损失曲线（如果存在）或其他指标 - 右下角
    plt.subplot(2, 2, 4)
    
    # 检查是否有分割损失
    if 'segmentation_loss' in metrics_history['val'][0]:
        plt.plot(epochs, [m.get('segmentation_loss', 0) for m in metrics_history['train']], label='训练分割损失')
        plt.plot(epochs, [m.get('segmentation_loss', 0) for m in metrics_history['val']], label='验证分割损失')
        plt.title('分割任务损失')
        plt.ylabel('Loss')
    else:
        # 如果没有分割损失，显示位置指标
        plt.plot(epochs, [m.get('position_precision', 0) for m in metrics_history['val']], label='验证精确率')
        plt.plot(epochs, [m.get('position_recall', 0) for m in metrics_history['val']], label='验证召回率')
        plt.title('位置分类精确率和召回率')
        plt.ylabel('Value')
    
    plt.xlabel('Epoch')
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 调整子图布局并保存
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    plt.close()  # 关闭图形，释放内存

def main(args):
    """
    主训练函数，增加了课程学习和优化器配置
    
    参数:
        args: 命令行参数
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"使用设备: {device}")
    
    # GPU优化配置
    if torch.cuda.is_available() and not args.no_cuda:
        # 设置cudnn为自动优化模式 - 根据硬件自动选择最高效的算法
        torch.backends.cudnn.benchmark = True
        
        # 设置GPU内存分配策略 - 尽可能预先分配所需内存而不是动态增长
        torch.cuda.empty_cache()  # 清空缓存
        
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
    
    # 添加路径调试信息
    print(f"\n数据配置:")
    print(f"数据根目录: {os.path.abspath(args.data_root)}")
    print(f"JSON标注目录: {os.path.abspath(args.json_root)}")
    
    # 检查目录是否存在
    if not os.path.exists(args.data_root):
        raise FileNotFoundError(f"数据根目录不存在: {args.data_root}")
    if not os.path.exists(args.json_root):
        raise FileNotFoundError(f"JSON标注目录不存在: {args.json_root}")
        
    # 检查目录内容
    try:
        data_contents = os.listdir(args.data_root)
        json_contents = os.listdir(args.json_root)
        print(f"数据根目录包含 {len(data_contents)} 个项目")
        print(f"JSON标注目录包含 {len(json_contents)} 个项目")
    except Exception as e:
        print(f"读取目录内容时出错: {e}")
    
    # 创建数据加载器
    try:
        train_loader, val_loader = get_dataloaders(
            data_root=args.data_root,
            json_root=args.json_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            img_size=args.img_size,
            train_ratio=args.train_ratio,
            aug_prob=args.aug_prob,
            use_extended_dataset=True,  # 启用扩展数据集
            pin_memory=torch.cuda.is_available()  # 使用pinned memory提高GPU传输速度
        )
        
        # 预热缓存 - 预加载部分数据到GPU以减少开始时的停顿
        if torch.cuda.is_available() and not args.no_cuda:
            warmup_loader = DataLoader(
                train_loader.dataset, 
                batch_size=2, 
                shuffle=False, 
                num_workers=1
            )
            warmup_iter = iter(warmup_loader)
            batch = next(warmup_iter)
            for item in batch:
                if isinstance(item, torch.Tensor):
                    _ = item.to(device)
            print("已完成GPU数据预热")
            
    except Exception as e:
        print(f"创建数据加载器时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e
        
    # 检查数据集大小
    if len(train_loader.dataset) == 0:
        raise ValueError("训练数据集大小为0，请检查数据路径和标注文件是否匹配")
    
    # 创建模型
    model = get_model(model_type=args.model_type, in_channels=args.in_channels, img_size=args.img_size)
    model = model.to(device)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总量: {total_params:,}")
    
    # 检查是否是SegNet模型
    is_segnet = "SegNet" in model.__class__.__name__
    
    # 解析任务权重
    task_weights = [float(w) for w in args.task_weights.split(',')]
    # 确保任务权重数量正确
    if is_segnet and len(task_weights) < 3:
        task_weights.append(0.2)
        total = sum(task_weights)
        task_weights = [w / total for w in task_weights]
        print(f"自动调整任务权重用于SegNet模型: 位置分类={task_weights[0]:.2f}, 等级回归={task_weights[1]:.2f}, 分割={task_weights[2]:.2f}")
    elif not is_segnet and len(task_weights) > 2:
        task_weights = task_weights[:2]
        total = sum(task_weights)
        task_weights = [w / total for w in task_weights]
        print(f"任务权重: 位置分类={task_weights[0]:.2f}, 等级回归={task_weights[1]:.2f}")
    else:
        if is_segnet:
            print(f"任务权重: 位置分类={task_weights[0]:.2f}, 等级回归={task_weights[1]:.2f}, 分割={task_weights[2]:.2f}")
        else:
            print(f"任务权重: 位置分类={task_weights[0]:.2f}, 等级回归={task_weights[1]:.2f}")
    
    # 使用混合损失函数
    hybrid_loss = HybridLoss(gamma=args.focal_gamma) if args.use_hybrid_loss else None
    if hybrid_loss:
        print("使用混合损失函数 (Focal+Label Smoothing, MSE+L1, BCE+Dice)")
    
    # 创建自适应任务权重实例
    adaptive_weights = AdaptiveTaskWeights(
        num_tasks=3 if is_segnet else 2,
        init_weights=task_weights,
        gamma=0.3,  # 平滑参数
        beta=2.0    # 温度参数
    ) if args.adaptive_weights else None
    
    if adaptive_weights:
        print("使用自适应任务权重平衡机制")
    
    # 定义损失函数
    if args.loss_type == 'focal':
        # 使用Focal Loss处理类别不平衡
        print("使用Focal Loss")
        position_criterion = FocalLoss(gamma=args.focal_gamma)
        grade_criterion = nn.MSELoss()  # 等级回归仍使用MSE
    else:
        # 标准交叉熵损失
        print("使用CrossEntropy Loss")
        position_criterion = nn.CrossEntropyLoss(weight=position_weights)
        grade_criterion = nn.MSELoss()
    
    # 定义分割损失函数（仅用于SegNet模型）
    segmentation_criterion = None
    if is_segnet:
        # 使用二元交叉熵损失进行分割任务
        print("为分割任务添加BinaryCrossEntropy Loss")
        segmentation_criterion = nn.BCEWithLogitsLoss()
    
    # 定义优化器 - 使用带权重衰减的AdamW
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f"使用AdamW优化器，学习率={args.lr}，权重衰减={args.weight_decay}")
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f"使用Adam优化器，学习率={args.lr}，权重衰减={args.weight_decay}")
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        print(f"使用SGD优化器，学习率={args.lr}，动量=0.9，权重衰减={args.weight_decay}")
    
    # 定义学习率调度器 - 使用余弦退火预热重启
    if args.lr_scheduler == 'cosine':
        # 余弦退火调度器，带预热和重启
        T_0 = max(1, args.epochs // 3)  # 确保T_0至少为1
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=T_0,  # 第一次重启的周期
            T_mult=2,              # 每次重启后周期乘数
            eta_min=args.min_lr    # 最小学习率
        )
        print(f"使用余弦退火学习率调度器，周期={T_0}，最小学习率={args.min_lr}")
    elif args.lr_scheduler == 'plateau':
        # 当指标停止改善时降低学习率
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=args.min_lr, verbose=True
        )
        print(f"使用ReduceLROnPlateau学习率调度器，耐心=5，最小学习率={args.min_lr}")
    elif args.lr_scheduler == 'step':
        # 步进调度，每10轮降低学习率
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.5
        )
        print(f"使用StepLR学习率调度器，步长=10，衰减率=0.5")
    else:
        scheduler = None
        print("不使用学习率调度器")
    
    # 如果提供了检查点路径，恢复训练
    start_epoch = args.start_epoch  # 从命令行参数获取开始轮次
    best_val_loss = float('inf')
    best_f1 = 0.0
    metrics_history = {'train': [], 'val': []}
    
    # 优先从权重文件加载
    if args.load_weights and os.path.isfile(args.load_weights):
        try:
            model.load_state_dict(torch.load(args.load_weights, map_location=device))
            print(f"从权重文件加载参数: {args.load_weights}")
            print(f"从轮次 {start_epoch} 开始训练")
        except Exception as e:
            print(f"加载权重文件时出错: {e}")
            print("尝试从头开始训练...")
            start_epoch = 0
    # 如果没有提供权重文件，但提供了检查点路径
    elif args.resume:
        if os.path.isfile(args.resume):
            try:
                # 显式设置weights_only=False以处理PyTorch 2.6兼容性问题
                checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if not args.start_epoch:  # 如果没有明确指定start_epoch
                    start_epoch = checkpoint['epoch'] + 1
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                best_f1 = checkpoint.get('best_f1', 0.0)
                metrics_history = checkpoint.get('metrics_history', {'train': [], 'val': []})
                
                print(f"从检查点恢复训练: {args.resume}")
                print(f"继续从轮次 {start_epoch} 开始")
            except Exception as e:
                print(f"加载检查点时出错: {e}")
                print("尝试从头开始训练...")
                start_epoch = 0
        else:
            print(f"检查点不存在: {args.resume}")
    
    # 训练循环
    print(f"\n开始训练 - 总轮次: {args.epochs}")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n轮次 [{epoch+1}/{args.epochs}]")
        
        # 训练阶段
        train_metrics = train_one_epoch(
            model, 
            train_loader, 
            optimizer, 
            position_criterion, 
            grade_criterion, 
            device, 
            task_weights=task_weights, 
            scaler=scaler,
            segmentation_criterion=segmentation_criterion,
            epoch=epoch,
            total_epochs=args.epochs,
            adaptive_weights=adaptive_weights,
            hybrid_loss=hybrid_loss
        )
        
        # 验证阶段
        val_metrics = evaluate(
            model, 
            val_loader, 
            position_criterion, 
            grade_criterion, 
            device, 
            task_weights=task_weights,
            segmentation_criterion=segmentation_criterion
        )
        
        # 打印当前指标
        print(f"训练 | 损失: {train_metrics['loss']:.4f}, 位置准确率: {train_metrics['position_accuracy']:.4f}, 等级MAE: {train_metrics['grade_mae']:.4f}")
        print(f"验证 | 损失: {val_metrics['loss']:.4f}, 位置准确率: {val_metrics['position_accuracy']:.4f}, F1: {val_metrics['position_f1']:.4f}, 等级MAE: {val_metrics['grade_mae']:.4f}")
        
        # 打印详细指标
        print(f"位置分类 - 精确率: {val_metrics['position_precision']:.4f}, 召回率: {val_metrics['position_recall']:.4f}")
        print(f"位置分类 - 各类F1: {[f'{f:.4f}' for f in val_metrics['position_f1_per_class']]}")
        
        # 如果是SegNet模型，打印分割指标
        if is_segnet and 'segmentation_loss' in val_metrics:
            print(f"分割 - 损失: {val_metrics['segmentation_loss']:.4f}")
        
        # 记录训练历史
        metrics_history['train'].append(train_metrics)
        metrics_history['val'].append(val_metrics)
        
        # 更新学习率
        if scheduler:
            if args.lr_scheduler == 'plateau':
                # 用验证损失更新ReduceLROnPlateau调度器
                scheduler.step(val_metrics['loss'])
            else:
                # 其他调度器按轮次更新
                scheduler.step()
            
            # 打印当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            print(f"当前学习率: {current_lr:.2e}")
        
        # 保存检查点
        is_best_loss = val_metrics['loss'] < best_val_loss
        is_best_f1 = val_metrics['position_f1'] > best_f1
        
        # 更新最佳指标
        if is_best_loss:
            best_val_loss = val_metrics['loss']
            print(f"新的最佳验证损失: {best_val_loss:.4f}")
        
        if is_best_f1:
            best_f1 = val_metrics['position_f1']
            print(f"新的最佳F1分数: {best_f1:.4f}")
        
        # 保存每个轮次的模型
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'best_f1': best_f1,
            'metrics_history': metrics_history
        }, is_best_loss, is_best_f1, epoch, args.output_dir)
        
        # 创建并保存训练指标图表
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            plot_metrics(metrics_history, args.output_dir)
    
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'last_model.pth'))
    print(f"训练完成！最终模型已保存至 {os.path.join(args.output_dir, 'last_model.pth')}")
    
    # 保存训练历史
    plot_metrics(metrics_history, args.output_dir)
    
    # 返回最佳指标
    return best_val_loss, best_f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='玉米南方锈病分类模型训练')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, default='./guanceng-bit',
                        help='数据根目录路径')
    parser.add_argument('--json_root', type=str, default='./biaozhu_json',
                        help='JSON标注根目录路径，如不提供则从data_root自动推断')
    parser.add_argument('--img_size', type=int, default=128,
                        help='图像大小')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='输入图像通道数')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例')
    parser.add_argument('--aug_prob', type=float, default=0.7,
                        help='数据增强应用概率')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='segnet',
                        choices=['simple', 'resnet', 'resnet+', 'segnet', 'segnet+'],
                        help='模型类型')
    
    # 损失函数参数
    parser.add_argument('--loss_type', type=str, default='focal',
                        choices=['ce', 'focal'],
                        help='损失函数类型(ce=CrossEntropy, focal=FocalLoss)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss的gamma参数')
    parser.add_argument('--weighted_loss', action='store_true',
                        help='是否使用加权损失函数处理类别不平衡')
    parser.add_argument('--task_weights', type=str, default='0.5,0.3,0.2',
                        help='多任务权重，用逗号分隔，例如"0.5,0.3,0.2" 表示位置任务、等级任务和分割任务的权重')
    
    # 优化器参数
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw', 'sgd'],
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
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练的路径')
    parser.add_argument('--no_cuda', action='store_true',
                        help='不使用CUDA')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='是否启用混合精度训练（自动混合精度，适用于RTX GPU）')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./output_segnet',
                        help='输出目录路径')
    
    # 添加从权重文件加载的参数
    parser.add_argument('--load_weights', type=str, default=None,
                        help='从权重文件直接加载模型参数的路径')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='开始训练的轮次')
    
    # 添加新的参数
    parser.add_argument('--use_hybrid_loss', action='store_true',
                        help='使用混合损失函数')
    parser.add_argument('--adaptive_weights', action='store_true',
                        help='使用自适应任务权重')
    
    args = parser.parse_args()
    main(args)