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

def train_one_epoch(model, train_loader, optimizer, position_criterion, grade_criterion, device, task_weights=[0.7, 0.3], scaler=None):
    """
    训练模型一个epoch
    
    参数:
        model: 模型实例
        train_loader: 训练数据加载器
        optimizer: 优化器实例
        position_criterion: 位置分类的损失函数(CrossEntropy)
        grade_criterion: 等级预测的损失函数(MSE回归损失)
        device: 计算设备(CPU/GPU)
        task_weights: 任务权重，默认[0.7, 0.3]表示位置任务占70%，等级任务占30%
        scaler: 混合精度训练的GradScaler实例
        
    返回:
        dict: 包含训练指标的字典，包括总损失、位置损失、等级损失、位置准确率和等级MAE
    """
    print("进入train_one_epoch函数")
    model.train()  # 设置模型为训练模式，启用Dropout和BatchNorm
    total_loss = 0.0  # 累计总损失
    position_loss_sum = 0.0  # 累计位置分类损失
    grade_loss_sum = 0.0  # 累计等级回归损失
    position_correct = 0  # 位置分类正确样本数
    total_samples = 0  # 总样本数
    grade_mae_sum = 0.0  # 等级预测平均绝对误差累计
    
    # 使用tqdm显示进度条，增强用户体验
    print(f"开始训练循环，训练集大小: {len(train_loader.dataset)}，批次大小: {train_loader.batch_size}")
    progress_bar = tqdm(train_loader, desc="训练中")
    
    # 遍历训练数据批次
    print("开始遍历数据批次...")
    batch_idx = 0
    for images, position_labels, grade_labels in progress_bar:
        print(f"处理批次 {batch_idx}，数据形状: 图像={images.shape}，位置={position_labels.shape}，等级={grade_labels.shape}")
        batch_idx += 1
        # 将数据移动到指定设备
        images = images.to(device)  # 输入图像
        position_labels = position_labels.to(device)  # 位置标签（0,1,2）
        
        # 确保位置标签是一维整数张量 (batch_size,)
        position_labels = position_labels.view(-1).long()
        
        # 将等级标签转换为float类型并添加维度，用于回归任务
        # 从形状[batch_size]变为[batch_size, 1]
        grade_labels = grade_labels.float().unsqueeze(1).to(device)
        
        # 清零梯度
        optimizer.zero_grad()  
        
        # 使用混合精度训练
        if scaler is not None:
            with autocast(device_type='cuda'):  # 指定设备类型为'cuda'
                # 前向传播
                position_logits, grade_values = model(images)
                
                # 计算位置分类损失 - 使用CrossEntropy
                loss_position = position_criterion(position_logits, position_labels)
                
                # 计算等级回归损失 - 使用MSE
                loss_grade = grade_criterion(grade_values, grade_labels)
                
                # 使用任务权重组合损失
                loss = task_weights[0] * loss_position + task_weights[1] * loss_grade
            
            # 使用scaler进行反向传播和参数更新
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 常规训练流程（不使用混合精度）
            # 前向传播
            position_logits, grade_values = model(images)
            
            # 计算位置分类损失 - 使用CrossEntropy
            loss_position = position_criterion(position_logits, position_labels)
            
            # 计算等级回归损失 - 使用MSE
            loss_grade = grade_criterion(grade_values, grade_labels)
            
            # 使用任务权重组合损失
            loss = task_weights[0] * loss_position + task_weights[1] * loss_grade
            
            # 反向传播和优化
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数
        
        # 统计指标
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size  # 累加总损失
        position_loss_sum += loss_position.item() * batch_size  # 累加位置损失
        grade_loss_sum += loss_grade.item() * batch_size  # 累加等级损失
        
        # 计算位置分类准确率
        _, position_preds = torch.max(position_logits, 1)  # 获取预测类别
        position_correct += (position_preds == position_labels).sum().item()  # 统计正确预测数
        
        # 计算等级预测MAE(平均绝对误差)
        grade_mae = torch.abs(grade_values - grade_labels).mean().item()  # 当前批次的MAE
        grade_mae_sum += grade_mae * batch_size  # 累加MAE
        
        total_samples += batch_size  # 累加样本数
        
        # 更新进度条显示当前性能指标
        progress_bar.set_postfix({
            'loss': loss.item(),  # 当前批次损失
            'pos_acc': position_correct / total_samples,  # 当前位置准确率
            'grade_mae': grade_mae_sum / total_samples  # 当前平均等级MAE
        })
    
    # 计算整个epoch的平均指标
    avg_loss = total_loss / total_samples  # 平均总损失
    avg_position_loss = position_loss_sum / total_samples  # 平均位置损失
    avg_grade_loss = grade_loss_sum / total_samples  # 平均等级损失
    position_accuracy = position_correct / total_samples  # 位置分类准确率
    grade_mae = grade_mae_sum / total_samples  # 等级预测平均MAE
    
    # 返回包含所有训练指标的字典
    return {
        'loss': avg_loss,
        'position_loss': avg_position_loss,
        'grade_loss': avg_grade_loss,
        'position_accuracy': position_accuracy,
        'grade_mae': grade_mae
    }

def evaluate(model, val_loader, position_criterion, grade_criterion, device, task_weights=[0.7, 0.3]):
    """
    评估模型在验证集上的性能
    
    参数:
        model: 模型实例
        val_loader: 验证数据加载器
        position_criterion: 位置分类的损失函数
        grade_criterion: 等级预测的损失函数（回归损失）
        device: 计算设备(CPU/GPU)
        task_weights: 任务权重，默认[0.7, 0.3]表示位置任务占70%，等级任务占30%
        
    返回:
        dict: 包含详细评估指标的字典，包括多种性能指标
    """
    model.eval()  # 设置模型为评估模式，禁用Dropout
    total_loss = 0.0  # 累计总损失
    position_loss_sum = 0.0  # 累计位置分类损失
    grade_loss_sum = 0.0  # 累计等级回归损失
    
    # 收集所有预测和真实标签，用于计算整体指标
    position_preds_all = []  # 所有位置预测
    position_labels_all = []  # 所有位置真实标签
    grade_values_all = []  # 所有等级预测
    grade_labels_all = []  # 所有等级真实标签
    
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
                # 前向传播
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
    
    # 计算等级回归指标
    grade_values_all = np.array(grade_values_all)
    grade_labels_all = np.array(grade_labels_all)
    grade_mae = np.mean(np.abs(grade_values_all - grade_labels_all))  # 平均绝对误差
    
    # 计算±2误差容忍率 - 在实际应用中，等级预测误差在±2范围内可接受
    tolerance = 2.0
    grade_tolerance_accuracy = np.mean(np.abs(grade_values_all - grade_labels_all) <= tolerance)
    
    # 返回包含所有评估指标的字典
    return {
        'loss': avg_loss,
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
        metrics_history: 包含各指标历史记录的字典
        save_dir: 图像保存目录
    """
    # 创建一个2x2的图表布局，显示4种主要指标
    plt.figure(figsize=(16, 12))
    
    # 绘制损失曲线 - 右上角
    plt.subplot(2, 2, 1)
    plt.plot(metrics_history['train_loss'], label='训练损失')
    plt.plot(metrics_history['val_loss'], label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练和验证损失')
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # 确保x轴只显示整数
    
    # 绘制位置准确率和F1曲线 - 右上角
    plt.subplot(2, 2, 2)
    plt.plot(metrics_history['train_position_accuracy'], label='训练位置准确率')
    plt.plot(metrics_history['val_position_accuracy'], label='验证位置准确率')
    plt.plot(metrics_history['val_position_f1'], label='验证位置F1')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy / F1')
    plt.title('位置分类性能')
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 绘制等级MAE曲线 - 左下角
    plt.subplot(2, 2, 3)
    plt.plot(metrics_history['train_grade_mae'], label='训练等级MAE')
    plt.plot(metrics_history['val_grade_mae'], label='验证等级MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('等级预测平均绝对误差')
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 绘制等级容忍率曲线 - 右下角
    plt.subplot(2, 2, 4)
    plt.plot(metrics_history['val_grade_tolerance'], label='验证等级容忍率(±2)')
    plt.xlabel('Epoch')
    plt.ylabel('Tolerance Rate')
    plt.title('等级预测容忍率')
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 调整子图布局并保存
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    plt.close()  # 关闭图形，释放内存

def main(args):
    """
    主训练函数，处理训练和验证流程
    
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
        train_loader, val_loader, test_loader = get_dataloaders(
            data_root=args.data_root,
            json_root=args.json_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            img_size=args.img_size,
            train_ratio=0.8,  # 固定为8:1:1的比例
            val_ratio=0.1,
            test_ratio=0.1,
            aug_prob=args.aug_prob,
            use_extended_dataset=True,  # 启用扩展数据集
            pin_memory=torch.cuda.is_available()  # 使用pinned memory提高GPU传输速度
        )
        
        # 打印数据集分割信息
        print(f"数据集分割: 训练集={len(train_loader.dataset)}样本, 验证集={len(val_loader.dataset)}样本, 测试集={len(test_loader.dataset)}样本")
        
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
    
    # 解析任务权重(字符串如"0.7,0.3"转为列表[0.7, 0.3])
    task_weights = [float(w) for w in args.task_weights.split(',')]
    print(f"任务权重: 位置分类={task_weights[0]}, 等级分类={task_weights[1]}")
    
    # 获取类别权重 - 用于处理类别不平衡
    position_weights = None
    grade_weights = None
    
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
    
    # 定义优化器
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
    # 定义学习率调度器
    if args.lr_scheduler == 'plateau':
        # 当指标停止改善时降低学习率
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=args.min_lr, verbose=True
        )
    elif args.lr_scheduler == 'cosine':
        # 余弦退火调度
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr
        )
    elif args.lr_scheduler == 'step':
        # 步进调度，每10轮降低学习率
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.5
        )
    else:
        scheduler = None
    
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
    no_improvement_count = 0  # 用于早停
    
    print(f"开始训练，共 {args.epochs} 轮...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\n轮次 {epoch+1}/{args.epochs}")
        
        # 训练一个轮次
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, position_criterion, grade_criterion, device, 
            task_weights=task_weights, scaler=scaler
        )
        
        # 验证
        val_metrics = evaluate(
            model, val_loader, position_criterion, grade_criterion, device, 
            task_weights=task_weights
        )
        
        # 保存指标历史
        metrics_history['train'].append(train_metrics)
        metrics_history['val'].append(val_metrics)
        
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
            no_improvement_count = 0
        elif current_f1 > best_f1:
            best_f1 = current_f1
            is_best = True
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        # 保存检查点
        start_time = time.time()
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'best_f1': best_f1,
            'metrics_history': metrics_history,
            'training_time': time.time() - start_time if 'start_time' in locals() else 0
        }
        
        # 保存最后一轮模型
        # 使用torch.save确保PyTorch 2.6兼容性
        try:
            torch.save(checkpoint, os.path.join(args.output_dir, 'last_model.pth'))
        except Exception as e:
            print(f"保存最后一轮模型时出错: {e}")
            # 尝试仅保存模型权重
            print("尝试仅保存模型权重...")
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'last_model_weights.pth'))
        
        # 保存最佳模型
        if is_best:
            try:
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
                print("发现最佳模型，已保存!")
            except Exception as e:
                print(f"保存最佳模型时出错: {e}")
        
        # 早停
        if no_improvement_count >= args.patience:
            print(f"验证指标 {args.patience} 轮没有改善，提前停止训练")
            break
    
    # 绘制训练过程指标
    plot_metrics(metrics_history, args.output_dir)
    
    # 加载最佳模型进行最终评估
    best_model_path = os.path.join(args.output_dir, 'best_model.pth')
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    # 最终验证评估
    final_metrics = evaluate(
        model, val_loader, position_criterion, grade_criterion, device, 
        task_weights=task_weights
    )
    
    # 绘制混淆矩阵
    plot_confusion_matrix(
        final_metrics['position_cm'],
        ['下部', '中部', '上部'],
        f"位置分类混淆矩阵 (准确率: {final_metrics['position_accuracy']:.4f})",
        os.path.join(args.output_dir, 'position_confusion_matrix.png')
    )
    
    # 测试集评估
    print("\n=== 测试集评估 ===")
    test_metrics = evaluate(
        model, test_loader, position_criterion, grade_criterion, device, 
        task_weights=task_weights
    )
    
    # 在测试集上绘制混淆矩阵
    plot_confusion_matrix(
        test_metrics['position_cm'],
        ['下部', '中部', '上部'],
        f"测试集位置分类混淆矩阵 (准确率: {test_metrics['position_accuracy']:.4f})",
        os.path.join(args.output_dir, 'test_position_confusion_matrix.png')
    )
    
    # 保存测试结果到文件
    with open(os.path.join(args.output_dir, 'test_results.txt'), 'w') as f:
        f.write("====== 测试集评估结果 ======\n")
        f.write(f"位置分类准确率: {test_metrics['position_accuracy']:.4f}\n")
        f.write(f"位置分类F1分数: {test_metrics['position_f1']:.4f}\n")
        f.write(f"位置分类精确率: {test_metrics['position_precision']:.4f}\n")
        f.write(f"位置分类召回率: {test_metrics['position_recall']:.4f}\n")
        f.write(f"等级预测MAE: {test_metrics['grade_mae']:.4f}\n")
        f.write(f"等级预测±2容忍率: {test_metrics['grade_tolerance_accuracy']:.4f}\n")
    
    # 打印最终结果
    print("\n训练完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"最佳位置F1分数: {best_f1:.4f}")
    print(f"验证集位置预测准确率: {final_metrics['position_accuracy']:.4f}")
    print(f"验证集等级预测MAE: {final_metrics['grade_mae']:.4f}")
    print(f"\n测试集位置预测准确率: {test_metrics['position_accuracy']:.4f}")
    print(f"测试集位置F1分数: {test_metrics['position_f1']:.4f}")
    print(f"测试集等级预测MAE: {test_metrics['grade_mae']:.4f}")
    
    return model, final_metrics

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
    parser.add_argument('--model_type', type=str, default='resnet_plus',
                        choices=['simple', 'resnet', 'resnet_plus'],
                        help='模型类型')
    
    # 损失函数参数
    parser.add_argument('--loss_type', type=str, default='focal',
                        choices=['ce', 'focal'],
                        help='损失函数类型(ce=CrossEntropy, focal=FocalLoss)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss的gamma参数')
    parser.add_argument('--weighted_loss', action='store_true',
                        help='是否使用加权损失函数处理类别不平衡')
    parser.add_argument('--task_weights', type=str, default='0.5,0.5',
                        help='多任务权重，用逗号分隔，例如"0.7,0.3"')
    
    # 优化器参数
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'],
                        help='优化器类型')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='最小学习率，用于学习率调度')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减系数')
    parser.add_argument('--lr_scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine', 'step', 'none'],
                        help='学习率调度器类型')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=80,
                        help='训练轮数')
    parser.add_argument('--patience', type=int, default=10,
                        help='早停耐心值，连续多少轮验证指标无改善时停止训练')
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练的路径')
    parser.add_argument('--no_cuda', action='store_true',
                        help='不使用CUDA')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--amp', action='store_true',
                        help='是否启用混合精度训练（自动混合精度，适用于RTX GPU）')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='输出目录路径')
    
    # 添加从权重文件加载的参数
    parser.add_argument('--load_weights', type=str, default=None,
                        help='从权重文件直接加载模型参数的路径')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='开始训练的轮次')
    
    args = parser.parse_args()
    main(args)