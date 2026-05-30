# 工具函数文件：提供各种辅助功能，包括自定义损失函数(FocalLoss)、模型保存与加载、性能指标计算、数据可视化以及数据集下载与生成工具
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import os
import gdown
import tarfile
import torch.nn as nn

class FocalLoss(nn.Module):
    """
    Focal Loss实现，针对类别不平衡问题
    
    FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)
    
    其中:
    - pt: 模型对真实类别的预测概率
    - gamma: 聚焦参数，增大时更关注困难样本（难以正确分类的样本）
    - alpha: 类别权重参数，形状为[num_classes]，为少数类赋予更高权重
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        初始化Focal Loss
        
        参数:
            alpha: 类别权重，形状为[num_classes]，None表示等权重
            gamma: 聚焦参数，大于0，默认为2，值越大对难分类样本关注越多
            reduction: 损失计算方式，'none', 'mean', 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 类别权重，形状为[num_classes]
        self.gamma = gamma  # 聚焦参数
        self.reduction = reduction  # 损失计算方式
        
    def forward(self, inputs, targets):
        """
        前向传播计算损失
        
        参数:
            inputs: 模型输出的logits，形状为 [N, C]
            targets: 真实标签，形状为 [N]
            
        返回:
            loss: 计算的损失值
        """
        # 检查输入维度，如果inputs是[N, C]而targets是[N]，则进行one-hot编码
        if len(inputs.shape) != len(targets.shape) and inputs.size(0) == targets.size(0):
            if len(inputs.shape) == 2:  # 如果是分类问题 [batch_size, num_classes]
                # 使用交叉熵损失，适合分类问题
                ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
                pt = torch.exp(-ce_loss)
                focal_weight = (1 - pt) ** self.gamma
                loss = focal_weight * ce_loss
            elif len(inputs.shape) == 1 or (len(inputs.shape) == 2 and inputs.size(1) == 1):  
                # 如果是回归问题 [batch_size] 或 [batch_size, 1]
                # 确保维度匹配
                inputs = inputs.view(-1)
                targets = targets.view(-1)
                # 使用MSE损失，适合回归问题
                mse_loss = nn.MSELoss(reduction='none')(inputs, targets)
                pt = torch.exp(-mse_loss)
                focal_weight = (1 - pt) ** self.gamma
                loss = focal_weight * mse_loss
        else:
            # 如果维度已经匹配，直接计算BCE损失
            bce_loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=self.alpha)(inputs, targets)
            probs = torch.sigmoid(inputs)
            p_t = probs * targets + (1 - probs) * (1 - targets)
            focal_weight = (1 - p_t) ** self.gamma
            loss = focal_weight * bce_loss
        
        # 根据reduction方式处理损失
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

def save_checkpoint(model, optimizer, epoch, save_path):
    """
    保存模型检查点，用于恢复训练或后续使用
    
    参数:
        model: 模型实例
        optimizer: 优化器实例
        epoch: 当前训练轮次
        save_path: 保存路径，建议使用.pth后缀
    """
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存模型和优化器状态
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)
    
    print(f"模型已保存到 {save_path}")

def load_checkpoint(model, optimizer, load_path):
    """
    加载模型检查点，恢复训练状态
    
    参数:
        model: 模型实例
        optimizer: 优化器实例
        load_path: 加载路径
        
    返回:
        int: 已训练的轮次
    """
    # 检查检查点是否存在
    if not os.path.exists(load_path):
        print(f"检查点 {load_path} 不存在")
        return 0
    
    # 加载检查点
    checkpoint = torch.load(load_path)
    
    # 恢复模型和优化器状态
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 获取已训练的轮次
    epoch = checkpoint['epoch']
    print(f"从轮次 {epoch} 加载模型")
    
    return epoch

def find_best_threshold(y_true, y_pred_probs, thresholds=None):
    """
    搜索最佳分类阈值，使F1分数最大化
    用于二分类或多标签分类场景
    
    参数:
        y_true: 真实标签 [N, C]
        y_pred_probs: sigmoid后的预测概率 [N, C]
        thresholds: 待评估的阈值列表，默认在0.1-0.9之间搜索
        
    返回:
        best_threshold: 最佳阈值
        best_f1: 最佳F1值
        threshold_results: 不同阈值的评估结果字典列表
    """
    # 设置默认阈值范围
    if thresholds is None:
        thresholds = np.arange(0.1, 0.91, 0.1)
    
    best_f1 = 0
    best_threshold = 0.5  # 默认阈值
    threshold_results = []
    
    # 转换为numpy数组进行计算
    y_true_np = y_true.cpu().numpy()
    y_pred_probs_np = y_pred_probs.cpu().numpy()
    
    # 尝试不同阈值
    for threshold in thresholds:
        # 应用阈值获得二分类结果
        y_pred_binary = (y_pred_probs_np > threshold).astype(np.float32)
        
        # 计算评估指标
        f1 = f1_score(y_true_np, y_pred_binary, average='macro', zero_division=0)
        precision = precision_score(y_true_np, y_pred_binary, average='macro', zero_division=0)
        recall = recall_score(y_true_np, y_pred_binary, average='macro', zero_division=0)
        
        # 保存当前阈值的结果
        threshold_results.append({
            'threshold': threshold,
            'f1': f1,
            'precision': precision,
            'recall': recall
        })
        
        # 更新最佳阈值
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1, threshold_results

def calculate_class_weights(y_true):
    """
    根据标签频率计算类别权重，用于处理类别不平衡
    
    参数:
        y_true: 真实标签 [N, C] 或 [N]
        
    返回:
        class_weights: 类别权重，形状为 [C]，反比于类别频率
    """
    # 计算每个类别的正样本数量
    positive_counts = y_true.sum(axis=0)
    total_samples = len(y_true)
    
    # 避免除零错误
    positive_counts = np.maximum(positive_counts, 1)
    
    # 计算类别权重：反比于频率
    # 对于稀有类别，给予更高的权重
    class_weights = total_samples / (positive_counts * len(positive_counts))
    
    return class_weights

def calculate_metrics(y_true, y_pred, threshold=0.3, search_threshold=True):
    """
    计算多标签分类指标，全面评估模型性能
    
    参数:
        y_true: 真实标签 [N, C]
        y_pred: 预测分数（logits） [N, C]
        threshold: 二分类阈值，降低至0.3以更容易预测正样本
        search_threshold: 是否搜索最佳阈值
        
    返回:
        dict: 包含各种性能指标的字典
    """
    # 对预测分数进行sigmoid激活，确保范围在[0,1]之间
    y_pred_probs = torch.sigmoid(y_pred).cpu().numpy()
    y_true_np = y_true.cpu().numpy()
    
    # 计算类别权重，用于后续训练优化
    class_weights = calculate_class_weights(y_true_np)
    
    if search_threshold:
        # 搜索最佳阈值
        best_threshold, _, threshold_results = find_best_threshold(
            y_true, torch.sigmoid(y_pred),
            thresholds=np.arange(0.1, 0.91, 0.1)
        )
        # 使用最佳阈值进行预测
        y_pred_binary = (y_pred_probs > best_threshold).astype(np.float32)
        used_threshold = best_threshold
    else:
        # 使用固定阈值
        y_pred_binary = (y_pred_probs > threshold).astype(np.float32)
        used_threshold = threshold
    
    # 计算各种指标
    f1_macro = f1_score(y_true_np, y_pred_binary, average='macro', zero_division=0)
    precision_macro = precision_score(y_true_np, y_pred_binary, average='macro', zero_division=0)
    recall_macro = recall_score(y_true_np, y_pred_binary, average='macro', zero_division=0)
    
    # 样本级别的F1
    f1_samples = f1_score(y_true_np, y_pred_binary, average='samples', zero_division=0)
    
    # 每个类别的F1
    f1_per_class = f1_score(y_true_np, y_pred_binary, average=None, zero_division=0)
    
    # 计算混淆矩阵
    # cm = confusion_matrix(y_true_np.flatten(), y_pred_binary.flatten())
    
    # 返回包含所有指标的字典
    return {
        'f1_macro': f1_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_samples': f1_samples,
        'f1_per_class': f1_per_class,
        'threshold': used_threshold,
        'class_weights': class_weights
    }

def plot_metrics(train_metrics, val_metrics, save_path=None):
    """
    绘制训练过程中的指标变化曲线
    
    参数:
        train_metrics: 包含训练集指标的字典列表，每个元素对应一个epoch
        val_metrics: 包含验证集指标的字典列表，每个元素对应一个epoch
        save_path: 图像保存路径
    """
    epochs = range(1, len(train_metrics) + 1)
    
    plt.figure(figsize=(15, 10))
    
    # 绘制损失
    plt.subplot(2, 2, 1)
    plt.plot(epochs, [m['loss'] for m in train_metrics], 'b-', label='训练损失')
    plt.plot(epochs, [m['loss'] for m in val_metrics], 'r-', label='验证损失')
    plt.title('损失变化')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制准确率
    plt.subplot(2, 2, 2)
    plt.plot(epochs, [m['accuracy'] for m in train_metrics], 'b-', label='训练准确率')
    plt.plot(epochs, [m['accuracy'] for m in val_metrics], 'r-', label='验证准确率')
    plt.title('准确率变化')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # 绘制F1
    plt.subplot(2, 2, 3)
    plt.plot(epochs, [m['f1_macro'] for m in train_metrics], 'b-', label='训练F1')
    plt.plot(epochs, [m['f1_macro'] for m in val_metrics], 'r-', label='验证F1')
    plt.title('F1分数变化')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    # 绘制精确率和召回率
    plt.subplot(2, 2, 4)
    plt.plot(epochs, [m['precision_macro'] for m in val_metrics], 'g-', label='验证精确率')
    plt.plot(epochs, [m['recall_macro'] for m in val_metrics], 'm-', label='验证召回率')
    plt.title('精确率与召回率变化')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"指标图表已保存到 {save_path}")
    
    plt.show()

def download_bigearthnet_mini(output_dir='bigearthnet'):
    """
    下载BigEarthNet数据集的迷你版本用于测试
    
    参数:
        output_dir: 保存目录
    
    返回:
        str: 数据集保存路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # BigEarthNet-Mini数据集的链接
    bigearthnet_url = 'https://drive.google.com/uc?id=1r3_gxQf-GRoFwQ-lRjfPZiYBbOYUiI6P'
    
    # 下载目标路径
    tar_file = os.path.join(output_dir, 'bigearthnet-mini.tar.gz')
    
    if not os.path.exists(tar_file):
        print("正在下载BigEarthNet-Mini数据集...")
        gdown.download(bigearthnet_url, tar_file, quiet=False)
    else:
        print(f"已存在下载文件: {tar_file}")
    
    # 解压文件
    extract_dir = os.path.join(output_dir, 'bigearthnet-mini')
    if not os.path.exists(extract_dir):
        print("正在解压数据集...")
        with tarfile.open(tar_file) as tar:
            tar.extractall(output_dir)
        print(f"数据集已解压到: {extract_dir}")
    else:
        print(f"已存在解压目录: {extract_dir}")
    
    return extract_dir

def generate_mock_data(directory, num_samples=100):
    """
    生成模拟数据，用于测试和开发
    
    参数:
        directory: 保存目录
        num_samples: 样本数量
    
    返回:
        str: 数据目录路径
    """
    # 创建目录结构
    os.makedirs(directory, exist_ok=True)
    os.makedirs(os.path.join(directory, 'images'), exist_ok=True)
    os.makedirs(os.path.join(directory, 'labels'), exist_ok=True)
    
    print(f"生成 {num_samples} 个模拟样本...")
    
    # 生成模拟图像和标签
    for i in range(num_samples):
        # 生成随机图像 (3通道，128x128)
        img = np.random.rand(128, 128, 3).astype(np.float32)
        
        # 添加一些随机形状
        cx, cy = np.random.randint(20, 108, 2)  # 形状中心
        r = np.random.randint(10, 30)  # 半径
        
        # 决定是否添加病斑
        has_disease = np.random.rand() > 0.5
        disease_class = np.random.randint(0, 3) if has_disease else -1  # -1表示无病害
        
        if has_disease:
            # 添加模拟病斑
            rr, cc = np.meshgrid(
                np.arange(max(0, cx-r), min(128, cx+r)),
                np.arange(max(0, cy-r), min(128, cy+r))
            )
            
            dist = np.sqrt((rr - cx)**2 + (cc - cy)**2)
            mask = dist <= r
            
            # 根据病害类型添加不同颜色
            if disease_class == 0:  # 锈病
                img[cc[mask], rr[mask], 0] += 0.5
                img[cc[mask], rr[mask], 1] -= 0.2
            elif disease_class == 1:  # 斑点病
                img[cc[mask], rr[mask], 1] += 0.5
                img[cc[mask], rr[mask], 2] -= 0.2
            else:  # 其他病害
                img[cc[mask], rr[mask], 2] += 0.5
                img[cc[mask], rr[mask], 0] -= 0.2
        
        # 裁剪到[0,1]范围
        img = np.clip(img, 0, 1)
        
        # 保存图像
        img_path = os.path.join(directory, 'images', f'sample_{i:04d}.npy')
        np.save(img_path, img)
        
        # 生成标签
        if has_disease:
            label = disease_class
            severity = np.random.randint(1, 6)  # 1-5的严重程度
        else:
            label = 0
            severity = 0
        
        # 保存标签
        label_path = os.path.join(directory, 'labels', f'sample_{i:04d}.txt')
        with open(label_path, 'w') as f:
            f.write(f"{label} {severity}")
    
    print(f"已生成模拟数据集在: {directory}")
    return directory

def visualize_attention(image, attention_map, save_path=None):
    """
    可视化注意力图，帮助理解模型决策
    
    参数:
        image: 原始图像, 形状为 [H, W, C]
        attention_map: 注意力图, 形状为 [H, W]
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 5))
    
    # 显示原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('原始图像')
    plt.axis('off')
    
    # 显示注意力图
    plt.subplot(1, 3, 2)
    plt.imshow(attention_map, cmap='jet')
    plt.title('注意力图')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    # 显示叠加图
    plt.subplot(1, 3, 3)
    
    # 确保注意力图尺寸与原图一致
    if attention_map.shape[:2] != image.shape[:2]:
        from skimage.transform import resize
        attention_map = resize(attention_map, image.shape[:2], preserve_range=True)
    
    # 将注意力图转换为热力图
    attention_heatmap = plt.cm.jet(attention_map / np.max(attention_map))[:, :, :3]
    
    # 叠加
    alpha = 0.6  # 透明度
    overlaid = alpha * attention_heatmap + (1 - alpha) * image / np.max(image)
    overlaid = np.clip(overlaid, 0, 1)
    
    plt.imshow(overlaid)
    plt.title('叠加效果')
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"注意力可视化图已保存到 {save_path}")
    
    plt.show()

def count_parameters(model):
    """
    计算模型的参数数量，便于分析模型复杂度
    
    参数:
        model: 模型实例
        
    返回:
        int: 参数总数
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    return total_params, trainable_params