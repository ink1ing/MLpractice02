# 数据集处理文件：负责读取和处理玉米南方锈病的多光谱图像数据，包括TIF图像加载、JSON标注解析、数据增强以及多任务学习的标签预处理（感染部位分类和感染等级回归）
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import rasterio
import json
from torchvision.transforms import functional as transforms_functional
import random
from collections import Counter
import glob
from skimage.transform import resize as sk_resize  # 导入scikit-image的resize函数

class CornRustDataset(Dataset):
    """
    玉米南方锈病数据集加载类
    处理.tif多光谱图像和.json标注文件
    支持多任务学习:
    1. 感染部位: 上部/中部/下部 -> 0/1/2 (3分类)
    2. 感染等级: 无/轻度/中度/重度/极重度 -> 0/3/5/7/9 -> 0-9 (回归任务)
    """
    def __init__(self, data_dir, json_dir=None, transform=None, img_size=128, use_extended_dataset=True):
        """
        初始化玉米南方锈病数据集
        
        参数:
            data_dir (str): .tif文件的数据集目录，包含多光谱图像
            json_dir (str, optional): .json标注文件目录，如果为None，则使用data_dir + '_json'
            transform (callable, optional): 数据增强和转换函数
            img_size (int, optional): 图像统一调整大小，默认128x128
            use_extended_dataset (bool, optional): 是否使用扩展数据集(9l/m/t, 14l/m/t, 19l/m/t)，默认True
        """
        self.data_dir = data_dir
        self.json_dir = json_dir if json_dir else data_dir + '_json'
        self.transform = transform
        self.img_size = img_size
        self.use_extended_dataset = use_extended_dataset
        
        # 映射字典 - 将文本标签映射为数值标签
        # 位置标签：l(下部)=0, m(中部)=1, t(上部)=2
        self.position_map = {"l": 0, "m": 1, "t": 2}  # 下部/中部/上部
        
        # 等级标签：以前是将0/3/5/7/9映射为0/1/2/3/4，现在直接使用原始值进行回归
        # 保留此映射用于向后兼容和统计
        self.grade_map = {0: 0, 3: 1, 5: 2, 7: 3, 9: 4}  # 无/轻度/中度/重度/极重度
        
        # 获取所有样本文件路径对
        self.samples = self._get_samples()
        
        # 缓存标签分布以计算类别权重 - 用于处理数据不平衡
        self.position_labels = []
        self.grade_labels = []
        self._cache_labels()
        
    def _get_samples(self):
        """
        获取所有样本路径和对应json文件路径
        
        返回:
            list: 包含(tif_path, json_path)元组的列表，每个元组对应一个样本
        """
        samples = []
        
        # 检查目录是否存在
        if not os.path.exists(self.data_dir):
            print(f"数据目录不存在: {self.data_dir}")
            return samples
        
        # 如果启用扩展数据集，扫描9l/m/t、14l/m/t和19l/m/t目录
        if self.use_extended_dataset:
            # 查找所有叶片目录
            leaf_patterns = ['9*', '14*', '19*']
            leaf_dirs = []
            
            for pattern in leaf_patterns:
                # 在主数据目录中寻找与pattern匹配的目录
                pattern_path = os.path.join(self.data_dir, pattern)
                matching_dirs = glob.glob(pattern_path)
                leaf_dirs.extend(matching_dirs)
            
            # 如果没有找到匹配的目录，尝试在父目录中寻找
            if not leaf_dirs and os.path.exists(os.path.dirname(self.data_dir)):
                parent_dir = os.path.dirname(self.data_dir)
                for pattern in leaf_patterns:
                    pattern_path = os.path.join(parent_dir, pattern)
                    matching_dirs = glob.glob(pattern_path)
                    leaf_dirs.extend(matching_dirs)
            
            # 如果依然没有找到，使用当前目录作为唯一目录
            if not leaf_dirs:
                leaf_dirs = [self.data_dir]
                print(f"警告: 未找到扩展数据集目录，仅使用当前目录: {self.data_dir}")
            
            # 处理找到的叶片目录
            for leaf_dir in leaf_dirs:
                # 确定对应的JSON目录
                dir_name = os.path.basename(leaf_dir)
                
                # 构建JSON子目录路径 - 修改这里以匹配标注结构
                # 例如: 9l -> 9l_json
                json_subdir = dir_name + '_json'
                json_dir = os.path.join(self.json_dir, json_subdir)
                
                # 查找TIF文件并配对JSON文件
                tif_files = [f for f in os.listdir(leaf_dir) if f.endswith('.tif')]
                
                # 检查JSON目录是否存在
                if not os.path.exists(json_dir):
                    print(f"警告: JSON目录不存在: {json_dir}")
                    continue
                
                # 添加配对的样本
                for tif_file in tif_files:
                    tif_path = os.path.join(leaf_dir, tif_file)
                    json_file = tif_file.replace('.tif', '.json')
                    json_path = os.path.join(json_dir, json_file)
                    
                    if os.path.exists(json_path):
                        samples.append((tif_path, json_path))
                    else:
                        print(f"警告: 找不到对应的JSON文件: {json_path}")
                
                print(f"从目录 {leaf_dir} 加载了 {len(tif_files)} 个样本")
        else:
            # 原始逻辑，仅查找数据目录中的.tif文件
            tif_files = [f for f in os.listdir(self.data_dir) if f.endswith('.tif')]
            
            # 遍历tif文件，找到对应的json文件
            for tif_file in tif_files:
                tif_path = os.path.join(self.data_dir, tif_file)
                # 找到对应的json文件
                json_file = tif_file.replace('.tif', '.json')
                json_path = os.path.join(self.json_dir, json_file)
                
                # 检查json文件是否存在
                if os.path.exists(json_path):
                    samples.append((tif_path, json_path))
                else:
                    print(f"警告: 找不到对应的json文件: {json_path}")
                    
        print(f"总共加载了 {len(samples)} 个样本")
        return samples
    
    def _cache_labels(self):
        """
        缓存所有样本的标签，用于计算类别权重和统计分布
        在初始化时调用一次，避免重复解析标签
        """
        self.position_labels = []
        self.grade_labels = []
        
        # 遍历所有样本解析标签
        for _, json_path in self.samples:
            position, grade = self._parse_json_label(json_path)
            self.position_labels.append(position)
            self.grade_labels.append(grade)
    
    def get_class_weights(self):
        """
        计算位置和等级分类的类别权重，用于处理类别不平衡问题
        反比于频率的权重，稀有类得到更高权重
        
        返回:
            tuple: (position_weights, grade_weights)
                - position_weights: 位置类别权重，形状为 [3]
                - grade_weights: 等级类别权重，形状为 [5] (用于向后兼容)
        """
        # 计算位置标签分布 - 使用Counter统计每个类别的样本数
        position_counter = Counter(self.position_labels)
        total_position = len(self.position_labels)
        position_weights = []
        
        # 为每个位置类别计算权重 (3个类别)
        for i in range(3):  # 下部/中部/上部 (0/1/2)
            count = position_counter.get(i, 0)
            # 避免除零错误
            if count == 0:
                position_weights.append(1.0)  # 如果没有样本，设置默认权重
            else:
                # 反比于频率的权重 - 频率越低权重越高
                # 乘以类别数，使权重平均值接近1
                position_weights.append(total_position / (count * 3))
        
        # 计算等级标签分布 (5个类别，用于向后兼容)
        grade_counter = Counter(self.grade_labels)
        total_grade = len(self.grade_labels)
        grade_weights = []
        
        for i in range(5):  # 无/轻度/中度/重度/极重度 (0/1/2/3/4)
            count = grade_counter.get(i, 0)
            # 避免除零错误
            if count == 0:
                grade_weights.append(1.0)
            else:
                # 反比于频率的权重
                grade_weights.append(total_grade / (count * 5))
        
        return position_weights, grade_weights
    
    def __len__(self):
        """
        返回数据集中样本数量
        
        返回:
            int: 样本数量
        """
        return len(self.samples)
    
    def _parse_json_label(self, json_path):
        """
        解析JSON标注文件，提取感染部位和感染等级信息
        
        参数:
            json_path (str): JSON标注文件路径
            
        返回:
            tuple: (position_label, grade_label) 
                - position_label: 感染部位的数值标签 (0-2)
                - grade_label: 感染等级的数值标签 (0-4为分类标签，但实际使用0-9的原始值进行回归)
        """
        try:
            # 读取JSON文件
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # 从文件路径提取位置信息(l/m/t)
            # 通过检测文件路径中的位置标识符来确定部位
            file_path = os.path.normpath(json_path).lower()
            path_components = file_path.split(os.sep)
            
            # 默认为中部
            position = 'm'
            
            # 检查文件路径中的位置标识符
            for component in path_components:
                # 检查目录或文件名是否包含位置标识
                if '_l' in component or component.endswith('l') or '/l/' in component or '\\l\\' in component or '9l' in component or '14l' in component or '19l' in component:
                    position = 'l'  # 下部
                    break
                elif '_m' in component or component.endswith('m') or '/m/' in component or '\\m\\' in component or '9m' in component or '14m' in component or '19m' in component:
                    position = 'm'  # 中部
                    break
                elif '_t' in component or component.endswith('t') or '/t/' in component or '\\t\\' in component or '9t' in component or '14t' in component or '19t' in component:
                    position = 't'  # 上部
                    break
            
            # 从JSON标注中提取疾病等级，默认为无疾病(0)
            grade = 0  
            
            # 解析JSON数据，查找是否存在健康标签
            is_healthy = True
            if 'shapes' in data:
                for shape in data['shapes']:
                    if 'label' in shape and shape['label'] != 'health':
                        # 发现非健康标签，说明存在疾病
                        is_healthy = False
                        # 尝试从标签中提取等级信息
                        # 假设标签格式可能包含等级信息，如 "disease_5"
                        label = shape['label']
                        if '_' in label:
                            try:
                                # 尝试提取数字部分作为等级
                                grade_str = label.split('_')[-1]
                                if grade_str.isdigit():
                                    grade = int(grade_str)
                                    # 检查是否为有效等级
                                    if grade in self.grade_map:
                                        break
                                    else:
                                        grade = 5  # 默认为中度
                            except:
                                grade = 5  # 解析失败，设置默认中度
                        else:
                            grade = 5  # 如果没有具体等级，默认为中度
            
            # 如果是健康样本，设置为0级
            if is_healthy:
                grade = 0
            elif grade not in self.grade_map and grade != 0:
                grade = 5  # 默认为中度
            
            # 转换为模型需要的标签
            position_label = self.position_map[position]  # 将文本位置转为数值
            
            # 现在直接返回原始等级值 (0-9) 用于回归任务
            # 但也保留分类标签，用于向后兼容和统计
            grade_label = self.grade_map.get(grade, 2)  # 默认为中度(2)
            
            return position_label, grade_label
            
        except Exception as e:
            print(f"解析JSON标签时出错: {e}")
            # 默认为中部(1)和无疾病(0)
            return 1, 0
    
    def __getitem__(self, idx):
        """
        获取单个样本的数据和标签
        PyTorch数据集的核心方法
        
        参数:
            idx (int): 样本索引
            
        返回:
            tuple: (image, position_label, grade_label)
                - image: 图像张量 [C, H, W]
                - position_label: 感染部位标签 (0-2)
                - grade_label: 感染等级标签 (0-9，用于回归)
        """
        # 获取路径对
        tif_path, json_path = self.samples[idx]
        
        # 解析标签
        position_label, grade_label = self._parse_json_label(json_path)
        
        # 读取.tif多光谱图像
        try:
            with rasterio.open(tif_path) as src:
                # 读取所有波段
                full_image = src.read()
                
                # 多光谱图像处理 - 现在我们知道有500个波段
                # 从500个光谱波段中选择3个有代表性的波段
                # 选择光谱中不同位置的波段以获得更多信息
                selected_bands = [0, 250, 499]  # 第一个、中间和最后一个波段
                
                # 确保选择的波段索引有效
                valid_bands = [min(b, full_image.shape[0]-1) for b in selected_bands]
                
                # 直接选择3个波段，避免后续维度问题
                # 重要：先选择波段，再调整大小
                selected_image = np.zeros((3, full_image.shape[1], full_image.shape[2]), dtype=full_image.dtype)
                for i, band_idx in enumerate(valid_bands[:3]):
                    selected_image[i] = full_image[band_idx]
                
                # 如果选择的波段少于3个，复制最后一个波段
                if len(valid_bands) < 3:
                    for i in range(len(valid_bands), 3):
                        selected_image[i] = selected_image[len(valid_bands)-1]
                
                # 现在selected_image的形状是[3, H, W]
                
                # 调整图像大小到模型要求的尺寸
                if selected_image.shape[1] != self.img_size or selected_image.shape[2] != self.img_size:
                    # 使用scikit-image的resize函数
                    # 注意：需要调整维度顺序，因为scikit-image期望[H, W, C]格式
                    selected_image = np.transpose(selected_image, (1, 2, 0))  # 转为[H, W, C]
                    resized_image = sk_resize(selected_image, (self.img_size, self.img_size), 
                                            anti_aliasing=True, preserve_range=True)
                    selected_image = np.transpose(resized_image, (2, 0, 1))  # 转回[C, H, W]
                    selected_image = selected_image.astype(np.float32)  # 确保类型正确
                
                # 标准化图像，保证像素值在[0, 1]范围内
                selected_image = np.clip(selected_image, 0, 1)
                
                # 转换为PyTorch张量
                image = torch.from_numpy(selected_image).float()
                
                # 应用数据增强变换
                if self.transform:
                    image = self.transform(image)
                
                # 确保最终输出的图像形状正确
                assert image.shape == (3, self.img_size, self.img_size), \
                    f"图像形状错误: {image.shape}, 期望: (3, {self.img_size}, {self.img_size})"
                
                return image, position_label, grade_label
                
        except Exception as e:
            print(f"读取图像时出错: {tif_path}, 错误: {e}")
            # 返回默认值 - 全零图像
            default_img = torch.zeros((3, self.img_size, self.img_size))
            return default_img, position_label, grade_label

class DataAugmentation:
    """
    数据增强类，对图像进行随机变换以增加样本多样性
    包括翻转、旋转、颜色抖动等方法
    """
    def __init__(self, aug_prob=0.5):
        """
        初始化数据增强类
        
        参数:
            aug_prob: 每种增强方法的应用概率，默认0.5
        """
        self.aug_prob = aug_prob
        
    def __call__(self, img):
        """
        对图像应用随机增强
        
        参数:
            img: 输入图像张量，形状为[C, H, W]
            
        返回:
            img: 增强后的图像张量，形状为[C, H, W]
        """
        # 随机水平翻转
        if random.random() < self.aug_prob:
            img = transforms_functional.hflip(img)
        
        # 随机垂直翻转
        if random.random() < self.aug_prob:
            img = transforms_functional.vflip(img)
        
        # 随机旋转(90/180/270度)
        if random.random() < self.aug_prob:
            angle = random.choice([90, 180, 270])
            img = transforms_functional.rotate(img, angle)
        
        # 随机亮度/对比度变化
        if random.random() < self.aug_prob:
            brightness_factor = random.uniform(0.8, 1.2)
            img = transforms_functional.adjust_brightness(img, brightness_factor)
        
        if random.random() < self.aug_prob:
            contrast_factor = random.uniform(0.8, 1.2)
            img = transforms_functional.adjust_contrast(img, contrast_factor)
        
        # 随机混合通道
        if random.random() < self.aug_prob and img.size(0) >= 3:
            # 随机排列通道顺序
            channel_indices = torch.randperm(img.size(0))
            img = img[channel_indices]
        
        return img

def get_dataloaders(data_root, json_root=None, batch_size=32, num_workers=4, img_size=128, 
                    train_ratio=0.8, aug_prob=0.5, use_extended_dataset=True):
    """
    创建训练集和验证集的数据加载器
    
    参数:
        data_root: 数据根目录
        json_root: JSON标注根目录，默认为None（自动推断）
        batch_size: 批次大小
        num_workers: 数据加载线程数
        img_size: 图像大小
        train_ratio: 训练集比例
        aug_prob: 数据增强应用概率
        use_extended_dataset: 是否使用扩展数据集
        
    返回:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 定义数据增强
    data_transform = DataAugmentation(aug_prob=aug_prob)
    
    # 创建完整数据集
    full_dataset = CornRustDataset(
        data_dir=data_root,
        json_dir=json_root,
        transform=data_transform,
        img_size=img_size,
        use_extended_dataset=use_extended_dataset
    )
    
    # 计算训练集和验证集大小
    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size
    
    # 拆分数据集
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # 打印数据集大小
    print(f"总数据集大小: {dataset_size}")
    print(f"训练集大小: {train_size}")
    print(f"验证集大小: {val_size}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader