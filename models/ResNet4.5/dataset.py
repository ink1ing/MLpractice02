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
import warnings
from rasterio.errors import NotGeoreferencedWarning

# 抑制rasterio的NotGeoreferencedWarning警告
warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)

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
        
        # 添加缓存机制提高性能
        self.cache = {}
        self.cache_size = min(100, len(self.samples))  # 最多缓存100个样本
        
        # 性能优化选项
        self.rasterio_threads = 1  # 限制rasterio使用的线程数，避免竞争
        
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
        # 检查缓存中是否有该样本
        if idx in self.cache:
            return self.cache[idx]
        
        try:
            # 获取路径对
            tif_path, json_path = self.samples[idx]
            
            # 解析标签
            position_label, grade_label = self._parse_json_label(json_path)
            
            try:
                # 读取.tif多光谱图像，使用环境变量限制线程数
                import os
                old_threads = os.environ.get('GDAL_NUM_THREADS', None)
                os.environ['GDAL_NUM_THREADS'] = str(self.rasterio_threads)
                
                with rasterio.open(tif_path) as src:
                    # 避免读取全部波段，只读取我们需要的波段
                    selected_bands = [0, 250, 499]  # 第一个、中间和最后一个波段
                    selected_image = np.zeros((3, self.img_size, self.img_size), dtype=np.float32)
                    
                    # 直接读取调整大小后的数据，减少内存使用
                    for i, band_idx in enumerate(selected_bands[:3]):
                        # 确保波段索引有效
                        band_idx = min(band_idx, src.count - 1)
                        # 直接读取并调整大小
                        band_data = src.read(band_idx + 1, out_shape=(self.img_size, self.img_size))
                        selected_image[i] = band_data
                    
                    # 标准化图像，保证像素值在[0, 1]范围内
                    selected_image = np.clip(selected_image / 255.0, 0, 1)
                    
                    # 转换为PyTorch张量
                    image = torch.from_numpy(selected_image).float()
                    
                    # 应用数据增强变换
                    if self.transform:
                        image = self.transform(image)
                    
                    # 恢复环境变量
                    if old_threads is not None:
                        os.environ['GDAL_NUM_THREADS'] = old_threads
                    else:
                        os.environ.pop('GDAL_NUM_THREADS', None)
                    
                    # 缓存结果
                    result = (image, position_label, grade_label)
                    if len(self.cache) < self.cache_size:
                        self.cache[idx] = result
                    
                    return result
                    
            except Exception as e:
                print(f"读取图像时出错: {tif_path}, 错误: {str(e)}")
                # 返回默认值 - 全零图像
                default_img = torch.zeros((3, self.img_size, self.img_size))
                result = (default_img, position_label, grade_label)
                return result
        except Exception as e:
            print(f"__getitem__错误(idx={idx}): {str(e)}")
            # 返回全默认值
            default_img = torch.zeros((3, self.img_size, self.img_size))
            return default_img, 1, 0

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
            
        # 旋转增强 - 两种旋转模式
        if random.random() < self.aug_prob:
            # 模式1: 标准旋转(90/180/270度) - 50%概率
            if random.random() < 0.5:
                angle = random.choice([90, 180, 270])
            # 模式2: 随机角度旋转(0-30度) - 50%概率
            else:
                angle = random.uniform(-30, 30)
            
            # 使用transforms_functional.rotate进行旋转
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
                    train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, aug_prob=0.5, use_extended_dataset=True, pin_memory=True, 
                    prefetch_factor=2):
    """
    创建训练集、验证集和测试集的DataLoader
    
    参数:
        data_root: 数据根目录路径，包含TIF图像
        json_root: JSON标注根目录路径
        batch_size: 批次大小
        num_workers: 数据加载线程数
        img_size: 图像大小
        train_ratio: 训练集比例，默认0.8
        val_ratio: 验证集比例，默认0.1
        test_ratio: 测试集比例，默认0.1
        aug_prob: 数据增强概率，默认0.5
        use_extended_dataset: 是否使用扩展数据集结构
        pin_memory: 是否使用固定内存，GPU训练时应设为True
        prefetch_factor: 预取队列大小因子，每个worker预取的样本数
        
    返回:
        tuple: (train_loader, val_loader, test_loader) 训练、验证和测试的数据加载器
    """
    print(f"\n=== DataLoader 配置信息 ===")
    print(f"批次大小: {batch_size}")
    print(f"加载线程数: {num_workers}")
    print(f"是否使用pin_memory: {pin_memory}")
    print(f"prefetch_factor: {prefetch_factor}")
    print(f"是否使用扩展数据集: {use_extended_dataset}")
    print(f"数据集分割比例: 训练={train_ratio}, 验证={val_ratio}, 测试={test_ratio}")
    
    # 检查分割比例和
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"数据集分割比例之和必须为1，当前为 {train_ratio + val_ratio + test_ratio}")
    
    # 性能优化提示
    if num_workers == 0:
        print("警告: 使用主线程加载数据(num_workers=0)，可能会导致训练速度变慢")
    elif num_workers > 0 and prefetch_factor < 2:
        print("警告: prefetch_factor应至少为2，否则性能可能会降低")
    
    # 检查是否需要设置环境变量以优化性能
    try:
        import os
        # 设置TIF/GDAL相关环境变量
        os.environ['GDAL_NUM_THREADS'] = str(max(1, num_workers // 2))
        # 禁用GDAL PAM以避免创建辅助文件
        os.environ['GDAL_PAM_ENABLED'] = 'NO'
        print(f"已设置GDAL环境变量以优化性能")
    except Exception as e:
        print(f"设置环境变量时出错: {e}")
    
    # GPU效率检查
    if torch.cuda.is_available():
        print(f"检测到GPU: {torch.cuda.get_device_name(0)}")
        if num_workers < 2:
            print(f"警告: 使用GPU时，建议num_workers至少为2，当前为{num_workers}")
        if not pin_memory:
            print("警告: 使用GPU时，建议将pin_memory设置为True以提高数据传输效率")
    
    # 创建数据集实例
    try:
        print(f"正在创建数据集，数据根目录: {data_root}")
        dataset = CornRustDataset(
            data_dir=data_root,
            json_dir=json_root,
            img_size=img_size,
            use_extended_dataset=use_extended_dataset
        )
        print(f"数据集创建成功，样本数量: {len(dataset)}")
    except Exception as e:
        print(f"创建数据集时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    # 创建数据增强变换
    data_aug = DataAugmentation(aug_prob=aug_prob)
    print(f"已创建数据增强，概率: {aug_prob}")
    
    # 计算训练集、验证集和测试集大小
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    # 检查是否有足够的数据
    if train_size <= 0 or val_size <= 0 or test_size <= 0:
        raise ValueError(f"数据集太小无法分割：总样本数{len(dataset)}，分割比例 训练:{train_ratio} 验证:{val_ratio} 测试:{test_ratio}")
    
    # 使用固定种子进行数据集分割，确保可重复性
    import numpy as np
    generator = torch.Generator().manual_seed(42)
    
    # 随机分割数据集为训练集、验证集和测试集
    try:
        print("正在分割数据集为训练集、验证集和测试集...")
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size], generator=generator
        )
        print("数据集分割成功")
        
        # 为训练集添加数据增强
        train_dataset.dataset.transform = data_aug
        # 验证集和测试集不使用增强，确保评估稳定性
        val_dataset.dataset.transform = None
        test_dataset.dataset.transform = None
    except Exception as e:
        print(f"分割数据集时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    # 打印数据集信息
    print(f"数据集总大小: {len(dataset)}")
    print(f"训练集大小: {train_size}")
    print(f"验证集大小: {val_size}")
    print(f"测试集大小: {test_size}")
    
    # 创建数据加载器
    try:
        print("正在创建训练数据加载器...")
        
        # 计算最佳的worker数量
        optimal_workers = num_workers
        if num_workers > 0 and batch_size < 8:
            # 对于小批次，减少worker数量以避免开销过大
            optimal_workers = max(1, num_workers // 2)
            print(f"警告: 批次大小较小 ({batch_size})，自动调整worker数量为 {optimal_workers}")
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=optimal_workers,
            pin_memory=pin_memory,  # 使用固定内存提高GPU传输效率
            prefetch_factor=prefetch_factor if optimal_workers > 0 else None,  # 预取因子
            persistent_workers=optimal_workers > 0,  # 保持worker进程活跃
            drop_last=True,  # 丢弃最后不完整的批次，避免批归一化问题
            # 启用非阻塞内存固定，减少GPU同步等待时间
            pin_memory_device="cuda" if torch.cuda.is_available() and pin_memory else ""
        )
        print(f"训练数据加载器创建成功，批次数: {len(train_loader)}")
        
        print("正在创建验证数据加载器...")
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=optimal_workers,
            pin_memory=pin_memory,  # 使用固定内存提高GPU传输效率
            prefetch_factor=prefetch_factor if optimal_workers > 0 else None,  # 预取因子
            persistent_workers=optimal_workers > 0,  # 保持worker进程活跃
            pin_memory_device="cuda" if torch.cuda.is_available() and pin_memory else ""
        )
        print(f"验证数据加载器创建成功，批次数: {len(val_loader)}")
        
        print("正在创建测试数据加载器...")
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=optimal_workers,
            pin_memory=pin_memory,  # 使用固定内存提高GPU传输效率
            prefetch_factor=prefetch_factor if optimal_workers > 0 else None,  # 预取因子
            persistent_workers=optimal_workers > 0,  # 保持worker进程活跃
            pin_memory_device="cuda" if torch.cuda.is_available() and pin_memory else ""
        )
        print(f"测试数据加载器创建成功，批次数: {len(test_loader)}")
    except Exception as e:
        print(f"创建数据加载器时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    print("=== DataLoader配置完成 ===\n")
    return train_loader, val_loader, test_loader