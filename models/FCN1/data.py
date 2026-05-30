def get_dataloaders(data_root, 
                  json_root=None, 
                  batch_size=8, 
                  num_workers=4, 
                  img_size=128, 
                  random_seed=42,
                  use_extended_dataset=False,
                  val_split=0.2,
                  pin_memory=True,
                  prefetch_factor=2):
    """
    创建数据加载器
    
    参数:
        data_root: 原始图像数据目录
        json_root: 标注数据目录
        batch_size: 批次大小
        num_workers: 数据加载线程数
        img_size: 图像大小
        random_seed: 随机种子
        use_extended_dataset: 是否使用增强的数据集
        val_split: 验证集比例
        pin_memory: 是否使用内存锁定提高GPU转移速度
        prefetch_factor: 预取因子，决定每个worker预取的样本数
    
    返回:
        train_loader, val_loader: 训练集和验证集的数据加载器
    """
    # 设置随机种子以确保可复现性
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # 定义数据变换
    transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # 创建数据集
    if use_extended_dataset and json_root is not None:
        dataset = DiseaseDatasetExtended(
            data_root=data_root,
            json_root=json_root,
            transform=transform
        )
    else:
        dataset = DiseaseDataset(
            data_root=data_root,
            json_root=json_root,
            transform=transform
        )
    
    # 划分训练集和验证集
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    # 为验证集单独设置转换
    if use_extended_dataset and json_root is not None:
        val_dataset = DiseaseDatasetExtended(
            data_root=data_root,
            json_root=json_root,
            transform=val_transform,
            indices=val_dataset.indices
        )
    else:
        val_dataset = DiseaseDataset(
            data_root=data_root,
            json_root=json_root,
            transform=val_transform,
            indices=val_dataset.indices
        )
    
    # 创建数据加载器
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'pin_memory': pin_memory,
        'num_workers': num_workers,
        'prefetch_factor': prefetch_factor if num_workers > 0 else None,
        'persistent_workers': num_workers > 0,  # 保持工作进程活跃以减少启动开销
    }
    
    train_loader = DataLoader(train_dataset, **loader_kwargs)
    
    # 验证集不需要洗牌
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        pin_memory=pin_memory, 
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    
    return train_loader, val_loader 