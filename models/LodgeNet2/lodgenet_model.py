#!/usr/bin/env python
# 核心文件：LodgeNet模型定义，实现玉米锈病识别的多任务学习架构
# LodgeNet模型定义文件：专门用于玉米锈病识别与预测的深度学习模型
# 实现图像分割和病害等级回归的多任务学习架构
# 基于U-Net和ResNet的混合架构，针对多光谱遥感图像优化

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DoubleConv(nn.Module):
    """
    双卷积块：LodgeNet的基础构建单元
    包含两个3x3卷积层，每个卷积层后跟BatchNorm和ReLU激活
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """
    下采样块：使用最大池化进行下采样，然后应用双卷积
    """
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """
    上采样块：使用转置卷积进行上采样，然后与跳跃连接特征拼接，最后应用双卷积
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # 处理输入尺寸不匹配的情况
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # 拼接跳跃连接
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class AttentionGate(nn.Module):
    """
    注意力门控机制：用于突出重要特征，抑制无关信息
    在跳跃连接中应用，提高分割精度
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 处理g和x的尺寸不匹配问题
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # 如果g1和x1的空间尺寸不同，将g1上采样到x1的尺寸
        if g1.size()[2:] != x1.size()[2:]:
            g1 = F.interpolate(g1, size=x1.size()[2:], mode='bilinear', align_corners=True)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # 确保psi和x的尺寸匹配
        if psi.size()[2:] != x.size()[2:]:
            psi = F.interpolate(psi, size=x.size()[2:], mode='bilinear', align_corners=True)
        
        return x * psi

class ASPP(nn.Module):
    """
    空洞空间金字塔池化（Atrous Spatial Pyramid Pooling）
    用于捕获多尺度上下文信息，提高分割性能
    """
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        
        # 不同膨胀率的空洞卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        
        # 全局平均池化
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 批归一化和激活
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.bn4 = nn.BatchNorm2d(out_channels)
        
        # 融合卷积
        self.conv_cat = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x)))
        x3 = F.relu(self.bn3(self.conv3(x)))
        x4 = F.relu(self.bn4(self.conv4(x)))
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv_cat(x)
        return x

class LodgeNet(nn.Module):
    """
    LodgeNet：专门用于玉米锈病识别的多任务学习网络
    
    功能：
    1. 图像分割：识别感染区域的精确位置
    2. 病害等级回归：预测感染严重程度（0-9连续值）
    3. 位置分类：预测感染部位（下部/中部/上部）
    
    架构特点：
    - 基于U-Net的编码器-解码器结构
    - 集成注意力门控机制
    - 使用ASPP模块捕获多尺度特征
    - 多任务学习头部设计
    """
    
    def __init__(self, n_channels=3, n_classes=2, img_size=128, bilinear=True):
        """
        初始化LodgeNet模型
        
        参数:
            n_channels: 输入图像通道数，默认3（RGB或选择的3个光谱通道）
            n_classes: 分割类别数，默认2（背景+感染区域）
            img_size: 输入图像尺寸，默认128x128
            bilinear: 是否使用双线性插值进行上采样，默认True
        """
        super(LodgeNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.img_size = img_size
        self.bilinear = bilinear
        
        # 编码器（下采样路径）
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # 瓶颈层：使用ASPP模块增强特征表示
        self.aspp = ASPP(1024 // factor, 1024 // factor)
        
        # 注意力门控 - 修正通道数配置
        self.att1 = AttentionGate(F_g=1024//factor, F_l=512, F_int=256)
        self.att2 = AttentionGate(F_g=512//factor, F_l=256, F_int=128)  # 修正：g来自up1的输出
        self.att3 = AttentionGate(F_g=256//factor, F_l=128, F_int=64)   # 修正：g来自up2的输出
        self.att4 = AttentionGate(F_g=128//factor, F_l=64, F_int=32)    # 修正：g来自up3的输出
        
        # 解码器（上采样路径）
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # 分割输出头
        self.segmentation_head = nn.Conv2d(64, n_classes, kernel_size=1)
        
        # 全局特征提取（用于分类和回归任务）
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 计算全连接层输入维度
        # 使用来自编码器不同层的特征进行全局任务
        self.fc_input_size = 64 + 128 + 256 + 512 + (1024 // factor)  # 多尺度特征融合
        
        # 位置分类头（3分类：下部/中部/上部）
        self.position_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.fc_input_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)  # 3个位置类别
        )
        
        # 病害等级回归头（回归任务：0-9连续值）
        self.grade_regressor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.fc_input_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)  # 1个回归输出
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        初始化网络权重
        使用He初始化方法初始化卷积层权重
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入图像张量，形状为 [batch_size, n_channels, height, width]
            
        返回:
            tuple: (segmentation_output, position_logits, grade_output)
                - segmentation_output: 分割结果，形状为 [batch_size, n_classes, height, width]
                - position_logits: 位置分类logits，形状为 [batch_size, 3]
                - grade_output: 病害等级回归输出，形状为 [batch_size, 1]
        """
        # 编码器路径
        x1 = self.inc(x)      # [B, 64, H, W]
        x2 = self.down1(x1)   # [B, 128, H/2, W/2]
        x3 = self.down2(x2)   # [B, 256, H/4, W/4]
        x4 = self.down3(x3)   # [B, 512, H/8, W/8]
        x5 = self.down4(x4)   # [B, 1024//factor, H/16, W/16]
        
        # 瓶颈层：应用ASPP增强特征
        x5 = self.aspp(x5)
        
        # 解码器路径（带注意力门控）
        # 应用注意力门控到跳跃连接
        x4_att = self.att1(g=x5, x=x4)
        x = self.up1(x5, x4_att)  # [B, 512//factor, H/8, W/8]
        
        x3_att = self.att2(g=x, x=x3)
        x = self.up2(x, x3_att)   # [B, 256//factor, H/4, W/4]
        
        x2_att = self.att3(g=x, x=x2)
        x = self.up3(x, x2_att)   # [B, 128//factor, H/2, W/2]
        
        x1_att = self.att4(g=x, x=x1)
        x = self.up4(x, x1_att)   # [B, 64, H, W]
        
        # 分割输出
        segmentation_output = self.segmentation_head(x)  # [B, n_classes, H, W]
        
        # 全局特征提取（用于分类和回归）
        # 从不同尺度提取全局特征并融合
        global_feat1 = self.global_pool(x1_att).flatten(1)    # [B, 64]
        global_feat2 = self.global_pool(x2_att).flatten(1)    # [B, 128]
        global_feat3 = self.global_pool(x3_att).flatten(1)    # [B, 256]
        global_feat4 = self.global_pool(x4_att).flatten(1)    # [B, 512]
        global_feat5 = self.global_pool(x5).flatten(1)        # [B, 1024//factor]
        
        # 多尺度特征融合
        global_features = torch.cat([
            global_feat1, global_feat2, global_feat3, 
            global_feat4, global_feat5
        ], dim=1)  # [B, fc_input_size]
        
        # 位置分类
        position_logits = self.position_classifier(global_features)  # [B, 3]
        
        # 病害等级回归
        grade_output = self.grade_regressor(global_features)  # [B, 1]
        
        return segmentation_output, position_logits, grade_output

def get_lodgenet_model(n_channels=3, n_classes=2, img_size=128, bilinear=True):
    """
    获取LodgeNet模型实例
    
    参数:
        n_channels: 输入通道数，默认3
        n_classes: 分割类别数，默认2（背景+感染区域）
        img_size: 图像尺寸，默认128
        bilinear: 是否使用双线性插值，默认True
        
    返回:
        LodgeNet: 模型实例
    """
    model = LodgeNet(
        n_channels=n_channels,
        n_classes=n_classes,
        img_size=img_size,
        bilinear=bilinear
    )
    return model

# 模型参数统计函数
def count_parameters(model):
    """
    统计模型参数数量
    
    参数:
        model: PyTorch模型
        
    返回:
        int: 可训练参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 测试函数
if __name__ == "__main__":
    # 创建模型实例
    model = get_lodgenet_model(n_channels=3, n_classes=2, img_size=128)
    
    # 打印模型信息
    print(f"LodgeNet模型参数数量: {count_parameters(model):,}")
    
    # 测试前向传播
    x = torch.randn(2, 3, 128, 128)  # 批次大小为2的测试输入
    seg_out, pos_out, grade_out = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"分割输出形状: {seg_out.shape}")
    print(f"位置分类输出形状: {pos_out.shape}")
    print(f"病害等级回归输出形状: {grade_out.shape}") 