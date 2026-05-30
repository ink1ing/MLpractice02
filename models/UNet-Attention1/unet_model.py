# [核心文件] UNet+Attention模型定义文件：实现用于玉米锈病识别的UNet网络结构，包含通道注意力和空间注意力机制，用于图像分割、感染部位分类和感染等级回归的多任务学习
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """
    通道注意力机制
    捕捉通道之间的依赖关系，对重要的通道赋予更高的权重
    结合平均池化和最大池化的信息，提高特征表示能力
    """
    def __init__(self, in_channels, reduction_ratio=16):
        """
        初始化通道注意力模块
        
        参数:
            in_channels: 输入特征图的通道数
            reduction_ratio: 降维比例，用于减少参数量
        """
        super(ChannelAttention, self).__init__()
        # 全局平均池化 - 捕获通道的全局分布
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 输出1x1特征图
        # 全局最大池化 - 捕获通道的显著特征
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 通过两个1x1卷积实现全连接层，减少参数量
        self.fc = nn.Sequential(
            # 第一个1x1卷积，降维
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            # 第二个1x1卷积，恢复维度
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
        # sigmoid激活函数，将注意力权重归一化到[0,1]范围
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x: 输入特征图，形状为 [batch_size, in_channels, height, width]
            
        返回:
            attention: 通道注意力权重，形状为 [batch_size, in_channels, 1, 1]
        """
        # 平均池化分支
        avg_out = self.fc(self.avg_pool(x))
        # 最大池化分支
        max_out = self.fc(self.max_pool(x))
        # 融合两个分支的信息
        out = avg_out + max_out
        # 应用sigmoid归一化
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """
    空间注意力机制
    关注图像的空间位置重要性，对重要区域赋予更高权重
    结合通道平均值和最大值的信息，增强模型对空间区域的感知能力
    """
    def __init__(self, kernel_size=7):
        """
        初始化空间注意力模块
        
        参数:
            kernel_size: 卷积核大小，默认为7，用于捕获更大的感受野
        """
        super(SpatialAttention, self).__init__()
        # 使用单层卷积学习空间注意力图
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()  # 注意力权重归一化

    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x: 输入特征图，形状为 [batch_size, channels, height, width]
            
        返回:
            attention: 空间注意力权重，形状为 [batch_size, 1, height, width]
        """
        # 沿通道维度计算平均值 - 捕获全局通道信息
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 沿通道维度计算最大值 - 捕获显著特征
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 拼接通道平均值和最大值
        x = torch.cat([avg_out, max_out], dim=1)  # 形状为 [batch_size, 2, height, width]
        # 通过卷积生成空间注意力图
        x = self.conv(x)  # 输出单通道特征图
        # 应用sigmoid归一化
        return self.sigmoid(x)

class ConvBlock(nn.Module):
    """UNet的卷积块，包含两个卷积层，每个卷积层后跟BatchNorm和ReLU激活函数"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)

class AttentionBlock(nn.Module):
    """注意力块，结合通道注意力和空间注意力机制"""
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
        
    def forward(self, x):
        # 应用通道注意力
        x = self.ca(x) * x
        # 应用空间注意力
        x = self.sa(x) * x
        return x

class DownBlock(nn.Module):
    """下采样块，包含最大池化和卷积块"""
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)
        self.attention = AttentionBlock(out_channels)
        
    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.attention(x)
        return x

class UpBlock(nn.Module):
    """上采样块，包含上采样卷积和卷积块"""
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)
        self.attention = AttentionBlock(out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 处理可能的尺寸不匹配
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, 
                        diffY // 2, diffY - diffY // 2])
        
        # 拼接特征图
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.attention(x)
        return x

class UNetAttention(nn.Module):
    """
    UNet+Attention模型，用于玉米南方锈病多任务学习:
    1. 图像分割：病害区域分割
    2. 感染部位分类：下部/中部/上部
    3. 感染等级回归：0-9连续值
    """
    def __init__(self, in_channels=3, img_size=128):
        super(UNetAttention, self).__init__()
        self.in_channels = in_channels
        self.img_size = img_size
        
        # 编码器部分
        self.inc = ConvBlock(in_channels, 64)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        self.down4 = DownBlock(512, 1024)
        
        # 解码器部分
        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)
        
        # 分割输出头 - 输出1通道的分割掩码
        self.outc = nn.Conv2d(64, 1, kernel_size=1)
        
        # 全局特征提取
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 特征整合层 - 结合编码器特征
        self.fc_features = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # 位置分类头 - 预测感染部位(下部/中部/上部)
        self.position_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 3)
        )
        
        # 等级回归头 - 预测感染等级(0-9连续值)
        self.grade_regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        # 编码器路径
        x1 = self.inc(x)  # 特征图尺寸: [B, 64, H, W]
        x2 = self.down1(x1)  # 特征图尺寸: [B, 128, H/2, W/2]
        x3 = self.down2(x2)  # 特征图尺寸: [B, 256, H/4, W/4]
        x4 = self.down3(x3)  # 特征图尺寸: [B, 512, H/8, W/8]
        x5 = self.down4(x4)  # 特征图尺寸: [B, 1024, H/16, W/16]
        
        # 从最深层提取全局特征，用于分类和回归任务
        global_features = self.avg_pool(x5)  # [B, 1024, 1, 1]
        global_features = global_features.view(global_features.size(0), -1)  # [B, 1024]
        
        # 解码器路径
        x = self.up1(x5, x4)  # [B, 512, H/8, W/8]
        x = self.up2(x, x3)  # [B, 256, H/4, W/4]
        x = self.up3(x, x2)  # [B, 128, H/2, W/2]
        x = self.up4(x, x1)  # [B, 64, H, W]
        
        # 分割输出
        mask = self.outc(x)  # [B, 1, H, W]
        mask = torch.sigmoid(mask)  # 应用sigmoid得到0-1范围的分割掩码
        
        # 特征整合
        shared_features = self.fc_features(global_features)  # [B, 512]
        
        # 位置分类
        position_logits = self.position_classifier(shared_features)  # [B, 3]
        
        # 等级回归
        grade_values = self.grade_regressor(shared_features)  # [B, 1]
        
        return mask, position_logits, grade_values

def get_unet_model(in_channels=3, img_size=128):
    """
    获取UNet+Attention模型实例
    
    参数:
        in_channels: 输入图像通道数，默认为3
        img_size: 输入图像尺寸，默认为128x128
        
    返回:
        model: UNet+Attention模型实例
    """
    return UNetAttention(in_channels=in_channels, img_size=img_size) 