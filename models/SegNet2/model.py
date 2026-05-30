# 核心文件，完整运行项目不可缺少
# 模型定义文件：包含玉米南方锈病识别的各种神经网络模型定义，包括简单CNN、ResNet和带注意力机制的增强ResNet，实现多任务学习（同时预测感染部位和感染等级）
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiseaseClassifier(nn.Module):
    """
    双头CNN模型，用于玉米南方锈病多任务分类：
    1. 感染部位: 下部/中部/上部 (3分类)
    2. 感染等级: 无/轻度/中度/重度/极重度 (5分类)
    """
    def __init__(self, in_channels=3, img_size=128):
        """
        初始化双头CNN分类器
        
        参数:
            in_channels: 输入图像的通道数，默认为3（RGB）
            img_size: 输入图像的尺寸，默认为128x128
        """
        super(DiseaseClassifier, self).__init__()
        self.in_channels = in_channels
        
        # 共享特征提取层 - 使用三层卷积网络提取图像特征
        self.features = nn.Sequential(
            # 第一个卷积块 - 输入通道 -> 32通道，使用3x3卷积核和2x2最大池化
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # 保持空间维度不变
            nn.BatchNorm2d(32),  # 批归一化加速训练并提高稳定性
            nn.ReLU(inplace=True),  # 使用ReLU激活函数引入非线性
            nn.MaxPool2d(kernel_size=2, stride=2),  # 下采样，减小特征图尺寸
            
            # 第二个卷积块 - 32通道 -> 64通道
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三个卷积块 - 64通道 -> 128通道
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 计算卷积后的特征图大小 - 经过三次2x2最大池化，尺寸变为原来的1/8
        conv_output_size = img_size // 8  # 三次下采样 (2^3=8)
        self.fc_input_size = 128 * conv_output_size * conv_output_size
        
        # 位置分类头 (3分类) - 将特征向量映射到3个类别
        self.position_classifier = nn.Sequential(
            nn.Flatten(),  # 将特征图展平为一维向量
            nn.Dropout(0.5),  # 防止过拟合
            nn.Linear(self.fc_input_size, 256),  # 全连接层降维
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # 第二个Dropout层进一步防止过拟合
            nn.Linear(256, 3)  # 输出3个类别的logits: 下部/中部/上部
        )
        
        # 等级分类头 (改为回归任务) - 将特征向量映射到1个输出值（回归）
        self.grade_classifier = nn.Sequential(
            nn.Flatten(),  # 将特征图展平为一维向量
            nn.Dropout(0.5),  # 防止过拟合
            nn.Linear(self.fc_input_size, 256),  # 全连接层降维
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)  # 输出1个值，用于回归预测感染等级
        )
    
    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x: 输入图像张量, 形状为 [batch_size, in_channels, height, width]
            
        返回:
            tuple: (position_logits, grade_logits)
                - position_logits: 位置分类的logits，形状为 [batch_size, 3]
                - grade_logits: 等级分类的logits，形状为 [batch_size, 5]
        """
        # 共享特征提取 - 对输入图像提取特征
        features = self.features(x)
        
        # 位置分类 - 通过位置分类头预测感染部位
        position_logits = self.position_classifier(features)
        
        # 等级分类 - 通过等级分类头预测感染等级
        grade_logits = self.grade_classifier(features)
        
        return position_logits, grade_logits

class ResidualBlock(nn.Module):
    """
    ResNet的基本残差块
    残差连接允许梯度直接流过网络，缓解深度网络的梯度消失问题
    """
    def __init__(self, in_channels, out_channels, stride=1):
        """
        初始化残差块
        
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            stride: 第一个卷积层的步长，用于下采样，默认为1
        """
        super(ResidualBlock, self).__init__()
        # 第一个卷积层，可能用于下采样（当stride > 1时）
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 第二个卷积层，保持空间维度不变
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 如果输入输出通道数不同，需要调整残差连接 - 使用1x1卷积进行通道调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x: 输入特征图，形状为 [batch_size, in_channels, height, width]
            
        返回:
            out: 残差块输出，形状为 [batch_size, out_channels, height/stride, width/stride]
        """
        residual = x  # 保存输入作为残差连接
        
        # 第一个卷积块
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # 第二个卷积块
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 添加残差连接，实现跳跃连接
        out += self.shortcut(residual)
        out = F.relu(out)  # 最后的非线性激活
        
        return out

class DiseaseResNet(nn.Module):
    """
    基于ResNet结构的双头模型，用于玉米南方锈病多任务分类
    使用残差连接和更深的网络结构增强特征提取能力
    """
    def __init__(self, in_channels=3, img_size=128):
        """
        初始化疾病ResNet模型
        
        参数:
            in_channels: 输入图像的通道数，默认为3
            img_size: 输入图像的尺寸，默认为128x128
        """
        super(DiseaseResNet, self).__init__()
        self.in_channels = in_channels
        
        # 初始卷积层 - 7x7大卷积核提取低级特征
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 下采样
        
        # ResNet块 - 构建三层残差网络
        self.layer1 = self._make_layer(64, 64, 2)  # 第一层：64->64通道，2个残差块
        self.layer2 = self._make_layer(64, 128, 2, stride=2)  # 第二层：64->128通道，2个残差块，下采样
        self.layer3 = self._make_layer(128, 256, 2, stride=2)  # 第三层：128->256通道，2个残差块，下采样
        
        # 计算卷积后的特征图大小
        # 图像尺寸经过初始卷积和MaxPool后变为 img_size/4
        # 再经过三个ResNet layer (其中两层有stride=2)，变为 img_size/16
        conv_output_size = img_size // 16
        self.fc_input_size = 256 * conv_output_size * conv_output_size
        
        # 全局平均池化 - 降低参数量并保留空间特征
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 输出1x1特征图
        
        # 位置分类头 - 预测感染部位
        self.position_classifier = nn.Sequential(
            nn.Flatten(),  # 将特征图展平
            nn.Dropout(0.5),  # 减少过拟合
            nn.Linear(256, 3)  # 全连接层输出3个类别
        )
        
        # 等级分类头 - 预测感染等级（回归任务）
        self.grade_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)  # 输出1个值进行回归
        )
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """
        创建残差层，包含多个残差块
        
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            blocks: 残差块数量
            stride: 第一个残差块的步长，用于下采样
            
        返回:
            nn.Sequential: 包含多个残差块的顺序容器
        """
        layers = []
        # 第一个block可能需要调整维度
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        # 剩余blocks保持维度不变
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x: 输入图像张量, 形状为 [batch_size, in_channels, height, width]
            
        返回:
            tuple: (position_logits, grade_logits)
                - position_logits: 位置分类的logits，形状为 [batch_size, 3]
                - grade_logits: 等级分类的logits，形状为 [batch_size, 5]
        """
        # 特征提取
        x = self.conv1(x)  # 初始卷积
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # 残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # 全局平均池化
        x = self.avgpool(x)  # 转为1x1特征图
        
        # 位置分类
        position_logits = self.position_classifier(x)
        
        # 等级分类
        grade_logits = self.grade_classifier(x)
        
        return position_logits, grade_logits

# 新增注意力机制模块
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

# 新增带注意力机制的残差块
class AttentionResidualBlock(nn.Module):
    """
    带注意力机制的残差块
    在基本残差块基础上增加了通道注意力和空间注意力机制
    结合CBAM(Convolutional Block Attention Module)思想，串联使用通道和空间注意力
    """
    def __init__(self, in_channels, out_channels, stride=1):
        """
        初始化带注意力机制的残差块
        
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            stride: 第一个卷积层的步长，默认为1
        """
        super(AttentionResidualBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 注意力模块 - 分别实现通道和空间注意力
        self.ca = ChannelAttention(out_channels)  # 通道注意力
        self.sa = SpatialAttention()  # 空间注意力
        
        # 如果输入输出通道数不同，需要调整残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x: 输入特征图，形状为 [batch_size, in_channels, height, width]
            
        返回:
            out: 残差块输出，形状为 [batch_size, out_channels, height/stride, width/stride]
        """
        residual = x  # 保存输入作为残差连接
        
        # 常规残差块前向传播
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 应用通道注意力 - 增强重要通道特征
        out = self.ca(out) * out
        # 应用空间注意力 - 突出重要空间区域
        out = self.sa(out) * out
        
        # 添加残差连接并激活
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out

# 新增改进的ResNet模型
class DiseaseResNetPlus(nn.Module):
    """
    增强版ResNet模型，增加注意力机制
    同时使用通道注意力和空间注意力提高特征提取能力
    针对玉米南方锈病的多任务分类问题
    """
    def __init__(self, in_channels=3, img_size=128):
        """
        初始化增强版ResNet模型
        
        参数:
            in_channels: 输入图像的通道数，默认为3
            img_size: 输入图像的尺寸，默认为128x128
        """
        super(DiseaseResNetPlus, self).__init__()
        self.in_channels = in_channels
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 增加Dropout防止过拟合
        self.dropout = nn.Dropout2d(0.1)
        
        # 使用带注意力的残差块构建ResNet
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # 计算卷积后特征图大小
        # 经过4次stride=2的下采样，尺寸变为原来的1/16
        conv_output_size = img_size // 16
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 特征整合层
        self.fc_features = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # 位置分类头
        self.position_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 3)
        )
        
        # 等级回归头 - 预测感染等级（回归值）
        self.grade_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 1)  # 输出1个值进行回归
        )
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """
        创建包含多个注意力残差块的层
        
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            blocks: 块的数量
            stride: 第一个块的步长
            
        返回:
            nn.Sequential: 包含多个残差块的顺序容器
        """
        layers = []
        # 第一个block可能改变维度
        layers.append(AttentionResidualBlock(in_channels, out_channels, stride))
        
        # 额外添加BatchNorm和Dropout提高稳定性
        if stride != 1:
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.Dropout2d(0.1))
        
        # 剩余blocks保持维度不变
        for _ in range(1, blocks):
            layers.append(AttentionResidualBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x: 输入图像张量, 形状为 [batch_size, in_channels, height, width]
            
        返回:
            tuple: (position_logits, grade_logits)
                - position_logits: 位置分类的logits，形状为 [batch_size, 3]
                - grade_logits: 等级分类的logits，形状为 [batch_size, 5]
        """
        # 特征提取
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # 应用Dropout
        x = self.dropout(x)
        
        # 残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 全局平均池化
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # 特征整合
        shared_features = self.fc_features(x)
        
        # 位置分类
        position_logits = self.position_classifier(shared_features)
        
        # 等级分类
        grade_logits = self.grade_classifier(shared_features)
        
        return position_logits, grade_logits

class SegNetEncoder(nn.Module):
    """
    SegNet编码器模块，用于特征提取和下采样
    包含编码器阶段的卷积层、批归一化和池化索引记录
    """
    def __init__(self, in_channels, out_channels):
        super(SegNetEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
    def forward(self, x):
        # 第一个卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # 最大池化并保存索引用于上采样
        x_size = x.size()
        x, indices = self.maxpool(x)
        
        return x, indices, x_size

class SegNetDecoder(nn.Module):
    """
    SegNet解码器模块，用于上采样和特征重建
    使用最大池化索引进行非线性上采样
    """
    def __init__(self, in_channels, out_channels):
        super(SegNetDecoder, self).__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, indices, output_size):
        # 最大反池化
        x = self.unpool(x, indices, output_size=output_size)
        
        # 第一个卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x

class DiseaseSegNet(nn.Module):
    """
    基于SegNet架构的玉米南方锈病多任务分类和分割模型
    使用编码器-解码器结构进行精确的图像分割
    同时完成两个任务：
    1. 感染部位分类: 下部/中部/上部 (3分类)
    2. 感染等级预测: 严重程度 (回归任务)
    """
    def __init__(self, in_channels=3, img_size=128):
        super(DiseaseSegNet, self).__init__()
        self.in_channels = in_channels
        self.img_size = img_size
        
        # 编码器 (下采样路径)
        self.enc1 = SegNetEncoder(in_channels, 64)
        self.enc2 = SegNetEncoder(64, 128)
        self.enc3 = SegNetEncoder(128, 256)
        self.enc4 = SegNetEncoder(256, 512)
        
        # 中间层 - 无下采样的编码器块
        self.middle_conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.middle_bn1 = nn.BatchNorm2d(512)
        self.middle_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.middle_bn2 = nn.BatchNorm2d(512)
        self.middle_relu = nn.ReLU(inplace=True)
        
        # 解码器 (上采样路径)
        self.dec4 = SegNetDecoder(512, 256)
        self.dec3 = SegNetDecoder(256, 128)
        self.dec2 = SegNetDecoder(128, 64)
        self.dec1 = SegNetDecoder(64, 64)
        
        # 分割输出层 - 生成逐像素分类图 (分割结果)
        self.segmentation_head = nn.Conv2d(64, 1, kernel_size=1)
        
        # 全局平均池化 - 用于分类任务
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 位置分类头
        self.position_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 3)  # 输出3个类别: 下部/中部/上部
        )
        
        # 等级回归头
        self.grade_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)  # 输出1个值进行回归
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重，使用He初始化提高训练效率"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 编码器路径 - 保存每层的池化索引和特征图尺寸
        x, indices1, size1 = self.enc1(x)
        x, indices2, size2 = self.enc2(x)
        x, indices3, size3 = self.enc3(x)
        x, indices4, size4 = self.enc4(x)
        
        # 中间层处理
        x = self.middle_conv1(x)
        x = self.middle_bn1(x)
        x = self.middle_relu(x)
        x = self.middle_conv2(x)
        x = self.middle_bn2(x)
        x = self.middle_relu(x)
        
        # 保存中间层特征用于分类任务
        features = x
        
        # 解码器路径 - 使用保存的池化索引进行精确上采样
        x = self.dec4(x, indices4, size4)
        x = self.dec3(x, indices3, size3)
        x = self.dec2(x, indices2, size2)
        x = self.dec1(x, indices1, size1)
        
        # 分割头 - 生成分割图
        segmentation = self.segmentation_head(x)
        
        # 分类头 - 预测位置和等级
        pooled_features = self.avgpool(features)
        position_logits = self.position_classifier(pooled_features)
        grade_values = self.grade_classifier(pooled_features)
        
        # 返回位置分类、等级回归和分割图
        return position_logits, grade_values, segmentation

class DiseaseSegNetPlus(nn.Module):
    """
    增强版SegNet模型，结合了SegNet分割能力和注意力机制
    添加深度可分离卷积、特征金字塔网络(FPN)和增强特征融合机制
    针对玉米南方锈病场景定制优化，平衡三个任务
    """
    def __init__(self, in_channels=3, img_size=128):
        super(DiseaseSegNetPlus, self).__init__()
        self.in_channels = in_channels
        self.img_size = img_size
        
        # 编码器 (下采样路径) - 使用深度可分离卷积以提高效率
        self.enc1 = self._make_encoder_block(in_channels, 64)
        self.enc2 = self._make_encoder_block(64, 128)
        self.enc3 = self._make_encoder_block(128, 256)
        self.enc4 = self._make_encoder_block(256, 512)
        
        # 注意力模块
        self.channel_attention = ChannelAttention(512)
        self.spatial_attention = SpatialAttention()
        
        # 中间层 - 使用普通卷积替代深度可分离卷积以避免维度问题
        self.middle_conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.middle_bn1 = nn.BatchNorm2d(512)
        self.middle_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.middle_bn2 = nn.BatchNorm2d(512)
        self.middle_relu = nn.ReLU(inplace=True)
        
        # FPN侧边连接 - 用于多尺度特征融合
        self.lateral_conv1 = nn.Conv2d(512, 256, kernel_size=1)
        self.lateral_conv2 = nn.Conv2d(256, 128, kernel_size=1)
        self.lateral_conv3 = nn.Conv2d(128, 64, kernel_size=1)
        
        # 用于上采样的操作，确保特征图尺寸匹配
        self.fpn_upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.fpn_upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.fpn_upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # 解码器 (上采样路径) - 使用简化版解码器避免维度问题
        # 解码器部分 - 重新设计，确保通道匹配
        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec4_conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.dec4_bn1 = nn.BatchNorm2d(256)
        self.dec4_relu1 = nn.ReLU(inplace=True)
        
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec3_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec3_bn1 = nn.BatchNorm2d(128)
        self.dec3_relu1 = nn.ReLU(inplace=True)
        
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec2_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec2_bn1 = nn.BatchNorm2d(64)
        self.dec2_relu1 = nn.ReLU(inplace=True)
        
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec1_conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dec1_bn1 = nn.BatchNorm2d(64)
        self.dec1_relu1 = nn.ReLU(inplace=True)
        
        # 分割头 - 使用1x1卷积逐步细化
        self.seg_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )
        
        # 全局特征提取 - 结合全局和局部信息
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(512*2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # 位置分类头 - 使用更细粒度的层级设计
        self.position_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 3)  # 3类：下部/中部/上部
        )
        
        # 等级回归头 - 使用连续激活函数提高回归精度
        self.grade_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _depthwise_conv(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        """深度可分离卷积，降低参数量提高效率"""
        return nn.Sequential(
            # 深度卷积 - 每个通道单独卷积
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # 逐点卷积 - 跨通道信息融合
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _make_encoder_block(self, in_channels, out_channels):
        """创建编码器块，使用深度可分离卷积"""
        return nn.Sequential(
            # 使用普通卷积而不是深度可分离卷积作为第一层，以处理通道数变化
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            self._depthwise_conv(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        )
    
    def _initialize_weights(self):
        """初始化模型权重，使用He初始化提高训练效率"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 保存所有编码器输出和索引用于跳跃连接
        skip_connections = []
        indices_list = []
        sizes_list = []
        
        # 编码器路径
        # Enc1
        x1 = x
        for layer in list(self.enc1.children())[:-1]:  # 除了最后的MaxPool
            x1 = layer(x1)
        skip_connections.append(x1)
        size1 = x1.size()
        x1, indices1 = self.enc1[-1](x1)  # MaxPool with indices
        indices_list.append(indices1)
        sizes_list.append(size1)
        
        # Enc2
        x2 = x1
        for layer in list(self.enc2.children())[:-1]:
            x2 = layer(x2)
        skip_connections.append(x2)
        size2 = x2.size()
        x2, indices2 = self.enc2[-1](x2)
        indices_list.append(indices2)
        sizes_list.append(size2)
        
        # Enc3
        x3 = x2
        for layer in list(self.enc3.children())[:-1]:
            x3 = layer(x3)
        skip_connections.append(x3)
        size3 = x3.size()
        x3, indices3 = self.enc3[-1](x3)
        indices_list.append(indices3)
        sizes_list.append(size3)
        
        # Enc4
        x4 = x3
        for layer in list(self.enc4.children())[:-1]:
            x4 = layer(x4)
        skip_connections.append(x4)
        size4 = x4.size()
        x4, indices4 = self.enc4[-1](x4)
        indices_list.append(indices4)
        sizes_list.append(size4)
        
        # 应用注意力机制
        ca_features = self.channel_attention(x4) * x4
        sa_features = self.spatial_attention(ca_features) * ca_features
        x = sa_features
        
        # 中间层处理
        x = self.middle_conv1(x)
        x = self.middle_bn1(x)
        x = self.middle_relu(x)
        x = self.middle_conv2(x)
        x = self.middle_bn2(x)
        x = self.middle_relu(x)
        
        # 保存中间层特征用于分类任务
        features = x
        
        # FPN侧边连接处理
        p4 = self.lateral_conv1(x)
        
        # 解码器路径与FPN特征融合 - 使用简化的解码器
        # 解码阶段4: 512->256
        x = self.unpool4(x, indices4, output_size=sizes_list[3])
        x = self.dec4_conv1(x)
        x = self.dec4_bn1(x)
        x = self.dec4_relu1(x)
        
        # 解码阶段3: 256->128
        x = self.unpool3(x, indices3, output_size=sizes_list[2])
        x = self.dec3_conv1(x)
        x = self.dec3_bn1(x)
        x = self.dec3_relu1(x)
        
        # 解码阶段2: 128->64
        x = self.unpool2(x, indices2, output_size=sizes_list[1])
        x = self.dec2_conv1(x)
        x = self.dec2_bn1(x)
        x = self.dec2_relu1(x)
        
        # 解码阶段1: 64->64
        x = self.unpool1(x, indices1, output_size=sizes_list[0])
        x = self.dec1_conv1(x)
        x = self.dec1_bn1(x)
        x = self.dec1_relu1(x)
        
        # 分割头 - 生成分割图
        segmentation = self.seg_head(x)
        
        # 分类头 - 结合全局平均池化和最大池化
        avg_pool = self.global_avg_pool(features).view(features.size(0), -1)
        max_pool = self.global_max_pool(features).view(features.size(0), -1)
        
        # 特征融合
        pooled_features = torch.cat([avg_pool, max_pool], dim=1)
        fused_features = self.fusion(pooled_features)
        
        # 预测位置和等级
        position_logits = self.position_classifier(fused_features)
        grade_values = self.grade_classifier(fused_features)
        
        # 返回位置分类、等级回归和分割图
        return position_logits, grade_values, segmentation

def get_model(model_type='simple', in_channels=3, img_size=128):
    """
    根据指定类型获取模型实例
    
    参数:
        model_type: 模型类型，可选值: 'simple'(简单CNN), 'resnet'(ResNet), 'resnet+'(ResNet+注意力), 'segnet'(SegNet), 'segnet+'(SegNet+注意力)
        in_channels: 输入图像的通道数
        img_size: 输入图像的尺寸
    
    返回:
        model: 模型实例
    """
    if model_type == 'simple':
        return DiseaseClassifier(in_channels, img_size)
    elif model_type == 'resnet':
        return DiseaseResNet(in_channels, img_size)
    elif model_type == 'resnet+':
        return DiseaseResNetPlus(in_channels, img_size)
    elif model_type == 'segnet':
        return DiseaseSegNet(in_channels, img_size)
    elif model_type == 'segnet+':
        return DiseaseSegNetPlus(in_channels, img_size)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}，可用选项: 'simple', 'resnet', 'resnet+', 'segnet', 'segnet+'")