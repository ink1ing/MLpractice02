# 玉米南方锈病多任务智能识别 · 九模型归档

本仓库归档同一课题下的九个深度学习模型实现：基于无人机多光谱图像(`.tif`)与人工标注(`.json`)，对玉米南方锈病进行多任务识别。各模型共同的两个核心任务：

1. **感染部位分类**：上部 / 中部 / 下部（3 分类）
2. **感染等级判断**：无 / 轻度 / 中度 / 重度 / 极重度（5 分类，对应病害等级 0/3/5/7/9）

部分模型在此基础上额外实现**病斑区域分割**（如 SegNet、UNet 系、LodgeNet）。

## 模型归档（九个）

九个模型均位于 `models/` 目录，每个子目录保留各自的源码与 README：

| 模型 | 架构 / 说明 |
| --- | --- |
| [`models/MLpractice04`](models/MLpractice04) | 早期基础多任务 CNN（阶段一小样本 → 阶段二完整数据集） |
| [`models/ResNet4`](models/ResNet4) | ResNet 主干，感染部位分类 + 等级判断双任务 |
| [`models/ResNet4.5`](models/ResNet4.5) | ResNet 改进迭代版 |
| [`models/SegNet2`](models/SegNet2) | SegNet 架构，新增病斑区域分割（三任务） |
| [`models/AUD1`](models/AUD1) | AUD（Attentive Unified Diffusion）架构 |
| [`models/UNet1`](models/UNet1) | UNet，分割 + 分类多任务 |
| [`models/UNet-Attention1`](models/UNet-Attention1) | UNet + Attention |
| [`models/LodgeNet2`](models/LodgeNet2) | U-Net + Attention + ASPP，混合精度，分割/部位/等级 |
| [`models/FCN1`](models/FCN1) | FCN 全卷积网络 |

各模型目录遵循统一约定：`dataset.py`（数据加载）、`model.py`（模型定义）、`train.py` / `test.py`（训练与评估）、`utils.py`（工具）、`check_*.py`（环境/数据自检）、`run_*.py`（运行入口）、`README.md`（说明）；部分模型含架构特有脚本（如 `unet_model.py`、`lodgenet_*.py`、`visualize_fcn.py` 等）。

**各模型的具体训练 / 测试命令与参数，请见其子目录下的 `README.md`。**

## 数据格式

- `.tif`：无人机多光谱图像
- `.json`：人工标注（感染部位与等级）

按部位组织的典型目录结构：

```
Mini Data/
    ├── mini_14l/        # 下部样本 .tif
    ├── mini_14l_json/   # 下部样本 .json
    ├── mini_14m/        # 中部样本 .tif
    ├── mini_14m_json/   # 中部样本 .json
    ├── mini_14t/        # 上部样本 .tif
    └── mini_14t_json/   # 上部样本 .json
```

## 环境要求

基于 PyTorch，主要依赖：

```bash
pip install torch torchvision rasterio scikit-learn matplotlib seaborn tqdm
```

- Python 3.6+
- PyTorch 1.7+
- rasterio（读取 `.tif`）

> 归档说明：各模型来自独立仓库，已统一清理——剔除各自的 `.git`、`__pycache__`、`*.pyc`、`.DS_Store`，删除备份/残留文件（`*.backup`、`*.fixed`、`*.save`、无引用的 `*_backup.py`）、训练日志（`logs/`）及冗余的嵌套 `.gitignore`（忽略规则集中到顶层）。源码与文档原样保留，未改动任何在用代码。

## 许可证

仅供学习与研究使用。
