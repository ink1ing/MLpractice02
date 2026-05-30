#!/usr/bin/env python
# 核心文件：演示脚本，展示LodgeNet模型的预测和可视化功能
# LodgeNet演示脚本：快速开始LodgeNet训练的简单示例
# 展示基本的训练流程和参数配置

import os
import sys

def main():
    """
    LodgeNet训练演示
    """
    print("=" * 60)
    print("🌽 LodgeNet - 玉米锈病识别与预测深度学习模型")
    print("=" * 60)
    
    print("\n📋 模型特点:")
    print("  ✅ 多任务学习：图像分割 + 位置分类 + 病害等级回归")
    print("  ✅ 注意力机制：自动聚焦病害相关区域")
    print("  ✅ 混合精度训练：提高训练效率")
    print("  ✅ 完整监控：50轮训练，无EarlyStopping")
    
    print("\n🎯 目标指标:")
    print("  • 位置分类准确率 > 90%")
    print("  • 位置分类F1分数 > 0.85")
    print("  • 位置分类召回率 > 0.85")
    print("  • 位置分类精确率 > 0.85")
    print("  • 病害等级MAE < 0.15")
    print("  • 总损失 < 0.2")
    
    print("\n🚀 快速开始:")
    print("  1. 测试模型架构:")
    print("     python test_lodgenet.py --test_architecture --test_loss")
    
    print("\n  2. 开始训练（推荐使用启动脚本）:")
    print("     python run_lodgenet.py")
    
    print("\n  3. 或直接运行训练:")
    print("     python lodgenet_train.py --data_root ./guanceng-bit --json_root ./biaozhu_json --num_epochs 50")
    
    print("\n  4. 测试训练好的模型:")
    print("     python test_lodgenet.py --model_path ./output_lodgenet/best_model.pth --visualize")
    
    print("\n📁 文件结构:")
    files = [
        "lodgenet_model.py     # LodgeNet模型定义",
        "lodgenet_train.py     # 训练脚本",
        "run_lodgenet.py       # 训练启动脚本",
        "test_lodgenet.py      # 测试和评估脚本",
        "README_LodgeNet.md    # 详细文档"
    ]
    
    for file in files:
        print(f"  📄 {file}")
    
    print("\n⚙️ 推荐配置:")
    config = {
        "批次大小": "8 (GPU内存允许可增大)",
        "训练轮数": "50 (无EarlyStopping)",
        "学习率": "0.0001 (Adam优化器)",
        "图像尺寸": "128x128",
        "任务权重": "[0.4, 0.3, 0.3] (分割, 位置, 等级)"
    }
    
    for key, value in config.items():
        print(f"  🔧 {key}: {value}")
    
    print("\n💡 提示:")
    print("  • 使用GPU训练可显著提升速度")
    print("  • 训练日志自动保存到 ./logs/ 目录")
    print("  • 模型检查点保存到 ./output_lodgenet/ 目录")
    print("  • 支持混合精度训练，自动优化内存使用")
    
    print("\n📊 模型信息:")
    try:
        from lodgenet_model import get_lodgenet_model, count_parameters
        model = get_lodgenet_model()
        param_count = count_parameters(model)
        print(f"  📈 参数数量: {param_count:,}")
        print(f"  🏗️ 架构: U-Net + 注意力机制 + ASPP")
        print(f"  🎯 任务: 3个（分割 + 分类 + 回归）")
    except Exception as e:
        print(f"  ⚠️ 无法加载模型信息: {e}")
    
    print("\n" + "=" * 60)
    
    # 交互式选择
    while True:
        print("\n请选择操作:")
        print("  1. 测试模型架构")
        print("  2. 开始训练")
        print("  3. 查看详细文档")
        print("  4. 退出")
        
        choice = input("\n请输入选择 (1-4): ").strip()
        
        if choice == '1':
            print("\n🧪 开始测试模型架构...")
            os.system("python test_lodgenet.py --test_architecture --test_loss")
            
        elif choice == '2':
            print("\n🚀 启动训练...")
            response = input("确认开始训练？这将需要较长时间 (y/N): ").strip().lower()
            if response in ['y', 'yes']:
                os.system("python run_lodgenet.py")
            else:
                print("训练已取消")
                
        elif choice == '3':
            print("\n📖 详细文档请查看: README_LodgeNet.md")
            if os.path.exists("README_LodgeNet.md"):
                print("文档已存在，可以直接查看")
            else:
                print("文档文件不存在")
                
        elif choice == '4':
            print("\n👋 感谢使用LodgeNet！")
            break
            
        else:
            print("❌ 无效选择，请重新输入")

if __name__ == "__main__":
    main() 