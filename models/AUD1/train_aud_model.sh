#!/bin/bash
# 核心文件，完整运行项目不可缺少
# 训练AUD模型的脚本
# 基于混合精度训练，充分利用GPU性能

# 确保输出目录存在
mkdir -p ./output_aud

# 训练AUD模型
python run_optimized.py \
    --data_root ./guanceng-bit \
    --json_root ./biaozhu_json \
    --model_type aud \
    --batch_size 8 \
    --lr 0.0001 \
    --weight_decay 1e-5 \
    --epochs 50 \
    --task_weights 0.7 0.3 \
    --output_dir ./output_aud \
    --amp \
    --focal_loss \
    --gamma 2.0 \
    --log_interval 10 \
    --save_interval 5

# 完成后分析训练结果
python check_training_results.py --output_dir ./output_aud

echo "AUD模型训练完成，结果保存在 ./output_aud 目录中"
echo "最佳F1模型: ./output_aud/best_model_f1.pth"
echo "最佳MAE模型: ./output_aud/best_model_mae.pth" 