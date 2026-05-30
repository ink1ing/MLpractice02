#!/bin/bash
# 训练FCN模型的脚本，使用混合精度训练和早停策略
# 自动将训练日志记录到文件中

# 设置环境变量优化CUDA性能
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 设置环境变量以启用PyTorch混合精度训练
export PYTHONIOENCODING=utf-8

# 设置训练参数
DATA_ROOT="./guanceng-bit"
JSON_ROOT="./biaozhu_json"
OUTPUT_DIR="./output_fcn"
BATCH_SIZE=32
NUM_WORKERS=8
IMG_SIZE=256
LEARNING_RATE=0.0001
WEIGHT_DECAY=1e-6
EPOCHS=50
MONITOR="f1"  # 可选：'f1'或'mae'
PATIENCE=5
POS_WEIGHT=0.7
GRAD_ACCUMULATION=2
LR_SCHEDULER="cosine"  # 'plateau', 'cosine', 'step'

# 创建日志目录
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

# 生成带时间戳的日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/fcn_training_${TIMESTAMP}.log"

echo "开始FCN模型训练，日志将保存至 ${LOG_FILE}"
echo "训练配置:"
echo "  CUDA内存优化: max_split_size_mb=128"
echo "  数据目录: ${DATA_ROOT}"
echo "  JSON目录: ${JSON_ROOT}"
echo "  输出目录: ${OUTPUT_DIR}"
echo "  批次大小: ${BATCH_SIZE}"
echo "  梯度累积步数: ${GRAD_ACCUMULATION} (有效批次大小: $((BATCH_SIZE * GRAD_ACCUMULATION)))"
echo "  工作进程: ${NUM_WORKERS}"
echo "  图像大小: ${IMG_SIZE}"
echo "  学习率: ${LEARNING_RATE}"
echo "  权重衰减: ${WEIGHT_DECAY}"
echo "  学习率调度器: ${LR_SCHEDULER}"
echo "  总轮数: ${EPOCHS}"
echo "  监控指标: ${MONITOR}"
echo "  早停耐心轮数: ${PATIENCE}"
echo "  位置任务权重: ${POS_WEIGHT}"
echo "=========================================="

# 执行训练命令并将输出通过tee同时显示在控制台和保存到日志文件
python run_fcn.py \
    --data_root "$DATA_ROOT" \
    --json_root "$JSON_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --img_size "$IMG_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay "$WEIGHT_DECAY" \
    --epochs "$EPOCHS" \
    --monitor_metric "$MONITOR" \
    --patience "$PATIENCE" \
    --pos_weight "$POS_WEIGHT" \
    --grad_accumulation "$GRAD_ACCUMULATION" \
    --lr_scheduler "$LR_SCHEDULER" \
    --cudnn_benchmark \
    --amp \
    --pin_memory \
    2>&1 | tee "$LOG_FILE"

# 检查训练是否成功完成
if [ $? -eq 0 ]; then
    echo "训练成功完成！"
    echo "模型保存在: $OUTPUT_DIR"
    echo "训练日志保存在: $LOG_FILE"
    
    # 创建可视化目录
    VIZ_DIR="$OUTPUT_DIR/visualization"
    
    # 使用最佳模型进行可视化
    echo "生成可视化结果..."
    python visualize_fcn.py \
        --data_root "$DATA_ROOT" \
        --json_root "$JSON_ROOT" \
        --model_path "$OUTPUT_DIR/best_model.pth" \
        --output_dir "$VIZ_DIR" \
        --num_samples 8 \
        --img_size "$IMG_SIZE"
    
    echo "可视化结果保存在: $VIZ_DIR"
else
    echo "训练过程中出现错误，请检查日志文件: $LOG_FILE"
fi 