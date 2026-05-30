@echo off
rem 训练FCN模型的批处理脚本，使用混合精度训练和早停策略
rem 适用于Windows系统

rem 设置环境变量优化CUDA性能
set CUDA_LAUNCH_BLOCKING=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

rem 设置训练参数
set DATA_ROOT=./guanceng-bit
set JSON_ROOT=./biaozhu_json
set OUTPUT_DIR=./output_fcn
set BATCH_SIZE=32
set NUM_WORKERS=8
set IMG_SIZE=256
set LEARNING_RATE=0.0001
set WEIGHT_DECAY=1e-6
set EPOCHS=50
set MONITOR=f1
set PATIENCE=5
set POS_WEIGHT=0.7
set GRAD_ACCUMULATION=2

rem 创建日志目录
if not exist logs mkdir logs

rem 生成带时间戳的日志文件名
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "TIMESTAMP=%dt:~0,8%_%dt:~8,6%"
set "LOG_FILE=logs\fcn_training_%TIMESTAMP%.log"

echo 开始FCN模型训练，日志将保存至 %LOG_FILE%
echo 训练配置:
echo   CUDA内存优化: max_split_size_mb=128
echo   数据目录: %DATA_ROOT%
echo   JSON目录: %JSON_ROOT%
echo   输出目录: %OUTPUT_DIR%
echo   批次大小: %BATCH_SIZE%
echo   梯度累积步数: %GRAD_ACCUMULATION% (有效批次大小: %BATCH_SIZE%*%GRAD_ACCUMULATION%)
echo   工作进程: %NUM_WORKERS%
echo   图像大小: %IMG_SIZE%
echo   学习率: %LEARNING_RATE%
echo   权重衰减: %WEIGHT_DECAY%
echo   总轮数: %EPOCHS%
echo   监控指标: %MONITOR%
echo   早停耐心轮数: %PATIENCE%
echo   位置任务权重: %POS_WEIGHT%
echo ==========================================

rem 执行训练命令并将输出同时显示在控制台和保存到日志文件
rem 注意：Windows下没有原生的tee命令，但PowerShell有类似功能
powershell "python run_fcn.py --data_root '%DATA_ROOT%' --json_root '%JSON_ROOT%' --output_dir '%OUTPUT_DIR%' --batch_size %BATCH_SIZE% --num_workers %NUM_WORKERS% --img_size %IMG_SIZE% --learning_rate %LEARNING_RATE% --weight_decay %WEIGHT_DECAY% --epochs %EPOCHS% --monitor_metric %MONITOR% --patience %PATIENCE% --pos_weight %POS_WEIGHT% --grad_accumulation %GRAD_ACCUMULATION% --cudnn_benchmark --amp --pin_memory | Tee-Object -FilePath '%LOG_FILE%'"

rem 检查训练是否成功完成
if %ERRORLEVEL% EQU 0 (
    echo 训练成功完成！
    echo 模型保存在: %OUTPUT_DIR%
    echo 训练日志保存在: %LOG_FILE%
    
    rem 创建可视化目录
    set VIZ_DIR=%OUTPUT_DIR%\visualization
    
    rem 使用最佳模型进行可视化
    echo 生成可视化结果...
    python visualize_fcn.py ^
        --data_root "%DATA_ROOT%" ^
        --json_root "%JSON_ROOT%" ^
        --model_path "%OUTPUT_DIR%\best_model.pth" ^
        --output_dir "%VIZ_DIR%" ^
        --num_samples 8 ^
        --img_size %IMG_SIZE%
    
    echo 可视化结果保存在: %VIZ_DIR%
) else (
    echo 训练过程中出现错误，请检查日志文件: %LOG_FILE%
) 