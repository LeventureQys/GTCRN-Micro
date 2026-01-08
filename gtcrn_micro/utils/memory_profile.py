"""
内存评估脚本
用于评估 GTCRN-Micro 模型的内存占用情况
"""
import os
import sys
import torch
import psutil
import tracemalloc
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gtcrn_micro.models.gtcrn_micro import GTCRNMicro


def format_size(bytes_size):
    """格式化字节大小为易读格式"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def get_model_memory(model, device='cpu'):
    """获取模型本身的内存占用"""
    model_size = 0
    for param in model.parameters():
        param_size = param.element_size() * param.nelement()
        model_size += param_size
    
    # 获取模型缓冲区的内存
    for buffer in model.buffers():
        buffer_size = buffer.element_size() * buffer.nelement()
        model_size += buffer_size
    
    return model_size


def get_model_param_count(model):
    """获取模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def profile_inference_memory(model, input_shape=(1, 257, 63, 2), device='cpu', warmup=3, runs=10):
    """分析推理时的内存占用"""
    model = model.to(device)
    model.eval()
    
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated(device)
        process = None  # 不需要 process 对象
    else:
        # CPU 内存分析
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
    
    # 预热
    dummy_input = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    # 清除缓存
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()
        after_warmup_memory = torch.cuda.memory_allocated(device)
    else:
        # process 在 else 分支中已定义
        assert process is not None, "process should be initialized for CPU mode"
        after_warmup_memory = process.memory_info().rss
    
    # 实际推理测试
    memory_usage = []
    with torch.no_grad():
        for _ in range(runs):
            if device == 'cuda' and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                _ = model(dummy_input)
                peak_memory = torch.cuda.max_memory_allocated(device)
                memory_usage.append(peak_memory - initial_memory)
            else:
                # process 在 else 分支中已定义
                assert process is not None, "process should be initialized for CPU mode"
                before = process.memory_info().rss
                _ = model(dummy_input)
                after = process.memory_info().rss
                memory_usage.append(after - before)
    
    avg_memory = sum(memory_usage) / len(memory_usage)
    max_memory = max(memory_usage)
    min_memory = min(memory_usage)
    
    if device == 'cuda' and torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated(device)
        reserved_memory = torch.cuda.memory_reserved(device)
        return {
            'initial': initial_memory,
            'after_warmup': after_warmup_memory,
            'peak': peak_memory,
            'reserved': reserved_memory,
            'avg_inference': avg_memory,
            'max_inference': max_memory,
            'min_inference': min_memory,
        }
    else:
        return {
            'initial': initial_memory,
            'after_warmup': after_warmup_memory,
            'avg_inference': avg_memory,
            'max_inference': max_memory,
            'min_inference': min_memory,
        }


def main():
    print("=" * 80)
    print("GTCRN-Micro 模型内存占用评估")
    print("=" * 80)
    
    # 检查预训练模型是否存在
    checkpoint_path = Path("gtcrn_micro/ckpts/best_model_dns3.tar")
    if not checkpoint_path.exists():
        print(f"警告: 预训练模型文件不存在: {checkpoint_path}")
        print("将使用未训练的模型进行评估")
        use_pretrained = False
    else:
        use_pretrained = True
        print(f"找到预训练模型: {checkpoint_path}")
    
    # 选择设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 版本: {torch.version.cuda}")
    
    # 创建模型
    print("\n" + "-" * 80)
    print("1. 创建模型")
    print("-" * 80)
    model = GTCRNMicro().to(device)
    
    # 加载预训练权重
    if use_pretrained:
        print(f"加载预训练权重: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        model.load_state_dict(state_dict, strict=False)
        print("预训练权重加载完成")
    
    model.eval()
    print("模型创建完成")
    
    # 获取模型参数信息
    print("\n" + "-" * 80)
    print("2. 模型参数统计")
    print("-" * 80)
    total_params, trainable_params = get_model_param_count(model)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"非训练参数: {total_params - trainable_params:,}")
    
    # 获取模型本身的内存占用
    print("\n" + "-" * 80)
    print("3. 模型参数内存占用")
    print("-" * 80)
    model_memory = get_model_memory(model, device)
    print(f"模型参数内存: {format_size(model_memory)}")
    print(f"参数量 × 4字节 (FP32): {format_size(total_params * 4)}")
    
    # 分析推理内存
    print("\n" + "-" * 80)
    print("4. 推理时内存占用分析")
    print("-" * 80)
    print("输入形状: (1, 257, 63, 2) - 批次大小=1, 频率=257, 时间=63, 通道=2")
    print("开始分析...")
    
    memory_stats = profile_inference_memory(model, input_shape=(1, 257, 63, 2), device=device)
    
    if device == 'cuda':
        print(f"\nGPU 内存使用:")
        print(f"  初始内存: {format_size(memory_stats['initial'])}")
        print(f"  预热后内存: {format_size(memory_stats['after_warmup'])}")
        print(f"  峰值内存: {format_size(memory_stats['peak'])}")
        print(f"  预留内存: {format_size(memory_stats['reserved'])}")
        print(f"  推理内存占用 (平均): {format_size(memory_stats['avg_inference'])}")
        print(f"  推理内存占用 (最大): {format_size(memory_stats['max_inference'])}")
        print(f"  推理内存占用 (最小): {format_size(memory_stats['min_inference'])}")
    else:
        print(f"\nCPU 内存使用:")
        print(f"  初始内存: {format_size(memory_stats['initial'])}")
        print(f"  预热后内存: {format_size(memory_stats['after_warmup'])}")
        print(f"  推理内存占用 (平均): {format_size(memory_stats['avg_inference'])}")
        print(f"  推理内存占用 (最大): {format_size(memory_stats['max_inference'])}")
        print(f"  推理内存占用 (最小): {format_size(memory_stats['min_inference'])}")
    
    # 测试不同输入大小
    print("\n" + "-" * 80)
    print("5. 不同输入大小的内存占用")
    print("-" * 80)
    test_shapes = [
        (1, 257, 63, 2),   # 1秒音频
        (1, 257, 126, 2),  # 2秒音频
        (1, 257, 252, 2),  # 4秒音频
        (1, 257, 504, 2),  # 8秒音频
    ]
    
    for shape in test_shapes:
        time_frames = shape[2]
        audio_duration = time_frames * 256 / 16000  # 假设采样率16kHz, hop_length=256
        mem_stats = profile_inference_memory(model, input_shape=shape, device=device, runs=5)
        
        if device == 'cuda':
            peak_mem = mem_stats['peak'] - mem_stats['initial']
        else:
            peak_mem = mem_stats['max_inference']
        
        print(f"  输入形状 {shape} (~{audio_duration:.1f}秒): {format_size(peak_mem)}")
    
    # 总结
    print("\n" + "=" * 80)
    print("内存评估总结")
    print("=" * 80)
    if device == 'cuda':
        total_peak = memory_stats['peak']
        model_only = model_memory
        inference_overhead = total_peak - model_memory - memory_stats['initial']
        print(f"模型参数内存: {format_size(model_only)}")
        print(f"推理峰值内存: {format_size(total_peak)}")
        print(f"推理开销 (包括激活值和中间结果): {format_size(inference_overhead)}")
    else:
        inference_mem = memory_stats['max_inference']
        model_only = model_memory
        print(f"模型参数内存: {format_size(model_only)}")
        print(f"推理内存占用: {format_size(inference_mem)}")
    
    print("\n评估完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()

