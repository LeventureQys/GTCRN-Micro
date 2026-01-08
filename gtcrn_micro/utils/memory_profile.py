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


def calculate_input_shape_from_bytes(audio_bytes, sample_rate=16000, bits_per_sample=16, 
                                     num_channels=1, hop_length=256, n_fft=512):
    """
    根据音频字节长度计算输入形状
    
    Args:
        audio_bytes: 音频字节数
        sample_rate: 采样率 (默认: 16000 Hz)
        bits_per_sample: 每个采样的位数 (默认: 16, 即2字节)
        num_channels: 声道数 (默认: 1, 单声道)
        hop_length: STFT hop length (默认: 256)
        n_fft: FFT 大小 (默认: 512)
    
    Returns:
        (batch, freq, time_frames, channels) 形状元组
    """
    # 计算每个采样的字节数
    bytes_per_sample = bits_per_sample // 8
    
    # 计算总采样点数（时间上的采样点数）
    # 对于多声道音频，需要除以声道数得到时间上的采样点数
    total_samples = audio_bytes // (bytes_per_sample * num_channels)
    
    # 计算 STFT 时间帧数
    # 时间帧数 = 采样点数 / hop_length
    time_frames = max(1, total_samples // hop_length)
    
    # 频率维度 (n_fft // 2 + 1)
    freq_bins = n_fft // 2 + 1  # 257 for n_fft=512
    
    return (1, freq_bins, time_frames, 2)  # 2 是复数频谱的实部和虚部


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
                # 使用预热后的内存作为基准，测量推理后的峰值内存
                # 对于小输入，单次推理的 before/after 差值可能不准确
                _ = model(dummy_input)
                # 强制垃圾回收以确保内存统计准确
                import gc
                gc.collect()
                current_memory = process.memory_info().rss
                memory_usage.append(max(0, current_memory - after_warmup_memory))
    
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
    
    # 测试不同音频字节长度的内存占用
    print("\n" + "-" * 80)
    print("5. 不同音频字节长度的内存占用 (16kHz采样率)")
    print("-" * 80)
    
    # 测试字节长度：512, 1024, 1536 bytes
    test_byte_sizes = [512, 1024, 1536]
    sample_rate = 16000
    hop_length = 256
    bits_per_sample = 16
    num_channels = 1
    
    print(f"音频格式: {bits_per_sample}位, {num_channels}声道, {sample_rate}Hz采样率")
    print(f"STFT参数: hop_length={hop_length}, n_fft=512")
    print("-" * 80)
    
    # 为了更准确地测量CPU内存，先清理内存
    import gc
    gc.collect()
    
    # 预热模型（使用一个中等大小的输入）
    warmup_shape = (1, 257, 32, 2)
    warmup_input = torch.randn(*warmup_shape, device=device)
    with torch.no_grad():
        _ = model(warmup_input)
    del warmup_input
    gc.collect()
    
    # 获取基线内存（预热后）
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()
        baseline_memory = torch.cuda.memory_allocated(device)
    else:
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss
    
    for audio_bytes in test_byte_sizes:
        # 计算输入形状
        input_shape = calculate_input_shape_from_bytes(
            audio_bytes, 
            sample_rate=sample_rate,
            bits_per_sample=bits_per_sample,
            num_channels=num_channels,
            hop_length=hop_length,
            n_fft=512
        )
        
        # 计算对应的音频信息
        bytes_per_sample = bits_per_sample // 8
        total_samples = audio_bytes // (bytes_per_sample * num_channels)
        audio_duration = total_samples / sample_rate if sample_rate > 0 else 0
        time_frames = input_shape[2]
        
        # 创建输入并计算理论内存
        dummy_input = torch.randn(*input_shape, device=device)
        input_memory = dummy_input.element_size() * dummy_input.nelement()
        
        # 计算输出内存
        with torch.no_grad():
            output = model(dummy_input)
        output_memory = output.element_size() * output.nelement()
        
        if device == 'cuda' and torch.cuda.is_available():
            # GPU模式：使用CUDA内存统计
            torch.cuda.empty_cache()
            before_mem = torch.cuda.memory_allocated(device)
            
            # 执行推理
            with torch.no_grad():
                _ = model(dummy_input)
            
            torch.cuda.empty_cache()
            after_mem = torch.cuda.memory_allocated(device)
            measured_mem = max(0, after_mem - before_mem)
            total_mem = max(0, after_mem - baseline_memory)
        else:
            # CPU模式：直接计算理论内存占用
            # 由于RSS粒度太大，对于小输入无法准确测量，所以使用理论计算
            # 输入内存
            input_mem = input_memory
            
            # 输出内存（与输入形状相同）
            output_mem = output_memory
            
            # 估算激活值内存
            # GTCRN-Micro的主要激活值来源：
            # 1. Encoder输出: (B, C, T, F) 其中通道数C=16，频率F会逐渐减小
            # 2. GTCN层激活值: 与encoder输出相同大小
            # 3. Decoder中间激活值
            
            # 估算：基于输入形状和网络结构
            # 输入: (1, 257, T, 2)
            # Encoder会下采样频率维度，时间维度基本不变
            # 激活值大约是输入的 2-4倍（考虑到中间层）
            B, F, T, C = input_shape
            # 估算激活值内存（基于经验值）
            # 对于GTCRN-Micro，激活值内存大约是输入的3-4倍
            activation_memory = input_memory * 3.5  # 经验估算值
            
            # 总内存 = 输入 + 激活值 + 输出
            measured_mem = input_mem + activation_memory + output_mem
            total_mem = measured_mem
        
        # 清理
        del dummy_input, output
        gc.collect()
        
        # 显示结果
        if device == 'cuda' and torch.cuda.is_available():
            print(f"  音频字节: {audio_bytes:4d}B | "
                  f"采样点数: {total_samples:4d} | "
                  f"时长: {audio_duration*1000:6.2f}ms | "
                  f"时间帧: {time_frames:2d} | "
                  f"输入形状: {str(input_shape):<20} | "
                  f"内存占用: {format_size(measured_mem)}")
        else:
            # CPU模式显示理论计算值
            print(f"  音频字节: {audio_bytes:4d}B | "
                  f"采样点数: {total_samples:4d} | "
                  f"时长: {audio_duration*1000:6.2f}ms | "
                  f"时间帧: {time_frames:2d} | "
                  f"输入形状: {str(input_shape):<20} | "
                  f"内存占用: {format_size(total_mem)} (估算: 输入+激活值+输出)")
    
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

