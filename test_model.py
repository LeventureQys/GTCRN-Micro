#!/usr/bin/env python3
"""
GTCRN-Micro 模型测试脚本
支持单文件测试和批量测试

使用方法:
    python test_model.py <音频文件或目录> [选项]
    
示例:
    # 测试单个文件
    python test_model.py noisy_audio.wav -o enhanced.wav
    
    # 批量处理目录
    python test_model.py input_dir/ -o output_dir/
    
    # 使用 GPU
    python test_model.py noisy_audio.wav -d cuda
"""
import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from gtcrn_micro.models.gtcrn_micro import GTCRNMicro


def load_model(checkpoint_path="gtcrn_micro/ckpts/best_model_dns3.tar"):
    """加载模型"""
    print(f"正在加载模型: {checkpoint_path}")
    model = GTCRNMicro().eval()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    state_dict = (
        checkpoint.get("model", None) or
        checkpoint.get("state_dict", None) or
        checkpoint.get("model_state_dict", None) or
        checkpoint
    )
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"警告: 缺少参数: {len(missing)} 个")
    if unexpected:
        print(f"警告: 未使用参数: {len(unexpected)} 个")
    
    print("模型加载完成！\n")
    return model


def process_audio(model, audio_path, output_path=None, device="cpu"):
    """处理单个音频文件"""
    print(f"处理音频: {audio_path}")
    
    # 加载音频
    try:
        audio, sr = sf.read(audio_path, dtype="float32")
    except Exception as e:
        print(f"错误: 无法读取音频文件: {e}")
        return None
    
    # 重采样到16kHz
    if sr != 16000:
        print(f"  重采样: {sr} Hz -> 16000 Hz")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    print(f"  音频长度: {len(audio) / sr:.2f} 秒")
    print(f"  采样点数: {len(audio)}")
    
    # 转换为频域
    model = model.to(device)
    audio_tensor = torch.from_numpy(audio).to(device)
    
    input_stft = torch.stft(
        audio_tensor,
        n_fft=512,
        hop_length=256,
        win_length=512,
        window=torch.hann_window(512).pow(0.5).to(device),
        return_complex=False,
    )
    
    # 模型推理
    print("  运行模型推理...")
    with torch.no_grad():
        output_stft = model(input_stft.unsqueeze(0))[0]
    
    # 转换回时域
    output_stft_complex = torch.view_as_complex(output_stft.contiguous())
    enhanced_audio = torch.istft(
        output_stft_complex,
        n_fft=512,
        hop_length=256,
        win_length=512,
        window=torch.hann_window(512).pow(0.5).to(device),
        return_complex=False,
    ).cpu().numpy()
    
    # 保存结果
    if output_path is None:
        output_path = str(Path(audio_path).with_name(
            Path(audio_path).stem + "_enhanced.wav"
        ))
    
    sf.write(output_path, enhanced_audio, sr)
    print(f"  增强音频已保存: {output_path}")
    
    # 统计信息
    print(f"\n  原始音频统计:")
    print(f"    RMS: {np.sqrt(np.mean(audio**2)):.4f}")
    print(f"    最大值: {np.abs(audio).max():.4f}")
    print(f"  增强音频统计:")
    print(f"    RMS: {np.sqrt(np.mean(enhanced_audio**2)):.4f}")
    print(f"    最大值: {np.abs(enhanced_audio).max():.4f}")
    print()
    
    return enhanced_audio, sr


def main():
    parser = argparse.ArgumentParser(
        description="GTCRN-Micro 模型测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 测试单个文件
  python test_model.py noisy_audio.wav -o enhanced.wav
  
  # 批量处理目录中的所有 wav 文件
  python test_model.py input_dir/ -o output_dir/
  
  # 使用 GPU（如果有）
  python test_model.py noisy_audio.wav -d cuda
  
  # 使用示例音频快速测试
  python test_model.py gtcrn_micro/examples/gtcrn_micro/noisy1.wav
        """
    )
    
    parser.add_argument(
        "audio_path",
        type=str,
        help="输入音频文件路径（.wav 文件）或目录（批量处理所有 .wav 文件）"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="输出音频文件路径（单文件）或目录（批量处理）"
    )
    parser.add_argument(
        "-c", "--checkpoint",
        type=str,
        default="gtcrn_micro/ckpts/best_model_dns3.tar",
        help="模型检查点路径（默认: gtcrn_micro/ckpts/best_model_dns3.tar）"
    )
    parser.add_argument(
        "-d", "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="计算设备（默认: cpu）"
    )
    
    args = parser.parse_args()
    
    # 检查设备
    if args.device == "cuda" and not torch.cuda.is_available():
        print("警告: CUDA 不可用，改用 CPU")
        args.device = "cpu"
    
    # 加载模型
    if not Path(args.checkpoint).exists():
        print(f"错误: 模型文件不存在: {args.checkpoint}")
        sys.exit(1)
    
    model = load_model(args.checkpoint)
    
    audio_path = Path(args.audio_path)
    
    # 单文件处理
    if audio_path.is_file() and audio_path.suffix.lower() == ".wav":
        output_path = args.output
        result = process_audio(model, str(audio_path), output_path, args.device)
        if result is None:
            sys.exit(1)
    
    # 批量处理
    elif audio_path.is_dir():
        output_dir = Path(args.output) if args.output else audio_path / "enhanced"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        audio_files = sorted(audio_path.glob("*.wav"))
        if not audio_files:
            print(f"错误: 目录中没有找到 .wav 文件: {audio_path}")
            sys.exit(1)
        
        print(f"找到 {len(audio_files)} 个音频文件\n")
        
        success_count = 0
        for i, audio_file in enumerate(audio_files, 1):
            print(f"[{i}/{len(audio_files)}]")
            output_file = output_dir / f"{audio_file.stem}_enhanced.wav"
            try:
                process_audio(model, str(audio_file), str(output_file), args.device)
                success_count += 1
            except Exception as e:
                print(f"  错误: 处理 {audio_file.name} 失败: {e}\n")
        
        print(f"批量处理完成: {success_count}/{len(audio_files)} 个文件成功处理")
        print(f"输出目录: {output_dir}")
    
    else:
        print(f"错误: 路径不存在或不是有效的音频文件/目录: {audio_path}")
        sys.exit(1)
    
    print("\n测试完成！")


if __name__ == "__main__":
    main()

