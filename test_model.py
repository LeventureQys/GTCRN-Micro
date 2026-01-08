#!/usr/bin/env python3
"""
GTCRN-Micro 模型测试脚本
支持单文件测试和批量测试，并计算音频质量评估指标

使用方法:
    python test_model.py <音频文件或目录> [选项]

示例:
    # 测试单个文件
    python test_model.py noisy_audio.wav -o enhanced.wav

    # 批量处理目录（带评估指标）
    python test_model.py input_dir/ -o output_dir/ --clean_dir clean_dir/ --evaluate

    # 使用 GPU
    python test_model.py noisy_audio.wav -d cuda
"""
import argparse
import csv
import sys
import warnings
from pathlib import Path
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import librosa
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from gtcrn_micro.models.gtcrn_micro import GTCRNMicro

# 尝试导入评估指标库
try:
    from pesq import pesq, PesqError
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False
    warnings.warn("PESQ not available. Install with: pip install pesq")

try:
    from espnet2.enh.layers.dnsmos import DNSMOS_local
    DNSMOS_AVAILABLE = True
except ImportError:
    DNSMOS_AVAILABLE = False
    warnings.warn("DNSMOS not available. Install espnet2 for DNS-MOS evaluation")


def download_dnsmos_models(dnsmos_dir="./DNSMOS"):
    """自动下载 DNS-MOS 模型文件"""
    dnsmos_path = Path(dnsmos_dir)
    dnsmos_path.mkdir(parents=True, exist_ok=True)

    models = {
        "sig_bak_ovr.onnx": "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx",
        "model_v8.onnx": "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/model_v8.onnx"
    }

    all_exist = True
    for filename, url in models.items():
        filepath = dnsmos_path / filename
        if not filepath.exists():
            all_exist = False
            print(f"正在下载 DNS-MOS 模型: {filename}")
            try:
                # 添加进度显示
                def reporthook(count, block_size, total_size):
                    if total_size > 0:
                        percent = int(count * block_size * 100 / total_size)
                        sys.stdout.write(f"\r  下载进度: {percent}%")
                        sys.stdout.flush()

                urllib.request.urlretrieve(url, filepath, reporthook=reporthook)
                print(f"\n  [OK] {filename} 下载完成")
            except Exception as e:
                print(f"\n  [ERROR] 下载失败: {e}")
                print(f"  请手动下载: {url}")
                return None

    if all_exist:
        print(f"DNS-MOS 模型已存在: {dnsmos_path}")
    else:
        print(f"DNS-MOS 模型下载完成: {dnsmos_path}")

    return dnsmos_path / "sig_bak_ovr.onnx", dnsmos_path / "model_v8.onnx"


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


def calculate_sisnr(reference, enhanced):
    """计算 SI-SNR (Scale-Invariant Signal-to-Noise Ratio)"""
    enhanced = enhanced - enhanced.mean()
    reference = reference - reference.mean()

    alpha = np.sum(enhanced * reference) / (np.sum(reference**2) + 1e-8)
    target = alpha * reference
    noise = enhanced - target

    sisnr = 10 * np.log10((np.sum(target**2) + 1e-8) / (np.sum(noise**2) + 1e-8))
    return sisnr


def calculate_pesq(reference, enhanced, sr=16000):
    """计算 PESQ (Perceptual Evaluation of Speech Quality)"""
    if not PESQ_AVAILABLE:
        return None

    try:
        # PESQ 只支持 8kHz 和 16kHz
        if sr == 8000:
            mode = "nb"
        elif sr == 16000:
            mode = "wb"
        elif sr > 16000:
            mode = "wb"
            reference = librosa.resample(reference, orig_sr=sr, target_sr=16000)
            enhanced = librosa.resample(enhanced, orig_sr=sr, target_sr=16000)
            sr = 16000
        else:
            print(f"  警告: PESQ 不支持采样率 {sr} Hz")
            return None

        pesq_score = pesq(sr, reference, enhanced, mode=mode, on_error=PesqError.RETURN_VALUES)

        if pesq_score == PesqError.NO_UTTERANCES_DETECTED:
            print("  警告: PESQ 未检测到语音")
            return None

        return pesq_score
    except Exception as e:
        print(f"  警告: PESQ 计算失败: {e}")
        return None


def calculate_dnsmos(enhanced, sr=16000, dnsmos_model=None):
    """计算 DNS-MOS (Deep Noise Suppression Mean Opinion Score)"""
    if not DNSMOS_AVAILABLE or dnsmos_model is None:
        return None

    try:
        # DNS-MOS 需要 16kHz
        if sr != 16000:
            enhanced = librosa.resample(enhanced, orig_sr=sr, target_sr=16000)
            sr = 16000

        with torch.no_grad():
            scores = dnsmos_model(enhanced, sr)

        # 返回主要指标
        return {
            "OVRL": scores["OVRL"].item(),
            "SIG": scores["SIG"].item(),
            "BAK": scores["BAK"].item(),
            "P808_MOS": scores["P808_MOS"].item(),
        }
    except Exception as e:
        print(f"  警告: DNS-MOS 计算失败: {e}")
        return None


def process_audio(model, audio_path, output_path=None, device="cpu", clean_audio=None, dnsmos_model=None):
    """处理单个音频文件，可选计算评估指标"""
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

    # 计算评估指标（如果提供了干净音频）
    metrics = {}
    if clean_audio is not None:
        print(f"\n  计算评估指标...")

        # 确保长度一致
        min_len = min(len(clean_audio), len(enhanced_audio))
        clean_audio = clean_audio[:min_len]
        enhanced_audio_eval = enhanced_audio[:min_len]

        # SI-SNR
        sisnr = calculate_sisnr(clean_audio, enhanced_audio_eval)
        metrics["SI-SNR"] = sisnr
        print(f"    SI-SNR: {sisnr:.2f} dB")

        # PESQ
        pesq_score = calculate_pesq(clean_audio, enhanced_audio_eval, sr)
        if pesq_score is not None:
            metrics["PESQ"] = pesq_score
            print(f"    PESQ: {pesq_score:.3f}")

        # DNS-MOS
        dnsmos_scores = calculate_dnsmos(enhanced_audio_eval, sr, dnsmos_model)
        if dnsmos_scores is not None:
            metrics.update(dnsmos_scores)
            print(f"    DNS-MOS OVRL: {dnsmos_scores['OVRL']:.3f}")
            print(f"    DNS-MOS SIG: {dnsmos_scores['SIG']:.3f}")
            print(f"    DNS-MOS BAK: {dnsmos_scores['BAK']:.3f}")
            print(f"    DNS-MOS P808: {dnsmos_scores['P808_MOS']:.3f}")

    print()

    return enhanced_audio, sr, metrics


def process_audio_wrapper(args):
    """多线程处理的包装函数"""
    audio_file, output_file, model, device, clean_audio_path, dnsmos_model, print_lock, index, total = args

    # 加载干净音频（如果需要评估）
    clean_audio = None
    if clean_audio_path and clean_audio_path.exists():
        try:
            clean_audio, _ = sf.read(clean_audio_path, dtype="float32")
        except Exception as e:
            with print_lock:
                print(f"  警告: 无法加载干净音频 {clean_audio_path}: {e}")

    try:
        with print_lock:
            print(f"[{index}/{total}]")

        result = process_audio(model, str(audio_file), str(output_file), device, clean_audio, dnsmos_model)

        if result is not None:
            _, _, metrics = result
            if metrics:
                metrics["filename"] = audio_file.name
            return True, metrics
        return False, None
    except Exception as e:
        with print_lock:
            print(f"  错误: 处理 {audio_file.name} 失败: {e}\n")
        return False, None


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

  # 批量处理并计算评估指标（需要提供干净音频目录）
  python test_model.py noisy_dir/ -o output_dir/ --clean_dir clean_dir/ --evaluate

  # 使用 4 个线程加速批量处理
  python test_model.py noisy_dir/ -o output_dir/ --num_threads 4

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
    parser.add_argument(
        "--clean_dir",
        type=str,
        default=None,
        help="干净音频目录路径（用于计算评估指标，文件名需与输入音频对应）"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="启用评估指标计算（需要 --clean_dir）"
    )
    parser.add_argument(
        "--dnsmos_primary",
        type=str,
        default="./DNSMOS/sig_bak_ovr.onnx",
        help="DNS-MOS 主模型路径"
    )
    parser.add_argument(
        "--dnsmos_p808",
        type=str,
        default="./DNSMOS/model_v8.onnx",
        help="DNS-MOS P808 模型路径"
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=8,
        help="批量处理时使用的线程数（默认: 8）"
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

    # 加载 DNS-MOS 模型（如果需要评估）
    dnsmos_model = None
    if args.evaluate and DNSMOS_AVAILABLE:
        # 检查模型文件是否存在，不存在则自动下载
        if not Path(args.dnsmos_primary).exists() or not Path(args.dnsmos_p808).exists():
            print("\nDNS-MOS 模型文件不存在，正在自动下载...")
            result = download_dnsmos_models()
            if result is None:
                print("警告: DNS-MOS 模型下载失败，将跳过 DNS-MOS 计算\n")
            else:
                args.dnsmos_primary, args.dnsmos_p808 = result
                print()

        if Path(args.dnsmos_primary).exists() and Path(args.dnsmos_p808).exists():
            print("正在加载 DNS-MOS 模型...")
            use_gpu = args.device == "cuda"
            dnsmos_model = DNSMOS_local(
                args.dnsmos_primary,
                args.dnsmos_p808,
                use_gpu=use_gpu,
                convert_to_torch=False,
            )
            print("DNS-MOS 模型加载完成！\n")
        else:
            print("警告: DNS-MOS 模型文件不存在，将跳过 DNS-MOS 计算")
            print(f"  需要: {args.dnsmos_primary}")
            print(f"  需要: {args.dnsmos_p808}\n")

    audio_path = Path(args.audio_path)

    # 单文件处理
    if audio_path.is_file() and audio_path.suffix.lower() == ".wav":
        output_path = args.output

        # 加载干净音频（如果需要评估）
        clean_audio = None
        if args.evaluate and args.clean_dir:
            clean_path = Path(args.clean_dir) / audio_path.name
            if clean_path.exists():
                clean_audio, _ = sf.read(clean_path, dtype="float32")
                print(f"已加载干净音频: {clean_path}\n")
            else:
                print(f"警告: 未找到对应的干净音频: {clean_path}\n")

        result = process_audio(model, str(audio_path), output_path, args.device, clean_audio, dnsmos_model)
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

        print(f"找到 {len(audio_files)} 个音频文件")
        print(f"使用 {args.num_threads} 个线程进行处理\n")

        # 准备 CSV 报告
        csv_path = output_dir / "evaluation_results.csv"
        all_metrics = []

        # 准备任务参数
        print_lock = Lock()
        tasks = []
        for i, audio_file in enumerate(audio_files, 1):
            output_file = output_dir / f"{audio_file.stem}_enhanced.wav"
            clean_audio_path = None
            if args.evaluate and args.clean_dir:
                clean_audio_path = Path(args.clean_dir) / audio_file.name

            tasks.append((
                audio_file,
                output_file,
                model,
                args.device,
                clean_audio_path,
                dnsmos_model,
                print_lock,
                i,
                len(audio_files)
            ))

        # 多线程处理
        success_count = 0
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            futures = [executor.submit(process_audio_wrapper, task) for task in tasks]

            for future in as_completed(futures):
                try:
                    success, metrics = future.result()
                    if success:
                        success_count += 1
                        if metrics:
                            all_metrics.append(metrics)
                except Exception as e:
                    print(f"  错误: 处理任务失败: {e}\n")

        print(f"\n批量处理完成: {success_count}/{len(audio_files)} 个文件成功处理")
        print(f"输出目录: {output_dir}")

        # 保存 CSV 报告
        if all_metrics:
            print(f"\n正在生成评估报告...")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                fieldnames = ["filename", "SI-SNR", "PESQ", "OVRL", "SIG", "BAK", "P808_MOS"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for metrics in all_metrics:
                    row = {
                        "filename": metrics.get("filename", ""),
                        "SI-SNR": f"{metrics.get('SI-SNR', ''):.2f}" if "SI-SNR" in metrics else "",
                        "PESQ": f"{metrics.get('PESQ', ''):.3f}" if "PESQ" in metrics else "",
                        "OVRL": f"{metrics.get('OVRL', ''):.3f}" if "OVRL" in metrics else "",
                        "SIG": f"{metrics.get('SIG', ''):.3f}" if "SIG" in metrics else "",
                        "BAK": f"{metrics.get('BAK', ''):.3f}" if "BAK" in metrics else "",
                        "P808_MOS": f"{metrics.get('P808_MOS', ''):.3f}" if "P808_MOS" in metrics else "",
                    }
                    writer.writerow(row)

            print(f"评估报告已保存: {csv_path}")

            # 计算平均值
            avg_metrics = {}
            for key in ["SI-SNR", "PESQ", "OVRL", "SIG", "BAK", "P808_MOS"]:
                values = [m[key] for m in all_metrics if key in m]
                if values:
                    avg_metrics[key] = np.mean(values)

            # 保存平均值报告
            summary_path = output_dir / "evaluation_summary.txt"
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write("GTCRN-Micro 评估结果汇总\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"处理文件数: {len(all_metrics)}\n\n")
                f.write("平均指标:\n")
                for key, value in avg_metrics.items():
                    f.write(f"  {key}: {value:.3f}\n")

            print(f"评估汇总已保存: {summary_path}")
            print(f"\n平均指标:")
            for key, value in avg_metrics.items():
                print(f"  {key}: {value:.3f}")

    else:
        print(f"错误: 路径不存在或不是有效的音频文件/目录: {audio_path}")
        sys.exit(1)

    print("\n测试完成！")


if __name__ == "__main__":
    main()

