# GTCRN-Micro 模型测试指南

本文档详细介绍如何测试 GTCRN-Micro 网络的效果，包括快速测试、批量推理和性能评估。

---

## 目录

1. [环境准备](#1-环境准备)
2. [快速测试（单个音频文件）](#2-快速测试单个音频文件)
3. [批量推理](#3-批量推理)
4. [性能评估](#4-性能评估)
5. [测试示例脚本](#5-测试示例脚本)
6. [常见问题](#6-常见问题)

---

## 1. 环境准备

### 1.1 安装依赖

确保已安装项目依赖：

```bash
# 使用 uv（推荐）
uv sync

# 或使用 pip
pip install torch librosa soundfile omegaconf tqdm pesq pystoi
```

### 1.2 检查预训练模型

确认预训练模型文件存在：

```bash
ls -lh gtcrn_micro/ckpts/best_model_dns3.tar
```

如果不存在，需要先下载或训练模型。

### 1.3 准备测试数据

**选项 A：使用项目提供的示例音频**

项目已包含示例音频文件：
- `gtcrn_micro/examples/gtcrn_micro/noisy1.wav` ~ `noisy5.wav`（带噪音频）
- `gtcrn_micro/examples/gtcrn_micro/enh1.wav` ~ `enh5.wav`（增强后的音频示例）

**选项 B：准备自己的测试音频**

- **格式要求**：WAV 格式，单声道，16kHz 采样率（其他采样率会自动重采样）
- **放置位置**：可以放在任意目录，或创建专门的测试目录

---

## 2. 快速测试（单个音频文件）

### 2.1 使用 Python 脚本快速测试

创建一个简单的测试脚本 `test_single_audio.py`：

```python
import torch
import soundfile as sf
import numpy as np
from gtcrn_micro.models.gtcrn_micro import GTCRNMicro

# 1. 加载模型
model = GTCRNMicro().eval()
checkpoint = torch.load("gtcrn_micro/ckpts/best_model_dns3.tar", map_location="cpu")
state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
model.load_state_dict(state_dict, strict=False)

# 2. 加载音频文件
audio_path = "gtcrn_micro/examples/gtcrn_micro/noisy1.wav"  # 修改为你的音频路径
noisy_audio, sr = sf.read(audio_path, dtype="float32")

# 如果采样率不是16kHz，需要重采样
if sr != 16000:
    import librosa
    noisy_audio = librosa.resample(noisy_audio, orig_sr=sr, target_sr=16000)
    sr = 16000

print(f"音频长度: {len(noisy_audio) / sr:.2f} 秒")
print(f"采样率: {sr} Hz")

# 3. 转换为频域（STFT）
input_stft = torch.stft(
    torch.from_numpy(noisy_audio),
    n_fft=512,
    hop_length=256,
    win_length=512,
    window=torch.hann_window(512).pow(0.5),
    return_complex=False,
)

# 4. 模型推理
with torch.no_grad():
    output_stft = model(input_stft.unsqueeze(0))[0]  # 添加batch维度

# 5. 转换回时域（ISTFT）
output_stft_complex = torch.view_as_complex(output_stft.contiguous())
enhanced_audio = torch.istft(
    output_stft_complex,
    n_fft=512,
    hop_length=256,
    win_length=512,
    window=torch.hann_window(512).pow(0.5),
    return_complex=False,
).numpy()

# 6. 保存增强后的音频
output_path = "enhanced_output.wav"
sf.write(output_path, enhanced_audio, sr)
print(f"增强后的音频已保存到: {output_path}")

# 7. 打印一些统计信息
print(f"\n原始音频统计:")
print(f"  最大值: {np.abs(noisy_audio).max():.4f}")
print(f"  均方根: {np.sqrt(np.mean(noisy_audio**2)):.4f}")
print(f"\n增强音频统计:")
print(f"  最大值: {np.abs(enhanced_audio).max():.4f}")
print(f"  均方根: {np.sqrt(np.mean(enhanced_audio**2)):.4f}")
```

**运行测试：**

```bash
python test_single_audio.py
```

### 2.2 使用命令行快速测试

你也可以直接在 Python 交互环境中快速测试：

```bash
uv run python
```

```python
import torch
import soundfile as sf
from gtcrn_micro.models.gtcrn_micro import GTCRNMicro

# 加载模型
model = GTCRNMicro().eval()
ckpt = torch.load("gtcrn_micro/ckpts/best_model_dns3.tar", map_location="cpu")
model.load_state_dict(ckpt["model"], strict=False)

# 加载和测试音频
audio, sr = sf.read("gtcrn_micro/examples/gtcrn_micro/noisy1.wav")
input_stft = torch.stft(torch.from_numpy(audio), 512, 256, 512, 
                        torch.hann_window(512).pow(0.5), return_complex=False)
with torch.no_grad():
    output = model(input_stft.unsqueeze(0))[0]
output_complex = torch.view_as_complex(output.contiguous())
enhanced = torch.istft(output_complex, 512, 256, 512, 
                       torch.hann_window(512).pow(0.5), return_complex=False)
sf.write("test_output.wav", enhanced.numpy(), 16000)
print("测试完成！")
```

---

## 3. 批量推理

对于多个音频文件的批量处理，使用项目提供的推理脚本：

### 3.1 配置推理参数

编辑 `gtcrn_micro/conf/cfg_infer.yaml`：

```yaml
test_dataset:
  # 带噪音频目录
  noisy_dir: gtcrn_micro/data/DNS3_test/noisy
  # 干净音频目录（用于有参考评估，可选）
  clean_dir: gtcrn_micro/data/DNS3_test/clean

network:
  exp_path: gtcrn_micro/data/exp_gtcrn_micro2_2025-12-23-14h45m
  config: ${network.exp_path}/config.yaml
  ckpt_name: best_model_157
  checkpoint: ${network.exp_path}/checkpoints/${network.ckpt_name}.tar
  enh_folder: ${network.exp_path}/${network.ckpt_name}/enhanced
```

**注意**：如果使用默认的预训练模型，需要修改配置文件：

```yaml
network:
  checkpoint: gtcrn_micro/ckpts/best_model_dns3.tar
  config: gtcrn_micro/conf/cfg_train_DNS3.yaml  # 或相应的配置文件
  enh_folder: ./enhanced_outputs
```

### 3.2 运行批量推理

```bash
# 使用默认配置
uv run python gtcrn_micro/infer.py

# 指定配置文件
uv run python gtcrn_micro/infer.py --config gtcrn_micro/conf/cfg_infer.yaml

# 指定 GPU 设备（如果有 GPU）
uv run python gtcrn_micro/infer.py --device 0
```

**输出结果：**
- 增强后的音频文件保存在 `enh_folder` 目录
- 生成 `inf.scp` 和 `ref.scp` 文件，用于后续评估

---

## 4. 性能评估

GTCRN-Micro 支持两种类型的评估指标：

### 4.1 有参考评估（Intrusive Metrics）

需要干净音频作为参考，计算以下指标：
- **SDR** (Signal-to-Distortion Ratio)：信噪比
- **SI-SNR** (Scale-Invariant SNR)：尺度不变信噪比
- **PESQ** (Perceptual Evaluation of Speech Quality)：感知语音质量评估
- **STOI** (Short-Time Objective Intelligibility)：短时客观可懂度

**运行有参考评估：**

```bash
# 先运行推理生成增强音频
uv run python gtcrn_micro/infer.py --config gtcrn_micro/conf/cfg_infer.yaml

# 然后运行评估
uv run python gtcrn_micro/eval/evaluate.py \
    --metric intrusive \
    --config gtcrn_micro/conf/cfg_infer.yaml
```

**查看评估结果：**

评估结果保存在 `{enh_folder}/scoring_intrusive/` 目录下，查看汇总文件：

```bash
cat {enh_folder}/scoring_intrusive/scores.txt
```

### 4.2 无参考评估（Non-Intrusive Metrics）

不需要干净音频，使用 DNSMOS 模型进行评估：
- **DNSMOS-P.808**：感知质量评分
- **BAK** (Background Activity)：背景噪声评分
- **SIG** (Signal)：信号质量评分
- **OVRL** (Overall)：总体质量评分

**运行无参考评估：**

```bash
# 先运行推理生成增强音频
uv run python gtcrn_micro/infer.py --config gtcrn_micro/conf/cfg_infer.yaml

# 然后运行 DNSMOS 评估
uv run python gtcrn_micro/eval/evaluate.py \
    --metric dnsmos \
    --config gtcrn_micro/conf/cfg_infer.yaml \
    --device 0
```

**注意**：DNSMOS 评估需要 GPU 支持，且需要确保 DNSMOS 模型文件存在。

### 4.3 评估结果解读

**有参考指标（越高越好）：**
- **SDR**：> 10 dB 表示降噪效果良好
- **SI-SNR**：> 9 dB 表示信号质量改善明显
- **PESQ**：范围 1.0-4.5，> 2.0 表示质量可接受
- **STOI**：范围 0-1，> 0.8 表示可懂度高

**无参考指标（越高越好）：**
- **DNSMOS-P.808**：> 3.0 表示质量良好
- **BAK**：> 3.5 表示背景噪声较少
- **SIG**：> 3.0 表示信号质量良好
- **OVRL**：> 2.5 表示总体质量可接受

**GTCRN-Micro 典型性能（参考值）：**
- SDR: 10.41 dB
- SI-SNR: 9.85 dB
- PESQ: 1.98
- STOI: 0.85
- DNSMOS-P.808: 3.25
- BAK: 3.60
- SIG: 2.99
- OVRL: 2.58

---

## 5. 测试示例脚本

### 5.1 完整测试脚本（推荐）

创建 `test_model.py`：

```python
#!/usr/bin/env python3
"""
GTCRN-Micro 模型测试脚本
支持单文件测试和批量测试
"""
import argparse
import os
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
    print(f"加载模型: {checkpoint_path}")
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
        print(f"警告: 缺少参数: {missing}")
    if unexpected:
        print(f"警告: 未使用参数: {unexpected}")
    
    print("模型加载完成")
    return model


def process_audio(model, audio_path, output_path=None, device="cpu"):
    """处理单个音频文件"""
    print(f"\n处理音频: {audio_path}")
    
    # 加载音频
    audio, sr = sf.read(audio_path, dtype="float32")
    
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
    print(f"\n  原始音频:")
    print(f"    RMS: {np.sqrt(np.mean(audio**2)):.4f}")
    print(f"    最大值: {np.abs(audio).max():.4f}")
    print(f"  增强音频:")
    print(f"    RMS: {np.sqrt(np.mean(enhanced_audio**2)):.4f}")
    print(f"    最大值: {np.abs(enhanced_audio).max():.4f}")
    
    return enhanced_audio, sr


def main():
    parser = argparse.ArgumentParser(description="GTCRN-Micro 模型测试工具")
    parser.add_argument(
        "audio_path",
        type=str,
        help="输入音频文件路径（或目录，批量处理）"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="输出音频文件路径（单文件）或目录（批量）"
    )
    parser.add_argument(
        "-c", "--checkpoint",
        type=str,
        default="gtcrn_micro/ckpts/best_model_dns3.tar",
        help="模型检查点路径"
    )
    parser.add_argument(
        "-d", "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="计算设备"
    )
    
    args = parser.parse_args()
    
    # 加载模型
    model = load_model(args.checkpoint)
    
    audio_path = Path(args.audio_path)
    
    # 单文件处理
    if audio_path.is_file():
        output_path = args.output
        process_audio(model, str(audio_path), output_path, args.device)
    
    # 批量处理
    elif audio_path.is_dir():
        output_dir = Path(args.output) if args.output else audio_path / "enhanced"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        audio_files = sorted(audio_path.glob("*.wav"))
        print(f"\n找到 {len(audio_files)} 个音频文件")
        
        for audio_file in audio_files:
            output_file = output_dir / f"{audio_file.stem}_enhanced.wav"
            try:
                process_audio(model, str(audio_file), str(output_file), args.device)
            except Exception as e:
                print(f"  错误: 处理 {audio_file} 失败: {e}")
    
    else:
        print(f"错误: 路径不存在: {audio_path}")
        sys.exit(1)
    
    print("\n测试完成！")


if __name__ == "__main__":
    main()
```

**使用方法：**

```bash
# 测试单个文件
uv run python test_model.py noisy_audio.wav -o enhanced.wav

# 批量处理目录中的所有 wav 文件
uv run python test_model.py input_dir/ -o output_dir/

# 使用 GPU（如果有）
uv run python test_model.py noisy_audio.wav -d cuda
```

### 5.2 简单测试脚本（最小示例）

创建 `simple_test.py`：

```python
import torch
import soundfile as sf
from gtcrn_micro.models.gtcrn_micro import GTCRNMicro

# 加载模型
model = GTCRNMicro().eval()
ckpt = torch.load("gtcrn_micro/ckpts/best_model_dns3.tar", map_location="cpu")
model.load_state_dict(ckpt["model"], strict=False)

# 读取音频
audio, sr = sf.read("gtcrn_micro/examples/gtcrn_micro/noisy1.wav")

# STFT
stft = torch.stft(torch.from_numpy(audio), 512, 256, 512, 
                  torch.hann_window(512).pow(0.5), return_complex=False)

# 推理
with torch.no_grad():
    out = model(stft.unsqueeze(0))[0]

# ISTFT
out_complex = torch.view_as_complex(out.contiguous())
enhanced = torch.istft(out_complex, 512, 256, 512, 
                       torch.hann_window(512).pow(0.5), return_complex=False)

# 保存
sf.write("output.wav", enhanced.numpy(), 16000)
print("完成！")
```

---

## 6. 常见问题

### 6.1 模型加载失败

**问题**：`RuntimeError: Error(s) in loading state_dict`

**解决方法**：
- 检查模型文件路径是否正确
- 确保模型文件完整（文件大小应该约几 MB）
- 尝试使用 `strict=False` 参数加载

### 6.2 音频采样率不匹配

**问题**：模型期望 16kHz，但输入是其他采样率

**解决方法**：
- 脚本会自动重采样，但也可以手动重采样：
  ```python
  import librosa
  audio_16k = librosa.resample(audio, orig_sr=original_sr, target_sr=16000)
  ```

### 6.3 内存不足

**问题**：处理长音频时内存不足

**解决方法**：
- 对于很长的音频，可以分段处理：
  ```python
  # 分段处理（每段约10秒）
  segment_length = 160000  # 10秒 @ 16kHz
  for i in range(0, len(audio), segment_length):
      segment = audio[i:i+segment_length]
      # 处理 segment
  ```

### 6.4 输出音频有噪声或失真

**可能原因**：
- 输入音频质量太差
- 模型不适用于该类型噪声
- 音频格式问题

**检查方法**：
- 确认输入音频是单声道 WAV 格式
- 检查音频是否过度削波（clipping）
- 尝试不同的音频文件

### 6.5 评估指标计算失败

**PESQ/STOI 计算失败：**
- 确保输入和参考音频长度一致
- 检查音频采样率是否为 16kHz

**DNSMOS 评估失败：**
- 需要 GPU 支持
- 确保 DNSMOS 模型文件存在

---

## 7. 快速开始示例

**最简单的测试流程：**

```bash
# 1. 进入项目目录
cd GTCRN-Micro

# 2. 使用示例音频快速测试
uv run python -c "
import torch, soundfile as sf
from gtcrn_micro.models.gtcrn_micro import GTCRNMicro

model = GTCRNMicro().eval()
model.load_state_dict(torch.load('gtcrn_micro/ckpts/best_model_dns3.tar', map_location='cpu')['model'], strict=False)

audio, sr = sf.read('gtcrn_micro/examples/gtcrn_micro/noisy1.wav')
stft = torch.stft(torch.from_numpy(audio), 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)
with torch.no_grad():
    out = model(stft.unsqueeze(0))[0]
out_complex = torch.view_as_complex(out.contiguous())
enhanced = torch.istft(out_complex, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)
sf.write('test_output.wav', enhanced.numpy(), 16000)
print('测试完成！输出文件: test_output.wav')
"
```

---

## 8. 测试清单

**基本功能测试：**
- [ ] 模型可以正常加载
- [ ] 单个音频文件可以成功处理
- [ ] 输出音频文件可以正常播放
- [ ] 输出音频长度与输入基本一致

**质量检查：**
- [ ] 对比原始和增强音频，主观听感有改善
- [ ] 输出音频无明显失真或伪影
- [ ] 音量水平合理（无过度放大或衰减）

**性能测试：**
- [ ] 测试不同长度的音频（短音频 1-5秒，长音频 >10秒）
- [ ] 测试不同采样率的音频（会自动重采样）
- [ ] 批量处理多个文件

**评估指标（可选）：**
- [ ] 使用有参考指标评估（SDR, PESQ, STOI）
- [ ] 使用无参考指标评估（DNSMOS）

---

## 9. 下一步

测试完成后，你可能想要：

1. **优化模型**：根据测试结果调整训练参数
2. **部署模型**：将模型转换为 ONNX 或 TFLite 格式用于边缘设备
3. **性能分析**：使用 `gtcrn_micro/utils/memory_profile.py` 分析内存占用
4. **流式处理**：测试流式处理版本（如果可用）

---

**文档版本**：v1.0  
**最后更新**：2026年1月8日  
**联系方式**：如有问题，请查看项目 README 或提交 Issue

