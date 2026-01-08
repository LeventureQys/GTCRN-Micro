# 音频降噪评估指标说明文档

本文档详细说明了 GTCRN-Micro 模型测试脚本中使用的音频质量评估指标，方便进行不同降噪模型之间的对比分析。

## 目录
- [评估指标概述](#评估指标概述)
- [指标详细说明](#指标详细说明)
- [输出文件说明](#输出文件说明)
- [使用方法](#使用方法)
- [指标对比建议](#指标对比建议)

---

## 评估指标概述

本测试脚本支持以下 6 个音频质量评估指标：

| 指标 | 类型 | 取值范围 | 越高越好 | 说明 |
|------|------|----------|----------|------|
| **SI-SNR** | 侵入式 | -∞ ~ +∞ dB | ✓ | 尺度不变信噪比 |
| **PESQ** | 侵入式 | -0.5 ~ 4.5 | ✓ | 感知语音质量评估 |
| **OVRL** | 非侵入式 | 1.0 ~ 5.0 | ✓ | DNS-MOS 整体质量 |
| **SIG** | 非侵入式 | 1.0 ~ 5.0 | ✓ | DNS-MOS 信号质量 |
| **BAK** | 非侵入式 | 1.0 ~ 5.0 | ✓ | DNS-MOS 背景噪声质量 |
| **P808_MOS** | 非侵入式 | 1.0 ~ 5.0 | ✓ | DNS-MOS P.808 主观评分 |

**指标类型说明：**
- **侵入式（Intrusive）**：需要干净的参考音频进行对比计算
- **非侵入式（Non-intrusive）**：仅需增强后的音频，无需参考音频

---

## 指标详细说明

### 1. SI-SNR (Scale-Invariant Signal-to-Noise Ratio)
**尺度不变信噪比**

#### 定义
SI-SNR 是一种改进的信噪比指标，对音频的幅度缩放不敏感，更适合评估语音增强效果。

#### 计算公式
```
SI-SNR = 10 * log10(||s_target||² / ||e_noise||²)
```
其中：
- `s_target = <s', s> / ||s||² * s`（目标信号投影）
- `e_noise = s' - s_target`（残余噪声）
- `s` 是干净参考信号，`s'` 是增强后信号

#### 取值范围
- 理论范围：-∞ ~ +∞ dB
- 实际范围：通常在 -10 dB ~ 30 dB
- **典型值**：
  - < 0 dB：增强效果差，可能引入失真
  - 0 ~ 10 dB：基本降噪效果
  - 10 ~ 20 dB：良好降噪效果
  - \> 20 dB：优秀降噪效果

#### 优点
- 对音频幅度缩放不敏感
- 计算简单快速
- 适合评估语音分离和增强任务

#### 局限性
- 不能完全反映人耳感知质量
- 对时域对齐敏感

---

### 2. PESQ (Perceptual Evaluation of Speech Quality)
**感知语音质量评估**

#### 定义
PESQ 是 ITU-T P.862 标准定义的客观语音质量评估指标，模拟人耳对语音质量的感知。

#### 计算方法
- 基于心理声学模型
- 比较参考信号和增强信号的感知差异
- 考虑频域掩蔽效应和时域对齐

#### 取值范围
- 标准范围：-0.5 ~ 4.5
- 实际范围：通常在 1.0 ~ 4.5
- **典型值**：
  - < 2.0：质量差
  - 2.0 ~ 3.0：一般质量
  - 3.0 ~ 3.5：良好质量
  - 3.5 ~ 4.0：优秀质量
  - \> 4.0：接近完美

#### 模式
- **NB (Narrowband)**：8 kHz 采样率，电话语音
- **WB (Wideband)**：16 kHz 采样率，高清语音

#### 优点
- 与人类主观评分高度相关
- 国际标准，广泛使用
- 考虑了人耳感知特性

#### 局限性
- 仅支持 8 kHz 和 16 kHz 采样率
- 对严重失真的音频可能失效
- 计算相对较慢

---

### 3. DNS-MOS (Deep Noise Suppression Mean Opinion Score)
**深度噪声抑制平均主观评分**

#### 定义
DNS-MOS 是微软开发的基于深度学习的非侵入式语音质量评估系统，无需参考音频即可预测主观评分。

#### 子指标说明

##### 3.1 OVRL (Overall Quality)
**整体质量评分**
- 综合评估语音的整体质量
- 考虑信号质量和背景噪声
- **典型值**：
  - < 2.5：质量差
  - 2.5 ~ 3.5：一般质量
  - 3.5 ~ 4.0：良好质量
  - \> 4.0：优秀质量

##### 3.2 SIG (Signal Quality)
**信号质量评分**
- 评估语音信号本身的清晰度和失真程度
- 不考虑背景噪声
- **典型值**：
  - < 3.0：信号失真明显
  - 3.0 ~ 3.5：信号质量一般
  - 3.5 ~ 4.0：信号质量良好
  - \> 4.0：信号质量优秀

##### 3.3 BAK (Background Quality)
**背景噪声质量评分**
- 评估背景噪声的抑制程度
- 分数越高表示噪声越少
- **典型值**：
  - < 3.0：噪声明显
  - 3.0 ~ 3.5：噪声较少
  - 3.5 ~ 4.0：噪声很少
  - \> 4.0：几乎无噪声

##### 3.4 P808_MOS (P.808 MOS)
**ITU-T P.808 主观评分预测**
- 基于 ITU-T P.808 标准的主观评分预测
- 模拟真实听众的主观感受
- **典型值**：
  - < 2.5：不可接受
  - 2.5 ~ 3.0：勉强可接受
  - 3.0 ~ 3.5：可接受
  - 3.5 ~ 4.0：良好
  - \> 4.0：优秀

#### 优点
- 无需参考音频（非侵入式）
- 基于深度学习，预测准确
- 提供多维度评估
- 与人类主观评分高度相关

#### 局限性
- 需要下载 ONNX 模型文件
- 计算相对较慢
- 仅支持 16 kHz 采样率

---

## 输出文件说明

### 1. evaluation_results.csv
**详细评估结果表格**

包含每个音频文件的所有评估指标：

```csv
filename,SI-SNR,PESQ,OVRL,SIG,BAK,P808_MOS
noisy1.wav,15.23,3.245,3.856,3.912,4.123,3.789
noisy2.wav,18.45,3.567,4.012,4.234,4.345,4.001
...
```

**字段说明：**
- `filename`：音频文件名
- `SI-SNR`：SI-SNR 值（dB），保留 2 位小数
- `PESQ`：PESQ 分数，保留 3 位小数
- `OVRL`：DNS-MOS 整体质量，保留 3 位小数
- `SIG`：DNS-MOS 信号质量，保留 3 位小数
- `BAK`：DNS-MOS 背景质量，保留 3 位小数
- `P808_MOS`：DNS-MOS P.808 评分，保留 3 位小数

### 2. evaluation_summary.txt
**评估结果汇总**

包含所有音频的平均指标：

```
GTCRN-Micro 评估结果汇总
==================================================

处理文件数: 100

平均指标:
  SI-SNR: 16.234
  PESQ: 3.456
  OVRL: 3.912
  SIG: 4.023
  BAK: 4.156
  P808_MOS: 3.867
```

---

## 使用方法

### 环境准备

1. **安装必要的 Python 包：**
```bash
pip install pesq librosa soundfile numpy torch
```

2. **安装 ESPnet2（用于 DNS-MOS）：**
```bash
pip install espnet espnet_model_zoo
```

3. **DNS-MOS 模型文件（自动下载）：**

**好消息！** 脚本会在首次使用 `--evaluate` 参数时自动下载 DNS-MOS 模型文件到 `./DNSMOS/` 目录。

如果自动下载失败，可以手动下载：
```bash
# 创建 DNSMOS 目录
mkdir DNSMOS
cd DNSMOS

# 下载模型文件
wget https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx
wget https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/model_v8.onnx
```

或者在 Windows 上使用浏览器下载：
- https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx
- https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/model_v8.onnx

将下载的文件放到 `DNSMOS/` 目录下。

### 基本使用

#### 1. 单文件测试（不计算指标）
```bash
python test_model.py noisy_audio.wav -o enhanced.wav
```

#### 2. 批量处理（不计算指标）
```bash
python test_model.py noisy_dir/ -o output_dir/
```

#### 3. 批量处理并计算评估指标
```bash
python test_model.py noisy_dir/ -o output_dir/ --clean_dir clean_dir/ --evaluate
```

**参数说明：**
- `noisy_dir/`：包含带噪音频的目录
- `output_dir/`：输出增强音频和评估报告的目录
- `clean_dir/`：包含干净参考音频的目录（文件名需与 noisy_dir 对应）
- `--evaluate`：启用评估指标计算

#### 4. 使用多线程加速批量处理
```bash
python test_model.py noisy_dir/ -o output_dir/ --clean_dir clean_dir/ --evaluate --num_threads 4
```

**参数说明：**
- `--num_threads 4`：使用 4 个线程并行处理（默认为 8）
- 建议根据 CPU 核心数设置线程数
- 多线程可以显著加快批量处理速度

#### 5. 使用 GPU 加速
```bash
python test_model.py noisy_dir/ -o output_dir/ --clean_dir clean_dir/ --evaluate -d cuda
```

#### 6. 自定义 DNS-MOS 模型路径
```bash
python test_model.py noisy_dir/ -o output_dir/ --clean_dir clean_dir/ --evaluate \
    --dnsmos_primary ./models/sig_bak_ovr.onnx \
    --dnsmos_p808 ./models/model_v8.onnx
```

### 目录结构要求

```
project/
├── noisy_dir/              # 带噪音频目录
│   ├── sample1.wav
│   ├── sample2.wav
│   └── ...
├── clean_dir/              # 干净音频目录（文件名需对应）
│   ├── sample1.wav
│   ├── sample2.wav
│   └── ...
└── output_dir/             # 输出目录（自动创建）
    ├── sample1_enhanced.wav
    ├── sample2_enhanced.wav
    ├── evaluation_results.csv      # 详细评估结果
    └── evaluation_summary.txt      # 评估汇总
```

**重要提示：**
- `clean_dir` 中的文件名必须与 `noisy_dir` 中的文件名完全一致
- 所有音频文件必须是 `.wav` 格式
- 建议使用 16 kHz 采样率（脚本会自动重采样）

---

## 指标对比建议

### 1. 对比不同降噪模型

当对比多个降噪模型时，建议关注以下指标组合：

#### 方案 A：全面对比（推荐）
使用所有 6 个指标进行综合评估：
- **SI-SNR**：客观信噪比改善
- **PESQ**：感知语音质量
- **OVRL**：整体主观质量
- **SIG**：信号失真程度
- **BAK**：噪声抑制效果
- **P808_MOS**：综合主观评分

#### 方案 B：快速对比
如果计算资源有限，可以只使用：
- **SI-SNR**：快速计算，反映基本降噪效果
- **PESQ**：感知质量，与主观评分相关性高
- **OVRL**：非侵入式整体质量

### 2. 创建对比表格

建议创建如下格式的对比表格：

```
| 模型 | SI-SNR (dB) | PESQ | OVRL | SIG | BAK | P808_MOS |
|------|-------------|------|------|-----|-----|----------|
| GTCRN-Micro | 16.23 | 3.456 | 3.912 | 4.023 | 4.156 | 3.867 |
| 模型 A | 14.56 | 3.234 | 3.678 | 3.789 | 3.912 | 3.567 |
| 模型 B | 17.89 | 3.678 | 4.123 | 4.234 | 4.289 | 4.012 |
```

### 3. 对比脚本示例

以下是一个简单的 Python 脚本，用于对比多个模型的评估结果：

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取多个模型的评估结果
gtcrn_results = pd.read_csv("gtcrn_output/evaluation_results.csv")
model_a_results = pd.read_csv("model_a_output/evaluation_results.csv")
model_b_results = pd.read_csv("model_b_output/evaluation_results.csv")

# 计算平均值
metrics = ["SI-SNR", "PESQ", "OVRL", "SIG", "BAK", "P808_MOS"]
comparison = pd.DataFrame({
    "GTCRN-Micro": [gtcrn_results[m].mean() for m in metrics],
    "Model A": [model_a_results[m].mean() for m in metrics],
    "Model B": [model_b_results[m].mean() for m in metrics],
}, index=metrics)

print(comparison)

# 可视化对比
comparison.T.plot(kind="bar", figsize=(12, 6))
plt.title("Model Comparison")
plt.ylabel("Score")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("model_comparison.png")
```

### 4. 指标权重建议

在综合评估时，可以考虑以下权重分配：

- **语音通信场景**：
  - PESQ: 30%
  - OVRL: 25%
  - SIG: 20%
  - BAK: 15%
  - SI-SNR: 10%

- **语音识别前处理**：
  - SI-SNR: 40%
  - PESQ: 30%
  - SIG: 20%
  - BAK: 10%

- **音频质量优先**：
  - PESQ: 35%
  - P808_MOS: 30%
  - OVRL: 20%
  - SIG: 15%

### 5. 注意事项

1. **测试集一致性**：确保所有模型使用相同的测试集
2. **采样率统一**：建议统一使用 16 kHz 采样率
3. **多次测试**：对于随机性较大的模型，建议多次测试取平均
4. **场景多样性**：测试集应包含多种噪声类型和信噪比
5. **主观评估**：客观指标不能完全替代主观听感，建议结合主观评估

---

## 常见问题

### Q1: PESQ 计算失败怎么办？
**A:** 检查以下几点：
- 确保音频采样率为 8 kHz 或 16 kHz
- 确保参考音频和增强音频长度一致
- 确保音频中包含语音内容（PESQ 无法处理纯静音）

### Q2: DNS-MOS 模型下载失败？
**A:** 可以手动从 GitHub 下载：
- https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS

### Q3: 如何解释负的 SI-SNR 值？
**A:** 负的 SI-SNR 表示增强后的音频质量比原始带噪音频更差，可能是：
- 模型过度抑制导致语音失真
- 模型引入了额外的伪影
- 参考音频和增强音频对齐问题

### Q4: 不同指标结果矛盾怎么办？
**A:** 这是正常现象，因为不同指标关注的方面不同：
- SI-SNR 关注信噪比改善
- PESQ 关注感知质量
- DNS-MOS 关注主观评分
建议综合多个指标进行评估，并结合实际应用场景选择重点指标。

---

## 参考文献

1. **SI-SNR**: Le Roux, J., et al. "SDR–half-baked or well done?." ICASSP 2019.
2. **PESQ**: ITU-T Recommendation P.862, "Perceptual evaluation of speech quality (PESQ)."
3. **DNS-MOS**: Reddy, C. K., et al. "DNSMOS: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors." ICASSP 2021.
4. **P.808**: ITU-T Recommendation P.808, "Subjective evaluation of speech quality with a crowdsourcing approach."

---

## 更新日志

- **2026-01-08**: 初始版本，支持 SI-SNR、PESQ、DNS-MOS 评估

---

## 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。
