# GTCRN-Micro降噪模型对比评估报告

## 执行摘要

本报告对比分析了 GTCRN-Micro 降噪模型的两个版本：**QAT_Version**（量化感知训练版本）和 **FullVersion**（完整精度版本）。通过对 108 个测试样本的多维度评估，发现两个版本在不同指标上各有优势，整体性能接近。

---

## 1. 测试数据集概况

### 1.1 数据集规模
- **样本总数**: 108 个音频文件
- **噪声类型**: 6 种（Fan Noise Level 1/2, Clapping, Door Knock, Keyboard Loud/Soft）
- **说话者数量**: 3 人
- **信噪比级别**: 6 个（-5dB, 0dB, +5dB, +10dB, +15dB, +20dB）
- **测试覆盖度**: 6 × 3 × 6 = 108 个完整组合

### 1.2 评估指标体系
- **SI-SNR** (Scale-Invariant Signal-to-Noise Ratio): 尺度不变信噪比，越高越好
- **PESQ** (Perceptual Evaluation of Speech Quality): 感知语音质量评估，范围 -0.5 到 4.5
- **OVRL**: 整体质量评分
- **SIG**: 信号失真评分
- **BAK**: 背景噪声评分
- **P808_MOS**: ITU-T P.808 平均意见分

---

## 2. 整体性能对比分析

### 2.1 六大指标对比

| 指标 | QAT_Version | FullVersion | 差异 | 相对变化 | 优势方 |
|------|-------------|-------------|------|----------|---------|
| **SI-SNR** | 14.371 dB | 14.401 dB | +0.030 dB | +0.21% | FullVersion |
| **PESQ** | 2.151 | 2.262 | +0.111 | +5.16% | FullVersion |
| **OVRL** | 3.473 | 3.161 | -0.312 | -8.98% | QAT_Version |
| **SIG** | 3.774 | 3.515 | -0.259 | -6.86% | QAT_Version |
| **BAK** | 4.274 | 3.944 | -0.330 | -7.72% | QAT_Version |
| **P808_MOS** | 3.473 | 3.731 | +0.258 | +7.43% | FullVersion |

### 2.2 关键发现

**FullVersion 优势领域**:
- **SI-SNR**: 在信噪比提升上略胜一筹（+0.03 dB）
- **PESQ**: 感知质量显著更优（+5.16%）
- **P808_MOS**: 主观评分明显更高（+7.43%），表明用户体验更好

**QAT_Version 优势领域**:
- **OVRL**: 整体质量评分更高（-8.98%意味着 QAT 更优）
- **SIG**: 信号失真控制更好（-6.86%）
- **BAK**: 背景噪声抑制更强（-7.72%）

**矛盾现象分析**:
QAT_Version 在 OVRL/SIG/BAK 三项指标上表现更好，但 P808_MOS（主观评分）却低于 FullVersion。这可能表明：
1. QAT 版本在客观指标上优化更激进，但可能引入了人耳敏感的伪影
2. FullVersion 在感知质量上更自然，尽管某些客观指标略低

---

## 3. 按噪声类型的详细分析

### 3.1 SI-SNR 性能对比（按噪声类型）

| 噪声类型 | QAT_Version | FullVersion | 差异 (dB) | 优势方 | 样本数 |
|----------|-------------|-------------|---------|---------|--------|
| **Fan Noise Level 1** | 15.94 | 16.32 | +0.39 | FullVersion | 18 |
| **Fan Noise Level 2** | 16.29 | 16.39 | +0.10 | FullVersion | 18 |
| **Clapping** | 14.18 | 14.73 | +0.54 | FullVersion | 18 |
| **Door Knock** | 14.98 | 14.88 | -0.09 | QAT_Version | 18 |
| **Keyboard Loud** | 12.36 | 12.17 | -0.19 | QAT_Version | 18 |
| **Keyboard Soft** | 12.48 | 11.91 | -0.58 | QAT_Version | 18 |

### 3.2 噪声类型特性分析

**FullVersion 擅长处理的噪声**:
1. **风扇噪声** (Fan Noise): 稳态、连续性噪声
   - Level 1: +0.39 dB 优势
   - Level 2: +0.10 dB 优势
   - 平均 SI-SNR: 16.36 dB（优于 QAT 的 16.12 dB）

2. **拍手声** (Clapping): 瞬态、冲击性噪声
   - +0.54 dB 优势（最大优势）
   - 平均 SI-SNR: 14.73 dB

**QAT_Version 擅长处理的噪声**:
1. **键盘噪声** (Keyboard): 复杂、不规则噪声
   - Keyboard Soft: -0.58 dB（QAT 最大优势）
   - Keyboard Loud: -0.19 dB
   - 在低SNR 环境下优势更明显

2. **敲门声** (Door Knock): 低频冲击噪声
   - -0.09 dB 微弱优势

**结论**: FullVersion 在稳态和瞬态噪声上表现更好，QAT_Version 在复杂不规则噪声上更有优势。

---

## 4. 按信噪比级别的性能分析

### 4.1 SI-SNR 随SNR 变化趋势

| 输入 SNR | QAT_Version | FullVersion | 差异 (dB) | 优势方 | 样本数 |
|----------|-------------|-------------|-----------|---------|--------|
| **-5dB** | 9.81 | 9.40 | -0.42 | QAT_Version | 18 |
| **0dB** | 12.24 | 11.99 | -0.26 | QAT_Version | 18 |
| **+5dB** | 14.23 | 14.19 | -0.04 | QAT_Version | 18 |
| **+10dB** | 15.72 | 15.90 | +0.18 | FullVersion | 18 |
| **+15dB** | 16.79 | 17.10 | +0.31 | FullVersion | 18 |
| **+20dB** | 17.43 | 17.83 | +0.40 | FullVersion | 18 |

### 4.2 关键洞察

**性能交叉点**: 在 +5dB 到 +10dB 之间存在性能交叉
- **低 SNR 场景** (-5dB ~ +5dB): QAT_Version 表现更好
  - -5dB 时优势最大（-0.42 dB）
  - 在极端噪声环境下更鲁棒
- **高 SNR 场景** (+10dB ~ +20dB): FullVersion 表现更好
  - +20dB 时优势最大（+0.40 dB）
  - 在相对干净环境下能提取更多细节

**应用建议**:
- 工业/嘈杂环境（低 SNR）→ 推荐 QAT_Version
- 办公/会议环境（高 SNR）→ 推荐 FullVersion

---

## 5. 极端案例分析

### 5.1 FullVersion 显著优于 QAT 的案例

| 文件名 | 噪声类型 | SNR | QAT SI-SNR | Full SI-SNR | 差异 |
|--------|----------|-----|------------|-------------|------|
| 00018_1_clapping_snr+20dB | Clapping | +20dB | 17.33 | 18.25 | **+0.92** |
| 00017_1_clapping_snr+15dB | Clapping | +15dB | 16.44 | 17.32 | **+0.88** |
| 00012_1_fan_noise_level2_snr+20dB | Fan L2 | +20dB | 18.22 | 19.06 | **+0.84** |
| 00024_1_door_knock_snr+20dB | Door Knock | +20dB | 18.13 | 18.90 | **+0.77** |
| 00006_1_fan_noise_level1_snr+20dB | Fan L1 | +20dB | 18.23 | 18.98 | **+0.75** |

**共同特征**:
- 全部为高SNR 场景（+15dB 或 +20dB）
- 主要是稳态噪声（风扇）和瞬态噪声（拍手、敲门）
- FullVersion 在高质量输入下能提取更多信息

### 5.2 QAT_Version 显著优于 FullVersion 的案例

| 文件名 | 噪声类型 | SNR | QAT SI-SNR | Full SI-SNR | 差异 |
|--------|----------|-----|------------|-------------|------|
| 00067_2_keyboard_soft_snr-5dB | Keyboard Soft | -5dB | 6.31 | 4.41 | **-1.90** |
| 00068_2_keyboard_soft_snr+0dB | Keyboard Soft | 0dB | 9.04 | 7.57 | **-1.47** |
| 00025_1_keyboard_loud_snr-5dB | Keyboard Loud | -5dB | 5.49 | 4.32 | **-1.17** |
| 00092_3_door_knock_snr+0dB | Door Knock | 0dB | 13.70 | 12.53 | **-1.17** |
| 00061_2_keyboard_loud_snr-5dB | Keyboard Loud | -5dB | 6.08 | 4.98 | **-1.10** |

**共同特征**:
- 主要为低 SNR 场景（-5dB 或 0dB）
- 键盘噪声占主导（5个中有 4 个）
- QAT_Version 在极端噪声下更稳定，差异可达 1.90 dB

---

## 6. 数据分布统计分析

### 6.1 CSV 数据完整性验证

**QAT_Version CSV**:
- 记录数: 108 条
- 字段完整性: 100%（所有 6 个指标均有数据）
- 数据范围验证: ✓ 通过

**FullVersion CSV**:
- 记录数: 108 条
- 字段完整性: 100%
- 数据范围验证: ✓ 通过

### 6.2 指标分布统计

**SI-SNR 分布**:
- QAT_Version: 最小值 5.49 dB, 最大值 18.23 dB, 标准差 ~3.8 dB
- FullVersion: 最小值 4.25 dB, 最大值 19.06 dB, 标准差 ~4.2 dB
- FullVersion 动态范围更大（14.81 dB vs 12.74 dB）

**PESQ 分布**:
- QAT_Version: 范围 1.197 ~ 2.782, 均值 2.151
- FullVersion: 范围 1.159 ~ 3.363, 均值 2.262
- FullVersion 在高质量样本上 PESQ 可达 3.363（显著更高）

---

## 7. 可视化图表分析

根据提供的 4 张图表：

### Chart 1 - 按噪声类型对比 (chart1_comparison_by_noise_type.png)
显示了 6 种噪声类型下两个版本的性能对比，清晰展示了 FullVersion 在风扇噪声和拍手声上的优势，以及 QAT 在键盘噪声上的优势。

### Chart 2 - SI-SNR 按噪声类型 (chart2_sisnr_by_noise_type.png)
详细展示了每种噪声类型的 SI-SNR 分布，可以看到风扇噪声整体性能最好（16+ dB），键盘噪声最具挑战性（12dB 左右）。

### Chart 3 - 柱状图对比 (chart3_bar_comparison.png)
直观对比了 6 大指标，显示了两个版本在不同维度上的权衡关系。

### Chart 4 - 差异热力图 (chart4_difference_heatmap.png)
展示了 108 个样本的逐一对比，可以清晰看到性能差异的分布模式和极端案例的位置。

---

## 8. 结论与建议

### 8.1 综合评价

**性能对比**: 两个版本整体性能接近，各有千秋
- **FullVersion**: 在感知质量（PESQ +5.16%, P808_MOS +7.43%）和高SNR 场景下更优
- **QAT_Version**: 在客观指标（OVRL, SIG, BAK）和低 SNR 场景下更优

### 8.2 应用场景推荐

| 应用场景 | 推荐版本 | 理由 |
|----------|----------|------|
| **移动设备/边缘计算** | QAT_Version | 量化模型更小、更快，低 SNR 性能好 |
| **云端服务/高质量要求** | FullVersion | 感知质量更好，用户体验更佳 |
| **工业/嘈杂环境** | QAT_Version | 极端噪声下更鲁棒 |
| **会议/办公环境** | FullVersion | 高SNR 下细节保留更好 |
| **键盘噪声抑制** | QAT_Version | 在键盘噪声上有明显优势 |
| **风扇/空调噪声** | FullVersion | 稳态噪声处理更优 |

### 8.3 优化方向建议

1. **混合策略**: 根据实时 SNR 估计动态选择模型版本
2. **QAT 改进**: 针对高 SNR 场景优化量化策略，减少精度损失
3. **FullVersion 优化**: 增强低 SNR 场景的鲁棒性
4. **特定噪声优化**: 针对键盘噪声（QAT 优势）和风扇噪声（Full 优势）分别优化

### 8.4 数据质量评估

- ✓ 数据集设计合理，覆盖全面
- ✓ 评估指标多维度，客观+主观结合
- ✓ 样本量充足（108 个），统计可靠
- ✓ 结果可重现，CSV 数据完整

---

## 附录：技术指标说明

- **SI-SNR**: 衡量信号分离质量，不受幅度缩放影响
- **PESQ**: ITU-T P.862 标准，模拟人耳感知
- **OVRL/SIG/BAK**: DNSMOS 指标，分别评估整体、信号、背景
- **P808_MOS**: ITU-T P.808 标准，主观质量评分

---

**报告生成时间**: 2026-01-09
**数据来源**: evaluation_fullversion.csv, evaluation_qat_version.csv
**分析工具**: Claude Code (Opus 4.5)
**原始对比报告**: comparison_report.md
