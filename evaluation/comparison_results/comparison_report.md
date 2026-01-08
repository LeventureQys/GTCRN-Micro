# 降噪测评对比分析报告

**对比对象**: QAT_Version vs FullVersion

**生成时间**: 2026-01-08 16:22:07

---

## 1. 数据概览

| 项目 | QAT_Version | FullVersion |
|------|---------|---------|
| 样本数量 | 108 | 108 |
| 噪声类型数 | 6 | 6 |
| 说话者数 | 3 | 3 |
| SNR级别数 | 6 | 6 |

---

## 2. 整体指标对比

| 指标 | QAT_Version | FullVersion | 差异 | 胜出 |
|------|---------|---------|------|------|
| SI-SNR | 14.371 | 14.401 | +0.030 ↑ | **FullVersion** |
| PESQ | 2.151 | 2.262 | +0.111 ↑ | **FullVersion** |
| OVRL | 3.473 | 3.161 | -0.312 ↓ | **QAT_Version** |
| SIG | 3.774 | 3.515 | -0.259 ↓ | **QAT_Version** |
| BAK | 4.274 | 3.944 | -0.330 ↓ | **QAT_Version** |
| P808_MOS | 3.473 | 3.731 | +0.258 ↑ | **FullVersion** |

---

## 3. 按噪声类型对比 (SI-SNR)

| 噪声类型 | QAT_Version | FullVersion | 差异 | 胜出 |
|----------|---------|---------|------|------|
| Fan Noise Level 1 | 15.94 | 16.32 | +0.39 | **FullVersion** |
| Fan Noise Level 2 | 16.29 | 16.39 | +0.10 | **FullVersion** |
| Clapping | 14.18 | 14.73 | +0.54 | **FullVersion** |
| Door Knock | 14.98 | 14.88 | -0.09 | **QAT_Version** |
| Keyboard Loud | 12.36 | 12.17 | -0.19 | **QAT_Version** |
| Keyboard Soft | 12.48 | 11.91 | -0.58 | **QAT_Version** |

---

## 4. 按SNR级别对比 (SI-SNR)

| SNR | QAT_Version | FullVersion | 差异 | 胜出 |
|-----|---------|---------|------|------|
| -5dB | 9.81 | 9.40 | -0.42 | **QAT_Version** |
| +0dB | 12.24 | 11.99 | -0.26 | **QAT_Version** |
| +5dB | 14.23 | 14.19 | -0.04 | **QAT_Version** |
| +10dB | 15.72 | 15.90 | +0.18 | **FullVersion** |
| +15dB | 16.79 | 17.10 | +0.31 | **FullVersion** |
| +20dB | 17.43 | 17.83 | +0.40 | **FullVersion** |

---

## 5. 详细文件对比

以下表格显示同名文件的SI-SNR差异（FullVersion - QAT_Version）：

### 5.1 差异最大的文件（FullVersion显著优于QAT_Version）

| 文件名 | QAT_Version SI-SNR | FullVersion SI-SNR | 差异 |
|--------|---------------|---------------|------|
| 00018_1_clapping_snr+20dB_noisy.wav... | 17.33 | 18.25 | +0.92 |
| 00017_1_clapping_snr+15dB_noisy.wav... | 16.44 | 17.32 | +0.88 |
| 00012_1_fan_noise_level2_snr+20dB_noisy.... | 18.22 | 19.06 | +0.84 |
| 00024_1_door_knock_snr+20dB_noisy.wav... | 18.13 | 18.90 | +0.77 |
| 00006_1_fan_noise_level1_snr+20dB_noisy.... | 18.23 | 18.98 | +0.75 |

### 5.2 差异最大的文件（QAT_Version显著优于FullVersion）

| 文件名 | QAT_Version SI-SNR | FullVersion SI-SNR | 差异 |
|--------|---------------|---------------|------|
| 00067_2_keyboard_soft_snr-5dB_noisy.wav... | 6.31 | 4.41 | -1.90 |
| 00068_2_keyboard_soft_snr+0dB_noisy.wav... | 9.04 | 7.57 | -1.47 |
| 00025_1_keyboard_loud_snr-5dB_noisy.wav... | 5.49 | 4.32 | -1.17 |
| 00092_3_door_knock_snr+0dB_noisy.wav... | 13.70 | 12.53 | -1.17 |
| 00061_2_keyboard_loud_snr-5dB_noisy.wav... | 6.08 | 4.98 | -1.10 |

---

## 6. 结论

### 整体表现

- **QAT_Version** 在 3 个指标上领先
- **FullVersion** 在 3 个指标上领先
- SI-SNR 平均差异: +0.03 dB

**结论**: 两者表现相近