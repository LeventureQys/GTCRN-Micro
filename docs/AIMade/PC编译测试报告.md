# GTCRN-Micro 项目 WSL 环境验证指南

## 1. 项目概述

GTCRN-Micro 是一个用于微控制器语音增强的项目。该项目将现代轻量级语音增强模型（基于 GTCRN）进行调整、量化并部署到 ESP32-S3 等微控制器上。

**重要提示**：本文档专门针对 **WSL (Windows Subsystem for Linux)** 环境下的验证和测试。项目提供了预训练的模型检查点，可以直接用于评估和测试，无需重新训练。

### 预训练模型

项目在 `gtcrn_micro/ckpts/best_model_dns3.tar` 提供了预训练的 GTCRN-Micro 模型：
- **模型参数**：19.01k
- **MMACs**：45.92
- **模型文件**：`gtcrn_micro/ckpts/best_model_dns3.tar`

该模型已在 DNS3 数据集上训练，可以直接加载使用进行推理和内存评估。

### 本文档重点

本文档专注于**在 WSL 环境下验证和评估模型**，包括：
- WSL 环境配置和依赖安装
- 如何加载预训练模型
- 如何分析模型参数内存占用
- 如何分析推理时的内存占用（CPU 环境）
- 如何评估不同输入大小对内存的影响

---

## 2. WSL 环境要求

### 2.1 系统要求
- **Windows 版本**：Windows 10 (版本 2004 或更高) 或 Windows 11
- **WSL 版本**：WSL 2（推荐）或 WSL 1
- **Linux 发行版**：Ubuntu 20.04/22.04（推荐）或其他支持的发行版
- **Python 版本**：Python >= 3.9（通常在 Linux 发行版中已包含）

### 2.2 WSL 安装和配置

#### 2.2.1 安装 WSL 2（如果尚未安装）

在 Windows PowerShell（以管理员身份运行）中执行：

```powershell
wsl --install
```

或手动安装：

```powershell
# 启用 WSL 功能
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart

# 启用虚拟机平台
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# 重启后，设置 WSL 2 为默认版本
wsl --set-default-version 2

# 安装 Ubuntu（或其他发行版）
wsl --install -d Ubuntu-22.04
```

#### 2.2.2 验证 WSL 安装

在 WSL 终端中运行：

```bash
# 检查 WSL 版本
wsl --version

# 检查 Linux 发行版信息
lsb_release -a

# 检查 Python 版本
python3 --version
```

### 2.3 硬件要求（内存评估）

- **CPU**：支持虚拟化的现代 CPU（WSL 2 需要）
- **内存**：至少 4GB RAM（推荐 8GB 或更多，WSL 2 会分配一部分内存）
- **GPU**（可选）：NVIDIA GPU（WSL 2 下可以通过 CUDA on WSL 使用，但配置较复杂）
  - **注意**：本文档主要针对 CPU 环境下的验证
- **存储空间**：至少 2GB 可用空间（用于 WSL、项目文件和预训练模型）

### 2.4 必需软件（在 WSL 中安装）

在 WSL Linux 环境中需要安装：

- **uv**：现代 Python 包管理器（项目使用 uv 作为依赖管理器）
- **Git**：用于克隆项目仓库
- **build-essential**：编译工具链（某些 Python 包可能需要）
- **curl** 或 **wget**：用于下载安装脚本

---

## 3. WSL 环境安装步骤

### 3.1 更新 WSL 系统包

进入 WSL 终端，首先更新系统包：

```bash
sudo apt update
sudo apt upgrade -y
```

### 3.2 安装基础工具

```bash
# 安装 Git 和构建工具
sudo apt install -y git curl wget build-essential

# 验证安装
git --version
curl --version
```

### 3.3 克隆项目

在 WSL 文件系统中克隆项目（建议在 WSL 文件系统中操作，避免跨文件系统性能问题）：

```bash
# 推荐：在 WSL 主目录下克隆
cd ~
git clone https://github.com/benjaminglidden/GTCRN-Micro.git
cd GTCRN-Micro
```

**注意**：
- 建议在 WSL 文件系统（`/home/username/`）中操作，而不是 Windows 文件系统（`/mnt/c/`）
- WSL 文件系统路径通常为：`~/` 或 `/home/username/`
- Windows 文件系统路径为：`/mnt/c/`、`/mnt/d/` 等（性能较差）

---

## 3. 安装步骤

### 3.1 克隆项目

```bash
git clone https://github.com/benjaminglidden/GTCRN-Micro.git
cd GTCRN-Micro
```

### 3.4 安装 uv 包管理器

项目使用 `uv` 作为依赖管理器。在 WSL Linux 环境中安装 uv：

```bash
# 使用 curl 安装（推荐）
curl -LsSf https://astral.sh/uv/install.sh | sh
```

如果没有 `curl`，可以使用 `wget`：

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

#### 配置 uv 环境变量

安装完成后，将 uv 添加到 PATH（通常会自动添加到 `~/.bashrc` 或 `~/.zshrc`）：

```bash
# 重新加载 shell 配置
source ~/.bashrc

# 或者手动添加到 PATH（如果未自动添加）
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

#### 验证 uv 安装

```bash
uv --version
```

应该返回 uv 的版本号，例如：`uv 0.x.x`

### 3.5 安装项目依赖

在项目根目录执行：

```bash
# 确保在项目根目录
cd ~/GTCRN-Micro  # 或您的项目路径

# 同步依赖（这会创建虚拟环境并安装所有依赖）
uv sync
```

该命令会：
- 创建 Python 虚拟环境（通常在 `.venv` 目录）
- 安装 `pyproject.toml` 中列出的所有依赖项
- 安装开发依赖（如 pytest 等）

**安装时间**：首次安装可能需要 5-15 分钟，取决于网络速度和依赖数量。

### 3.6 验证安装

可以通过导入主要模块验证安装是否成功：

```bash
# 验证安装
uv run python -c "import gtcrn_micro; print('安装成功！')"
```

如果出现错误，可能需要安装额外的系统依赖：

```bash
# 安装音频处理相关的系统库（如果缺少）
sudo apt install -y libsndfile1 libsndfile1-dev

# 安装其他可能需要的库
sudo apt install -y python3-dev python3-pip
```

---

## 4. 使用预训练模型

### 4.1 检查预训练模型

项目提供了预训练模型检查点，位于 `gtcrn_micro/ckpts/best_model_dns3.tar`。

在 WSL 终端中验证模型文件是否存在：

```bash
# 检查模型文件
ls -lh gtcrn_micro/ckpts/best_model_dns3.tar

# 如果文件不存在，检查目录
ls -la gtcrn_micro/ckpts/

# 查看文件大小和详细信息
file gtcrn_micro/ckpts/best_model_dns3.tar
```

### 4.2 快速加载和测试预训练模型

可以使用以下 Python 代码快速验证模型是否正常加载：

```python
import torch
from gtcrn_micro.models.gtcrn_micro import GTCRNMicro

# 加载预训练模型
checkpoint_path = "gtcrn_micro/ckpts/best_model_dns3.tar"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GTCRNMicro().to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()

print("模型加载成功！")
print(f"设备: {device}")
print(f"参数数量: {sum(p.numel() for p in model.parameters()):,}")
```

---

## 5. 内存占用评估

### 5.1 使用内存评估脚本（推荐）

项目提供了专门的内存评估脚本 `gtcrn_micro/utils/memory_profile.py`。

#### 运行内存评估

```bash
uv run python -m gtcrn_micro.utils.memory_profile
```

该脚本会自动：
1. 检查并加载预训练模型（如果存在）
2. 统计模型参数数量和内存占用
3. 分析推理时的内存占用（包括激活值和中间结果）
4. 测试不同输入大小对内存的影响
5. 生成详细的内存使用报告

#### 评估输出示例

脚本会输出类似以下内容：

```
================================================================================
GTCRN-Micro 模型内存占用评估
================================================================================
找到预训练模型: gtcrn_micro/ckpts/best_model_dns3.tar

使用设备: cuda
GPU: NVIDIA GeForce RTX 3090
CUDA 版本: 11.8

--------------------------------------------------------------------------------
1. 创建模型
--------------------------------------------------------------------------------
加载预训练权重: gtcrn_micro/ckpts/best_model_dns3.tar
预训练权重加载完成
模型创建完成

--------------------------------------------------------------------------------
2. 模型参数统计
--------------------------------------------------------------------------------
总参数数量: 19,010
可训练参数: 19,010
非训练参数: 0

--------------------------------------------------------------------------------
3. 模型参数内存占用
--------------------------------------------------------------------------------
模型参数内存: 76.04 KB
参数量 × 4字节 (FP32): 76.04 KB

--------------------------------------------------------------------------------
4. 推理时内存占用分析
--------------------------------------------------------------------------------
输入形状: (1, 257, 63, 2) - 批次大小=1, 频率=257, 时间=63, 通道=2
开始分析...

GPU 内存使用:
  初始内存: 0.00 MB
  预热后内存: 1.23 MB
  峰值内存: 2.45 MB
  预留内存: 3.00 MB
  推理内存占用 (平均): 2.40 MB
  推理内存占用 (最大): 2.45 MB
  推理内存占用 (最小): 2.35 MB
```

### 5.2 手动评估模型参数内存

如果你想手动计算模型参数的内存占用：

```python
import torch
from gtcrn_micro.models.gtcrn_micro import GTCRNMicro

model = GTCRNMicro().eval()
checkpoint = torch.load("gtcrn_micro/ckpts/best_model_dns3.tar", map_location="cpu")
model.load_state_dict(checkpoint["model"])

# 计算参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数数量: {total_params:,}")

# 计算内存占用（FP32，每个参数4字节）
memory_fp32 = total_params * 4 / (1024 * 1024)  # MB
print(f"FP32 内存占用: {memory_fp32:.2f} MB")

# 计算内存占用（FP16，每个参数2字节）
memory_fp16 = total_params * 2 / (1024 * 1024)  # MB
print(f"FP16 内存占用: {memory_fp16:.2f} MB")

# 计算内存占用（INT8，每个参数1字节）
memory_int8 = total_params * 1 / (1024 * 1024)  # MB
print(f"INT8 内存占用: {memory_int8:.2f} MB")
```

### 5.3 评估推理时内存占用（GPU）

如果使用 GPU，可以使用 PyTorch 的内存分析工具：

```python
import torch
from gtcrn_micro.models.gtcrn_micro import GTCRNMicro

device = torch.device("cuda")
model = GTCRNMicro().to(device)
checkpoint = torch.load("gtcrn_micro/ckpts/best_model_dns3.tar", map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()

# 清除缓存
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# 创建输入
dummy_input = torch.randn(1, 257, 63, 2, device=device)

# 推理
with torch.no_grad():
    output = model(dummy_input)

# 获取内存使用情况
peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)  # MB
reserved_memory = torch.cuda.memory_reserved(device) / (1024 * 1024)  # MB

print(f"峰值内存: {peak_memory:.2f} MB")
print(f"预留内存: {reserved_memory:.2f} MB")
```

### 5.4 评估推理时内存占用（CPU）

如果使用 CPU，可以使用 `psutil` 库：

```python
import torch
import psutil
import os
from gtcrn_micro.models.gtcrn_micro import GTCRNMicro

model = GTCRNMicro().eval()
checkpoint = torch.load("gtcrn_micro/ckpts/best_model_dns3.tar", map_location="cpu")
model.load_state_dict(checkpoint["model"])

process = psutil.Process(os.getpid())

# 获取初始内存
initial_memory = process.memory_info().rss / (1024 * 1024)  # MB

# 创建输入
dummy_input = torch.randn(1, 257, 63, 2)

# 推理
with torch.no_grad():
    output = model(dummy_input)

# 获取推理后内存
after_inference_memory = process.memory_info().rss / (1024 * 1024)  # MB

print(f"初始内存: {initial_memory:.2f} MB")
print(f"推理后内存: {after_inference_memory:.2f} MB")
print(f"推理内存增量: {after_inference_memory - initial_memory:.2f} MB")
```

### 5.5 评估不同输入大小的内存占用

可以测试不同长度的音频对内存的影响：

```python
import torch
from gtcrn_micro.models.gtcrn_micro import GTCRNMicro

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GTCRNMicro().to(device)
checkpoint = torch.load("gtcrn_micro/ckpts/best_model_dns3.tar", map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()

# 测试不同的输入大小（时间帧数）
test_sizes = [63, 126, 252, 504]  # 对应约1秒、2秒、4秒、8秒音频

for time_frames in test_sizes:
    dummy_input = torch.randn(1, 257, time_frames, 2, device=device)
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(dummy_input)
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        print(f"时间帧 {time_frames} (~{time_frames*256/16000:.1f}秒): {peak_mem:.2f} MB")
    else:
        # CPU 评估（使用 psutil）
        process = psutil.Process(os.getpid())
        before = process.memory_info().rss / (1024 * 1024)
        with torch.no_grad():
            _ = model(dummy_input)
        after = process.memory_info().rss / (1024 * 1024)
        print(f"时间帧 {time_frames} (~{time_frames*256/16000:.1f}秒): {after - before:.2f} MB")
```

---

## 6. 模型复杂度分析

除了内存占用，还可以分析模型的计算复杂度：

### 6.1 使用 ptflops 分析模型复杂度

项目已经在 `gtcrn_micro/models/gtcrn_micro.py` 中集成了复杂度分析：

```bash
uv run python -m gtcrn_micro.models.gtcrn_micro
```

这会输出：
- 模型的 FLOPs（浮点运算次数）
- 每层的参数统计
- 总参数数量

### 6.2 手动分析模型复杂度

```python
import torch
from ptflops import get_model_complexity_info
from gtcrn_micro.models.gtcrn_micro import GTCRNMicro

model = GTCRNMicro().eval()
checkpoint = torch.load("gtcrn_micro/ckpts/best_model_dns3.tar", map_location="cpu")
model.load_state_dict(checkpoint["model"])

# 分析复杂度（输入形状：频率=257, 时间=63, 通道=2）
flops, params = get_model_complexity_info(
    model,
    (257, 63, 2),  # (F, T, C)
    as_strings=True,
    print_per_layer_stat=True,
    verbose=True
)

print(f"FLOPs: {flops}")
print(f"参数: {params}")
```

---

## 7. 内存评估结果示例

### 7.1 典型内存占用（基于 19.01k 参数）

根据模型参数数量（19.01k），不同精度下的内存占用：

| 精度 | 参数量 | 模型参数内存 | 典型推理内存 (GPU) | 典型推理内存 (CPU) |
|------|--------|--------------|-------------------|-------------------|
| FP32 | 19,010 | ~76 KB | ~2-3 MB | ~1-2 MB |
| FP16 | 19,010 | ~38 KB | ~1-2 MB | N/A |
| INT8 | 19,010 | ~19 KB | ~0.5-1 MB | ~0.5-1 MB |

**注意**：
- 模型参数内存只包括模型权重本身
- 推理内存包括：模型参数 + 激活值 + 中间计算结果 + 缓存
- 实际内存占用会根据输入大小（音频长度）而变化

### 7.2 不同输入大小的内存占用

对于不同的音频长度（时间帧数），内存占用会有所不同：

| 时间帧 | 音频长度（秒） | 推理内存（近似） |
|--------|---------------|-----------------|
| 63     | ~1.0          | ~2 MB           |
| 126    | ~2.0          | ~3 MB           |
| 252    | ~4.0          | ~5 MB           |
| 504    | ~8.0          | ~8 MB           |

### 7.3 内存优化建议

如果需要在资源受限的环境中部署：

1. **量化**：使用 INT8 量化可以大幅减少内存占用（约 4 倍）
2. **流式处理**：使用流式推理，每次只处理一小段音频，减少峰值内存
3. **批次大小**：推理时使用批次大小=1，减少内存占用
4. **CPU vs GPU**：在某些情况下，CPU 推理可能比 GPU 推理占用更少内存（但速度更慢）

---

## 8. 常见问题和故障排除

### 8.1 预训练模型相关问题

#### 问题：找不到预训练模型文件

**解决方案**：
- 确认文件路径：`gtcrn_micro/ckpts/best_model_dns3.tar`
- 检查文件是否存在：`ls gtcrn_micro/ckpts/` 或 Windows: `dir gtcrn_micro\ckpts\`
- 如果文件不存在，可能需要从项目仓库重新下载或克隆

#### 问题：加载模型时出现 KeyError

**解决方案**：
- 模型检查点可能使用不同的键名，尝试：
  ```python
  state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
  model.load_state_dict(state_dict, strict=False)
  ```
- 使用 `strict=False` 可以忽略不匹配的键

### 8.2 内存评估问题

#### 问题：GPU 内存不足（WSL 环境下）

**解决方案**：
- **WSL 环境主要使用 CPU**：WSL 环境下 GPU 支持配置较复杂，建议使用 CPU 进行评估
- 使用 CPU 进行评估：将 `device` 设置为 `"cpu"`（WSL 环境下的默认方式）
- 减小输入大小：使用更短的音频片段
- 如果需要 GPU，需要安装 CUDA on WSL（配置较复杂，本文档主要针对 CPU 验证）

**注意**：本文档主要针对 WSL 环境下的 CPU 验证，GPU 验证不在本文档范围内。

#### 问题：内存评估脚本运行失败

**解决方案**：
- 确保已安装所有依赖：`uv sync`
- 检查 `psutil` 是否已安装：`uv run python -c "import psutil"`
- 确保在项目根目录运行脚本

#### 问题：CPU 内存评估不准确

**解决方案**：
- CPU 内存评估受系统其他进程影响，多次运行取平均值
- 关闭其他占用内存的程序
- 使用 `memory_profile.py` 脚本，它会进行多次测试并取平均

### 8.3 WSL 环境特定问题

#### 问题：WSL 文件系统性能慢

**解决方案**：
- **在 WSL 文件系统中操作**：避免在 Windows 文件系统（`/mnt/c/`）中运行项目
- 将项目克隆到 WSL 文件系统：`~/GTCRN-Micro` 而不是 `/mnt/c/...`
- 检查当前路径：`pwd`（应该在 `/home/username/` 下）
- 如果在 `/mnt/` 下，建议迁移项目到 WSL 文件系统

#### 问题：WSL 中 Python 版本问题

**解决方案**：
```bash
# 检查 Python 版本
python3 --version

# 如果需要，安装 Python 3.9+
sudo apt update
sudo apt install -y python3.9 python3.9-venv python3-pip

# 如果系统有多个 Python 版本，可以设置默认版本
sudo update-alternatives --config python3
```

#### 问题：WSL 权限问题

**解决方案**：
- 确保在用户目录下操作：`cd ~`
- 避免使用 `sudo` 运行 Python 脚本（除非必要）
- 如果遇到权限问题，检查文件所有者：`ls -la`

#### 问题：WSL 中某些 Python 包编译失败

**解决方案**：
```bash
# 安装编译工具和开发库
sudo apt install -y build-essential python3-dev

# 安装音频处理库
sudo apt install -y libsndfile1 libsndfile1-dev

# 安装其他可能需要的系统库
sudo apt install -y libffi-dev libssl-dev
```

### 8.4 依赖安装问题

#### 问题：`uv sync` 失败

**解决方案**：
```bash
# 检查 Python 版本
python3 --version  # 应该 >= 3.9

# 尝试更新 uv
uv self update

# 检查网络连接
ping -c 3 8.8.8.8

# 如果网络较慢，可以设置代理（如有）
# export HTTP_PROXY=http://proxy:port
# export HTTPS_PROXY=http://proxy:port

# 清理并重新安装
rm -rf .venv
uv sync --verbose
```

#### 问题：CUDA 相关错误（WSL 环境）

**解决方案**：
- **WSL 环境默认使用 CPU**：这是正常的，本文档主要针对 CPU 验证
- 如果出现 CUDA 错误，可以忽略（在 CPU 模式下不影响使用）
- CPU 版本的 PyTorch 完全满足本文档的验证需求
- 如果需要 GPU，需要安装 CUDA on WSL 和 NVIDIA WSL 驱动（配置较复杂，不在本文档范围内）

#### 问题：音频库相关错误

**解决方案**：
```bash
# 安装音频处理库
sudo apt install -y libsndfile1 libsndfile1-dev

# 如果仍有问题，尝试重新安装 Python 音频库
uv pip install --force-reinstall soundfile librosa
```

---

## 9. 性能参考

根据项目 README，预训练模型的性能指标如下：

### 模型基本信息
- **模型参数**：19.01k
- **MMACs**：45.92
- **模型文件**：`gtcrn_micro/ckpts/best_model_dns3.tar`

### DNS3 合成测试集性能
- **SDR**: 10.41
- **SI-SNR**: 9.85
- **PESQ**: 1.98
- **STOI**: 0.85

### DNS3 盲测集性能（DNSMOS）
- **DNSMOS-P.808**：3.25
- **BAK**：3.60
- **SIG**：2.99
- **OVRL**：2.58

---

## 10. WSL 环境验证总结

本指南详细介绍了如何在 **WSL (Windows Subsystem for Linux)** 环境下验证和评估 GTCRN-Micro 预训练模型的内存占用。

### 关键要点

1. **WSL 环境**：本文档专门针对 WSL 2 环境，提供了完整的设置步骤
2. **预训练模型**：项目提供了预训练模型检查点，可以直接使用，无需训练
3. **CPU 验证**：本文档主要针对 CPU 环境下的验证（WSL 环境下推荐方式）
4. **内存评估工具**：提供了专门的内存评估脚本 `gtcrn_micro/utils/memory_profile.py`
5. **全面评估**：可以评估模型参数内存、推理内存，以及不同输入大小的影响

### WSL 环境优势

- ✅ **Linux 兼容性**：可以使用所有 Linux 工具和命令
- ✅ **易于安装**：Python 生态系统的包安装更顺畅
- ✅ **性能良好**：在 WSL 文件系统中操作性能良好
- ✅ **隔离性好**：不影响 Windows 系统环境

### 推荐工作流程（WSL）

1. **WSL 环境准备**：安装和配置 WSL 2
2. **基础工具安装**：Git、curl、build-essential 等
3. **项目环境设置**：安装 uv 和项目依赖
4. **验证模型**：确认预训练模型文件存在
5. **运行评估**：使用 `memory_profile.py` 脚本进行完整的内存评估
6. **分析结果**：根据输出结果分析模型的内存占用情况

### 快速开始（WSL）

```bash
# 1. 进入 WSL 终端，切换到项目目录
cd ~/GTCRN-Micro

# 2. 安装依赖（如果尚未安装）
uv sync

# 3. 验证安装
uv run python -c "import gtcrn_micro; print('安装成功！')"

# 4. 运行内存评估（CPU 模式）
uv run python -m gtcrn_micro.utils.memory_profile
```

### 重要提示

- **文件系统**：建议在 WSL 文件系统（`~/`）中操作，而不是 Windows 文件系统（`/mnt/c/`）
- **CPU 验证**：本文档主要针对 CPU 环境，这是 WSL 环境下最简单可靠的方式
- **GPU 支持**：WSL 环境下 GPU 支持需要额外配置（不在本文档范围内）

---

**报告更新日期**：2026年1月8日  
**适用环境**：WSL 2 (Windows Subsystem for Linux)  
**主要验证方式**：CPU 推理和内存评估

---

## 附录

### A. 有用的命令速查表

```bash
# 安装依赖
uv sync

# 验证安装
uv run python -c "import gtcrn_micro; print('安装成功！')"

# 检查预训练模型（WSL/Linux 环境）
ls -lh gtcrn_micro/ckpts/best_model_dns3.tar

# 运行内存评估（推荐）
uv run python -m gtcrn_micro.utils.memory_profile

# 分析模型复杂度
uv run python -m gtcrn_micro.models.gtcrn_micro

# 运行单元测试
uv run pytest tests/models/test_gtcrn_micro.py -v
```

### B. 相关资源

- **项目仓库**：https://github.com/benjaminglidden/GTCRN-Micro
- **原始 GTCRN**：https://github.com/Xiaobin-Rong/gtcrn
- **uv 文档**：https://docs.astral.sh/uv/
- **onnx2tf**：https://github.com/PINTO0309/onnx2tf

### C. 配置文件说明

- `pyproject.toml`：项目配置、依赖和元数据
- `gtcrn_micro/conf/cfg_train_DNS3.yaml`：DNS3 数据集训练配置
- `gtcrn_micro/conf/cfg_infer.yaml`：推理配置
- `gtcrn_micro/streaming/tflite/replace_gtcrn_micro.json`：ONNX 到 TFLite 转换的替换配置

---

**报告生成日期**：2026年1月8日  
**项目版本**：v0.1.0  
**Python 要求**：>= 3.9

