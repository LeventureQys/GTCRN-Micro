# GTCRN-Micro: *微控制器语音增强* 
<div align="center">

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/github/license/bglid/SERTime)](https://github.com/bglid/SERTime/blob/main/LICENSE)
[![Actions status](https://github.com/bglid/SERTime/workflows/build-desktop/badge.svg)](https://github.com/bglid/SERTime/actions)
[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
- - -
**进行中**
- - -

*项目仍在进行中。目前正在为 ESP32-S3 部署进行架构重构，使用调整后模型架构的 `tflite` 表示形式。正在训练一个考虑到这一点的新架构* 

*第二个目标是使用 STM32Cube_AI 将更接近原始* [GTCRN](https://github.com/Xiaobin-Rong/gtcrn) *的架构部署到 STM32 硬件上*

*有关更新，请查看问题或项目路线图 [here](./docs/plan.md) 和 [here](./docs/TODO.md)。如果您发现任何问题，请提交。非常感谢！*

- - - 
</div>

## 项目背景

该项目的目标是将一个现代、强大、轻量级的语音增强模型进行调整，将其量化为 int8 表示形式，并尝试使用 `tflite` 将其部署到 ESP32-S3 上，同时尽可能保持性能。

这个项目的动机来自于对设计可在微控制器上运行的语音处理（主要是语音增强）模型的普遍兴趣。像 GTCRN 这样令人印象深刻的模型展示了在设计语音增强方面的重大进步，在保持出色性能的同时非常轻量。我一直对量化并部署像 GTCRN 这样的模型到微控制器的过程很好奇。最终，这是一个让我能够在这一感兴趣的领域培养技能的热情项目，并帮助为其他想要做同样事情的人提供见解。 

请查看[致谢！](#致谢)

<!-- ## 如何使用

### 设置
<details>

#### 克隆项目：
```bash
git clone https://github.com/benjaminglidden/GTCRN-Micro.git
cd GTCRN-Micro
```
此项目使用 uv 作为依赖管理器。 
- 要设置项目依赖项，首先确保在您的设备上安装了 [uv](https://docs.astral.sh/uv/)：

#### 安装 uv：
**Linux & Mac OS**

从**终端**：
 - 使用 curl 下载脚本并使用 sh 执行：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
 - 如果由于某种原因您没有 `curl`，请使用 `wget`

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

**Windows**

从 **PowerShell：**

 - 使用 `irm` 下载脚本并使用 `iex` 执行：

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

####  验证 UV 安装

 - 要验证您是否已正确安装，请从终端（或 PowerShell）运行：
```
uv --version
```
 - 您应该会收到 UV 的版本号

#### 安装依赖项
```bash
uv sync
```
</details>

### 使用离线非量化模型
*进行中*

- 训练好的模型检查点可以在 [ckpts](./gtcrn_micro/ckpts/) 中找到
- Onnx 文件可以在 [onxx](./gtcrn_micro/models/onnx/) 中找到
...

### 量化模型
*进行中*

- 量化的 tflite 文件可以在 [tflite](./gtcrn_micro/models/tflite/) 中找到
     - 如果需要重新创建量化，例如必要的[替换 .json](./gtcrn_micro/models/tflite/replace_gtcrn_micro.json) 在那里
... -->
- - - 

## 路线图 / 待办事项 

##### *更新 2025年12月26日：*

已训练新模型，并进行了一些架构更改。主要更改是：流式兼容的膨胀和填充，添加了可量化的 TRA 和 SFE 的 Lite 版本，并修复了 GTCN 部分的膨胀。该模型的结果可以在 [gtcrn_micro](./gtcrn_micro/README.md) 目录中找到。

下一步是使用创建的流式架构：
 - 1. 测试 PyTorch 流式变体的性能
 - 2. 将流式变体导出到 ONNX 并测试性能
 - 3. 将 ONNX 流式变体导出到 TFLite 并测试性能。

一旦完成这些，这将转向 MCU 部署测试，首先针对 **ESP32-S3**。 


##### *更新 2025年12月19日：*

  需要调整模型架构以获得更好的流式性能。该模型将通过完整的 PyTorch $\rightarrow$ ONNX $\rightarrow$ TFLite 转换。目前正在训练支持此硬件的流式模型。 

未来的目标是尝试跳过 `tflite` 转换并使用 STM32CubeAI 进行部署。

*有关更新，请查看问题或项目路线图 [here](./docs/plan.md) 和 [here](./docs/TODO.md)。如果您发现任何问题，请提交。非常感谢！*

- - -
## 致谢

###### 1. 本模型基于 [GTCRN: A Speech Enhancement Model Requiring Ultralow Computational Resources](https://ieeexplore.ieee.org/document/10448310)，利用了 [GTCRN](https://github.com/Xiaobin-Rong/gtcrn) 的实现代码。训练和更改模型的大量设置基于同一作者的项目 [SEtrain](https://github.com/Xiaobin-Rong/SEtrain/tree/plus)。他们有非常令人印象深刻的 SE 研究！请查看他们的研究并给他们的作品一个星标！
###### 2. 该项目还需要从 **PyTorch $\rightarrow$ ONNX $\rightarrow$ .tflite** 进行转换以在 ESP32 上运行推理。如果没有 [PINTO0309](https://github.com/PINTO0309) 的直接帮助和他们的出色项目 [onnx2tf](https://github.com/PINTO0309/onnx2tf)，这一切都不可能实现。如果您正在阅读本文并想做一个类似的项目，我强烈建议您查看他们的工作。请考虑给他们的作品一个星标！
###### 3. SFE 和 TRA 的更改遵循了 [gtcrn-light](https://github.com/zerong7777-boop/gtcrn-light) 中的实现。他们的实现能够在没有 GRU 的情况下保持这些块的功能，使其可以在没有外部 CPU 的情况下在 TFLM 中表示。他们的实现很棒，我建议您查看一下。 
- - - 

*没有他们的努力，这个项目是不可能实现的。请考虑引用他们并在给本项目星标之前先给他们一个星标！* 

- - - 

