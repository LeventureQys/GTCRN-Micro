# GTCRN-Micro: *An attempt to rebuild a Speech Enhancement model for Microcontrollers* 
<div align="center">

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/github/license/bglid/SERTime)](https://github.com/bglid/SERTime/blob/main/LICENSE)
[![Actions status](https://github.com/bglid/SERTime/workflows/build-desktop/badge.svg)](https://github.com/bglid/SERTime/actions)
[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
- - -

*Initially, this the goal of this project was to build a TCN-focused rebuild of GTCRN onto an MCU. Unfortunately, the current approach does not seem viable for targeting MCUs. I encourage anyone reading to give an attempt with the given PyTorch checkpoints and streaming models to attempt to quantize to int8. If you manage to be successful, please let me know! I would love to be wrong here.*

*I unfortunately need to step away from the project for some time. Therefore, I will be unable to continue testing out ways to try and push this project towards MCU Speech Enhancement with this approach for a short while.*

- - -

*For updates, check out either the issues, or the project roadmap [here](./docs/plan.md) and [here](./docs/TODO.md). Please submit any issues if you catch any. It's greatly appreciated! If you manage to get an integer quantized version running, please let me know!*

- - - 
</div>

## Initial Project Background

The goal of this project was to adjust a modern, powerful, lightweight speech ehancement model to quantize it to an int8 representation. Quantization was the goal to attempt deploy it to an ESP32-S3 with `tflite`, while trying to preserve as much performance as possible.

The motivation for this project comes from a general interest in designing speech processing (mainly speech enhancement) models that can run on microcontrollers. Impresive models such as GTCRN showcase significant advancements in designing speech enhancement that maintain great performance whilst being very lightweight. I have been generally curious in working through the process of quantizing and deploying a model like GTCRN to a microcontroller for quite some time. Ultimately, it's a passion project that allows me to build skills in this area of interest, and help provide insight for anyone else looking to do the same. 

Please check out the [acknowledgements!](#acknowledgements)

## How to use

 - For a detailed setup, see [how_to.md](./docs/how_to.md)


#### Using the offline torch model

- Trained model checkpoints can be found in [ckpts](./gtcrn_micro/ckpts/)
- Also if for some reason you want a *non-streaming* version of the `ONNX` model, that can be found here: [gtcrn_micro.onnx](./gtcrn_micro/streaming/onnx/gtcrn_micro.onnx)

#### Using the streaming variants of the model

- **PyTorch Streaming model**:
   - Essentially, you'll want to load up the non-streaming model weights as normal, then run convert_to_stream function passing in the streaming model in eval. See the "how_to" guide for more details: [how_to.md](./docs/how_to.md)

- **Streaming `ONNX` models**: 
  - Found at [gtcrn_micro_stream.onnx](./gtcrn_micro/streaming/onnx/gtcrn_micro_stream.onnx) & [gtcrn_micro_stream_simple.onnx](./gtcrn_micro/streaming/onnx/gtcrn_micro_stream_simple.onnx)
   - These should both be the same, the underlying graph is just simplified in the latter if you decide to try and analyze the model with Netron
- **Streaming `TFLite (LiteRT)` models**: 
  - Found at [gtcrn_micro/streaming/tflite](./gtcrn_micro/streaming/tflite/)
   - **NOTE:** These are mainly included here for completeness, but their performance is notably degraded. These were a part of the attempt to deploy to MCUs. I advise that you use the `ONNX` models.


- - - 

## Archived Roadmap / to-dos 

##### *Update 01/07/2025:*

Unfortunately, I am unsure on the viability of this approach for MCUs. Certain bottlenecks when quantizing have completely halted being able to move forward in it's current state. I unfortunately will have to step away from this project for some time. Therefore, I will update the description of the project, focused on providing a TCN-focused rebuild of GTCRN. The PyTorch and ONNX model are viable to use, however they do not outperform GTCRN, so I recommend using the original model.

I am still running some final evaluations for the streaming models and will update with these metrics once they're done. For now, for performance on the TCN focused architecture, see [gtcrn_micro](./gtcrn_micro/README.md).

##### *Update 12/26/2025:*

New model has been trained with a few architecture changes. The main changes are: Streaming-compatible dilation and padding, Lite versions of the TRA and SFE, which are quantizable, were added, and the GTCN section's dilation was fixed. Results for this model can be found in the [gtcrn_micro](./gtcrn_micro/README.md) directory.

Next step is to use the created Streaming architecture to:
 - 1. Test performance for the PyTorch streaming variant
 - 2. Export the streaming variant to ONNX and test performance
 - 3. ~~Export ONNX streaming varian to TFLite and test performance.~~

~~Once those are completed, this will move to MCU deployment tests, targeting the **ESP32-S3** first.~~


##### *Update 12/19/2025:*

  Need to adjust the model architecture for better streaming performance. This model will be the one that goes through the full PyTorch $\rightarrow$ ONNX $\rightarrow$ TFLite conversion. Currently training said model that will support this hardware in streaming form. 

Future goals are to attempt to skip the `tflite` conversion and use STM32CubeAI to deploy.

*For updates, check out either the issues, or the project roadmap [here](./docs/plan.md) and [here](./docs/TODO.md). Please submit any issues if you catch any. It's greatly appreciated!*

- - -
## Acknowledgements

###### 1. The original model this is based off of is [GTCRN: A Speech Enhancement Model Requiring Ultralow Computational Resources](https://ieeexplore.ieee.org/document/10448310), leveraging the implementation code at [GTCRN](https://github.com/Xiaobin-Rong/gtcrn). A notable amount of the setup to train and change the model was based off of the same authors project [SEtrain](https://github.com/Xiaobin-Rong/SEtrain/tree/plus). They have some seriously impressive SE research! Please check out their research and throw some of their work a star!
###### 2. The project also requires moving from **PyTorch $\rightarrow$ ONNX $\rightarrow$ .tflite** to run inference on ESP32s. None of this could have been possible without the direct help and work of [PINTO0309](https://github.com/PINTO0309) & their awesome project [onnx2tf](https://github.com/PINTO0309/onnx2tf). I highly recommend you check out their work if you are reading this and want to do a similar project. Please consider throwing some of their work a star!
###### 3. The SFE and TRA were changed following an implementation in [gtcrn-light](https://github.com/zerong7777-boop/gtcrn-light). Their implementation managed to preserve these blocks functionality without the GRUs, making it representable in TFLM without an external CPU. Their implementation is great, and I recommend you check it out. 
- - - 

*This project would not have been possible without their efforts. Please consider citing them and giving them a star first before this project's!* 

- - - 
