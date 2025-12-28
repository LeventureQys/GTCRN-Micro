import os

import onnx
import torch
import torch.nn as nn
from onnxsim import simplify


def stream2onnx(
    stream_model: nn.Module,
    sample_input: torch.Tensor,
    conv_cache: torch.Tensor,
    tra_cache: torch.Tensor,
    tcn_cache: list[list[torch.Tensor]],
    model_name: str,
) -> None:
    """Convert Torch model to .onnx.

    Args:
        stream_model (nn.Module): Streaming model to convert to onnx
        sample_input (torch.Tensor): Sample small input for conversion
        conv_cache (torch.Tensor): Convolution cache used for streaming input
        tra_cache (torch.Tensor): TRA Lite cache used for streaming input
        tcn_cache (list[list[torch.Tensor]]): TCN cache used for streaming input
        model_name (str): Name of onnx file that will be saved - "name".onnx
    """
    ONNX_PATH = "./gtcrn_micro/streaming/onnx/"

    # creating all of the TCN caches:
    tcn_in_names = [f"tcn_cache_{k}" for k in range(len(tcn_cache[0]) * 2)]
    tcn_out_names = [f"tcn_cache_out_{k}" for k in range(len(tcn_cache[0]) * 2)]

    print("starting onnx export:")
    torch.onnx.export(
        stream_model,
        (sample_input, conv_cache, tra_cache, tcn_cache),  # Exporting with small input
        f"{ONNX_PATH}{model_name}.onnx",
        opset_version=16,  # Lowerin opset for LN
        dynamo=False,
        input_names=["audio", "conv_cache", "tra_cache", *tcn_in_names],
        output_names=["mask", "conv_cahe_out", "tra_cache_out", *tcn_out_names],
        dynamic_axes=None,
        export_params=True,
        do_constant_folding=True,
        report=True,
    )

    # checking the model
    onnx_model = onnx.load(f"{ONNX_PATH}{model_name}.onnx")
    onnx.checker.check_model(onnx_model)
    # print ONNX input shape
    # print("Onnx input shape:\n----------")
    # for i in onnx_model.graph.input:
    #     print("\n", i.type.tensor_type.shape)

    # simplifying the model for onnx2tf
    if not os.path.exists(
        f"{ONNX_PATH}{model_name}.onnx".split(".onnx")[0] + "_simple.onnx"
    ):
        model_simplified, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(
            model_simplified,
            f"{ONNX_PATH}{model_name}.onnx".split(".onnx")[0] + "_simple.onnx",
        )


#
# if __name__ == "__main__":
#     # loading model
#     cfg_infer = OmegaConf.load("gtcrn_micro/conf/cfg_infer.yaml")
#     cfg_network = OmegaConf.load(cfg_infer.network.config)
#     model = GTCRNMicro(**cfg_network["network_config"])
#
#     # loading test data
#     mix, fs = sf.read(
#         "./gtcrn_micro/data/DNS3/noisy_blind_testset_v3_challenge_withSNR_16k/ms_realrec_emotional_female_SNR_17.74dB_headset_A2AHXGFXPG6ZSR_Water_far_Laughter_12.wav",
#         dtype="float32",
#     )
#
#     ckpt = "./gtcrn_micro/ckpts/best_model_dns3.tar"
#     stream2onnx(model, mix, model_name="gtcrn_micro", checkpoint=ckpt)
