import numpy as np
import onnxruntime
import soundfile as sf
import torch
from librosa import istft
from tqdm import tqdm

from gtcrn_micro.models.gtcrn_micro import GTCRNMicro
from gtcrn_micro.streaming.conversion.convert import convert_to_stream
from gtcrn_micro.streaming.gtcrn_micro_stream import StreamGTCRNMicro


def output_test() -> None:
    # loading data
    mix, fs = sf.read(
        "./gtcrn_micro/data/DNS3/noisy_blind_testset_v3_challenge_withSNR_16k/ms_realrec_nonenglish_female_SNR_23.01dB_headset_10_spanish_1.wav",
        dtype="float32",
    )
    assert fs == 16000, f"Expected fs of 16000, instead got {fs}"
    x = torch.from_numpy(mix)
    # running stft
    x = torch.stft(
        x, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False
    )[None]
    # Check PyTorch output
    # load state dict from checkpoint
    model = GTCRNMicro()
    ckpt = torch.load(
        "./gtcrn_micro/ckpts/best_model_dns3.tar",
        map_location="cpu",
    )

    state = (
        ckpt.get("state_dict", None)
        or ckpt.get("model_state_dict", None)
        or ckpt.get("model", None)
        or ckpt
    )
    # Handling if ckpt was saved from DDP and has module prefixes
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.removeprefix("module."): v for k, v in state.items()}

    # print state dict info
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("-" * 20)
    print(f"\tmissing keys: {missing}")
    print(f"\tunexpected keys: {unexpected}")

    # explicitly setting model to eval in function
    device = torch.device("cpu")
    model.eval()
    model.to("cpu")
    stream_model = StreamGTCRNMicro().to(device).eval()
    convert_to_stream(stream_model, model)

    conv_cache = torch.zeros(2, 1, 16, 6, 33).to(device)
    tra_cache = torch.zeros(2, 3, 1, 8, 2).to(device)
    tcn_cache = [
        [torch.zeros(1, 16, 2 * d, 33, device=device) for d in [1, 2, 4, 8]],
        [torch.zeros(1, 16, 2 * d, 33, device=device) for d in [1, 2, 4, 8]],
    ]
    ys = []
    for i in tqdm(range(x.shape[2])):
        xi = x[:, :, i : i + 1]
        with torch.no_grad():
            yi, conv_cache, tra_cache, tcn_cache = stream_model(
                xi, conv_cache, tra_cache, tcn_cache
            )
        ys.append(yi)
    ys = torch.cat(ys, dim=2)

    enhanced_stream = torch.view_as_complex(ys.contiguous())
    enhanced_stream = torch.istft(
        enhanced_stream,
        512,
        256,
        512,
        torch.hann_window(512).pow(0.5),
        return_complex=False,
    )
    enhanced_pytorch = enhanced_stream.squeeze(0).cpu().numpy()

    # check onnx output
    # session = onnxruntime.InferenceSession(
    #     "./gtcrn_micro/streaming/onnx/gtcrn_micro.onnx"
    # )
    # onnx_output = session.run(
    #     ["mask"],
    #     {
    #         "audio": input.numpy(),
    #     },
    # )
    # print("\nonnx output done")

    onnx_file = "./gtcrn_micro/streaming/onnx/gtcrn_micro_stream.onnx"
    session = onnxruntime.InferenceSession(
        onnx_file.split(".onnx")[0] + "_simple.onnx",
        None,
        providers=["CPUExecutionProvider"],
    )
    # re-init the caches
    conv_cache = np.zeros([2, 1, 16, 6, 33], dtype="float32")
    tra_cache = np.zeros([2, 3, 1, 8, 2], dtype="float32")
    tcn_caches = [
        np.zeros((1, 16, 2, 33), dtype=np.float32),
        np.zeros((1, 16, 4, 33), dtype=np.float32),
        np.zeros((1, 16, 8, 33), dtype=np.float32),
        np.zeros((1, 16, 16, 33), dtype=np.float32),
        np.zeros((1, 16, 2, 33), dtype=np.float32),
        np.zeros((1, 16, 4, 33), dtype=np.float32),
        np.zeros((1, 16, 8, 33), dtype=np.float32),
        np.zeros((1, 16, 16, 33), dtype=np.float32),
    ]

    outputs = []

    print("\nRunning ONNX inference...\n")

    inputs = x.numpy()
    for i in tqdm(range(inputs.shape[-2])):
        output_onnx = session.run(
            None,
            {
                "audio": inputs[..., i : i + 1, :],
                "conv_cache": conv_cache,
                "tra_cache": tra_cache,
                **{f"tcn_cache_{k}": tcn_caches[k] for k in range(8)},
            },
        )
        # need to unpack the onnx outputs
        out_i = output_onnx[0]
        conv_cache = output_onnx[1]
        tra_cache = output_onnx[2]
        tcn_caches = output_onnx[3:]

        outputs.append(out_i)

    outputs = np.concatenate(outputs, axis=2)
    enhanced_onnx = istft(
        outputs[..., 0] + 1j * outputs[..., 1],
        n_fft=512,
        hop_length=256,
        win_length=512,
        window=np.hanning(512) ** 0.5,
    )

    # TFLite
    ## Load tflite model and compare outputs
    # tflite_path = "./gtcrn_micro/streaming/tflite/gtcrn_micro_full_integer_quant.tflite"
    # # tflite_path = "./gtcrn_micro/streaming/tflite/gtcrn_micro_float32.tflite"
    # interpreter = tf.lite.Interpreter(model_path=tflite_path)
    # input_data1 = input.permute(0, 2, 3, 1).detach().numpy().astype(np.float32)

    # print("\n------------\nTFLite shape check:\n")
    # print(">>INPUTS<<")
    # for d in interpreter.get_input_details():
    #     print(f"{d['name']}, shape: {d['shape']}\ndtype: {d['dtype']}")
    #
    # for d in interpreter.get_output_details():
    #     print("\n>>OUTPUTS<<")
    #     print(f"{d['name']}, shape: {d['shape']}\ndtype: {d['dtype']}")

    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()
    #
    # interpreter.resize_tensor_input(
    #     input_details[0]["index"], input_data1.shape, strict=True
    # )
    # interpreter.allocate_tensors()
    #
    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()
    #
    # print(
    #     "in quant:",
    #     input_details[0]["quantization"],
    #     input_details[0]["quantization_parameters"],
    # )
    # print(
    #     "out quant:",
    #     output_details[0]["quantization"],
    #     output_details[0]["quantization_parameters"],
    # )
    #
    # # fix input scale
    # in_scale, in_zero = input_details[0]["quantization"]
    # out_scale, out_zero = output_details[0]["quantization"]
    #
    # # setting input data to match the input details shape and size
    # if input_details[0]["dtype"] == np.int8:
    #     q = np.round(input_data1 / in_scale + in_zero)
    #
    #     sat_lo = np.mean(q < -128)
    #     sat_hi = np.mean(q > 127)
    #     print(f"saturation lo%: {sat_lo * 100:.4f}  hi%: {sat_hi * 100:.4f}")
    #
    #     float_min = input_data1.min()
    #     float_max = input_data1.max()
    #     q_float_min = (-128 - in_zero) * in_scale
    #     q_float_max = (127 - in_zero) * in_scale
    #     print("float min/max:", float_min, float_max)
    #     print("quant float range:", q_float_min, q_float_max)
    #
    #     print("pre-clip q min/max:", q.min(), q.max())
    #     q = np.clip(q, -128, 127)
    #     x_q = q.astype(np.int8)
    #     # x_q = np.round(input_data1 / in_scale + in_zero).astype(np.int8)
    # else:
    #     x_q = input_data1
    #
    # interpreter.set_tensor(input_details[0]["index"], x_q)
    # interpreter.invoke()
    #
    # y_q = interpreter.get_tensor(output_details[0]["index"])
    #
    # # comparing quant version of pytorch ouput
    # p = pytorch_output.astype(np.float32)
    # p_q = np.round(p / out_scale + out_zero)
    # p_q = np.clip(p_q, -128, 127).astype(np.int8)
    # print("pytorch out min/max:", p.min(), p.max())
    # print("pytorch out p1/p99:", np.percentile(p, 1), np.percentile(p, 99))
    #
    # int8_mae = np.mean(np.abs(p_q.astype(np.int16) - y_q.astype(np.int16)))
    # print("INT8-domain MAE (counts):", int8_mae)
    #
    # print("\noutput shape: ", y_q.shape, "dtype: ", y_q.dtype)
    # # dequantizing for comparison
    # if output_details[0]["dtype"] == np.int8:
    #     tflite_output = (y_q.astype(np.float32) - out_zero) * out_scale
    # else:
    #     tflite_output = y_q.astype(np.float32)
    #
    # print("\n-------------")
    #
    # print("\nTFLite output done\n")
    print("*" * 10)
    print("\nOUTPUT STATS\n")

    print(
        f"Onnx outputs error vs pytorch: {np.mean(np.abs(enhanced_onnx - enhanced_pytorch))}"
    )
    diff_onnx = enhanced_onnx - enhanced_pytorch
    abs_diff = np.abs(diff_onnx)

    print("onnx MAE:", abs_diff.mean())
    print("onnx median abs diff:", np.median(abs_diff))
    # print(
    #     f"Tflite outputs error vs pytorch: {np.mean(np.abs(pytorch_output - tflite_output))}"
    # )
    # print(
    #     f"Tflite outputs error vs onnx: {np.mean(np.abs(onnx_output[0] - tflite_output))}"
    # )
    # diff_tflite = tflite_output - pytorch_output
    # abs_diff_t = np.abs(diff_tflite)
    # print("TFLite MAE:", abs_diff_t.mean())
    # print("TFLite median abs diff:", np.median(abs_diff_t))

    print("DONE\n")
    print("*" * 10)


if __name__ == "__main__":
    output_test()
