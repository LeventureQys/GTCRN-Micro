import numpy as np
import onnxruntime
import soundfile as sf
import torch
from librosa import istft
from tqdm import tqdm

from gtcrn_micro.models.gtcrn_micro import GTCRNMicro
from gtcrn_micro.streaming.conversion.convert import convert_to_stream
from gtcrn_micro.streaming.gtcrn_micro_stream import StreamGTCRNMicro
from gtcrn_micro.utils.tflite_utils import tflite_stream_infer


def output_test() -> None:
    """Test output of trained (streaming) model in different formats."""
    # loading data
    mix, fs = sf.read(
        "./gtcrn_micro/streaming/sample/noisy1.wav",
        # "./gtcrn_micro/streaming/sample/noisy2.wav",
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
    # for f-domain comparison
    p_stft = ys.squeeze(0).cpu().numpy()

    enhanced_stream = torch.istft(
        enhanced_stream,
        512,
        256,
        512,
        torch.hann_window(512).pow(0.5),
        return_complex=False,
    )
    enhanced_pytorch = enhanced_stream.squeeze(0).cpu().numpy()
    sf.write(
        "gtcrn_micro/streaming/sample/enh_torch1.wav",
        enhanced_pytorch,
        16000,
    )

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
    o_stft = outputs
    enhanced_onnx = istft(
        outputs[..., 0] + 1j * outputs[..., 1],
        n_fft=512,
        hop_length=256,
        win_length=512,
        window=np.hanning(512) ** 0.5,
    )
    sf.write(
        "gtcrn_micro/streaming/sample/enh_onnx1.wav",
        enhanced_onnx.squeeze(),
        16000,
    )

    # TFLite
    # Load tflite model and compare outputs
    tflite_path = (
        "gtcrn_micro/streaming/tflite/gtcrn_micro_stream_simple_float16.tflite"
    )
    # tflite_path = "gtcrn_micro/streaming/tflite/gtcrn_micro_stream_simple_dynamic_range_quant.tflite"
    tflite_stft = tflite_stream_infer(x, model_path=tflite_path)
    t_stft = tflite_stft
    enhanced_tflite = istft(
        tflite_stft[..., 0] + 1j * tflite_stft[..., 1],
        n_fft=512,
        hop_length=256,
        win_length=512,
        window=np.hanning(512) ** 0.5,
    )
    sf.write(
        "gtcrn_micro/streaming/sample/enh_tflite_float16_1.wav",
        # "gtcrn_micro/streaming/sample/enh_tflite_dynamic_range1.wav",
        enhanced_tflite.squeeze(),
        16000,
    )

    print("\n-------------")

    print("\nTFLite output done\n")
    print("*" * 10)
    print("\nOUTPUT STATS\n")
    print("\nF-Domain:\n")
    print(f"STFT MAE ONNX vs PT: {np.mean(np.abs(o_stft - p_stft))}")
    print(f"STFT MAE TFLite vs PT: {np.mean(np.abs(t_stft - p_stft))}")
    print(f"STFT MAE TFLite vs ONNX: {np.mean(np.abs(t_stft - o_stft))}")

    m = np.mean(np.abs(t_stft - p_stft), axis=(0, 1, 3))
    print(
        f"TFLite vs PT frame MAE start - mid - end: {m[0]} - {m[len(m) // 2]} - {m[-1]}"
    )

    print("\nTime-Domain:\n")
    print(
        f"Onnx outputs error vs pytorch: {np.mean(np.abs(enhanced_onnx - enhanced_pytorch))}"
    )
    diff_onnx = enhanced_onnx - enhanced_pytorch
    abs_diff = np.abs(diff_onnx)

    print("onnx MAE:", abs_diff.mean())
    print("onnx median abs diff:", np.median(abs_diff))

    print(
        f"TFLite outputs error vs pytorch: {np.mean(np.abs(enhanced_tflite - enhanced_pytorch))}"
    )
    print(
        f"TFLite outputs error vs onnx: {np.mean(np.abs(enhanced_tflite - enhanced_onnx))}"
    )
    diff_tflite = enhanced_tflite - enhanced_pytorch
    abs_diff_t = np.abs(diff_tflite)
    print("TFLite MAE:", abs_diff_t.mean())
    print("TFLite median abs diff:", np.median(abs_diff_t))

    print("*" * 10)


if __name__ == "__main__":
    output_test()
