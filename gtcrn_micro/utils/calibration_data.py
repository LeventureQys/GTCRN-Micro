# used for onnx2tf.sh
# ----------------
from __future__ import print_function

import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from numpy.typing import NDArray
from tqdm import tqdm

from gtcrn_micro.models.gtcrn_micro import GTCRNMicro
from gtcrn_micro.streaming.conversion.convert import convert_to_stream
from gtcrn_micro.streaming.gtcrn_micro_stream import StreamGTCRNMicro

# CONSTANTS FOR STFT INFO
N_FFT = 512
HOP = 256


# function to generate stft tensors
def wav_2_tensor(wav_path: Path) -> torch.Tensor:
    """Transform input to np tensor for model input.

    Args:
        wav_path (Path): Path to input wavs

    Returns:
        torch.Tensor: STFT of wav

    """
    mix, fs = sf.read(
        str(wav_path),
        dtype="float32",
    )

    assert fs == 16000, f"Expected 16kHz, got {fs}"

    stft = torch.stft(
        torch.from_numpy(mix),
        N_FFT,
        HOP,
        N_FFT,
        torch.hann_window(N_FFT).pow(0.5),
        return_complex=False,
    )[None]
    print("-" * 20)
    print("STFT Streaming Shape:\n")
    print(stft.shape)

    # specifically for the tf model conversion we need to transpose the inputs
    # stft_np = stft.numpy().transpose(1, 2, 0)
    # fixing for TF NHWC
    # stft_np = stft.numpy().transpose(1, 2, 0)  # -> (T, F, 2)
    # stft = stft.permute(1, 2, 0)
    return stft


def scale_write(data: NDArray[np.float32], output_path: Path, file_name: str):
    """Write stacked npy data to .npy files for quantization calibration. Also generates and calculates scale.

    Args:

        data (NDArray[np.float32]): Input concat data for calibration.
        output_path (Path): Directory of calibration data to write to.
        file_name (str): File name to append to .npy file and .txt scale file.
    """
    if not os.path.exists(output_path):
        print("\nCreating output path...\n")
        os.makedirs(output_path)
    print("\n**********\nData info\n**********")
    # debugging:
    print(data.shape, data.dtype)
    print(f"Min/max: {data.min(), data.max()}")
    print(f"po1/p99: {np.percentile(data, 1), np.percentile(data, 99)}")
    print("**********")

    # clipping data for quantization
    a = np.percentile(np.abs(data), 99.99)
    scale = 2.0 * a * 1.06
    # scale_low = np.percentile(data, 0.1)
    # scale_high = np.percentile(data, 99.9)
    # scale = max(abs(scale_low), abs(scale_high)) * 2.0

    # clipped_data = np.clip(data / scale + 0.5, 0.0, 1.0).astype(np.float32)

    data_norm = (data / scale) + 0.5
    data_norm = np.clip(data_norm, 0.0, 1.0).astype(np.float32)
    npy_filename = file_name + ".npy"
    npy_file = Path(os.path.join(output_path, npy_filename))
    np.save(npy_file, data_norm)
    print(f"Scale = {scale}")
    print("**********\nCalibration info\n**********")
    x = np.load(npy_file)
    print(f"Shape = {x.shape}")
    print(f"Min/max: {x.min(), x.max()}")
    print(f"po1/p99: {np.percentile(x, 1), np.percentile(x, 99)}")
    print("**********")

    # writing the scale to use in the onnx2tf conversion
    txt_filename = file_name + ".txt"
    txt_file = Path(os.path.join(output_path, txt_filename))
    txt_file.write_text(f"{scale}\n")
    print(f"Wrote the scale to {txt_file}: {scale}\n")


def gen_calib_data(
    stream_model: nn.Module,
    warmup: int,
    n_samples: int,
    calib_data: Path,
    output_path: Path,
):
    """Generate calibration data set by input wav."""
    # getting .wav files in directory
    wavs = sorted(calib_data.glob("*.wav"))[:n_samples]

    # initializing all input data
    audio_samples = []
    conv_samples = []
    tra_samples = []
    tcn_samples = [[] for _ in range(8)]
    for i in wavs:
        # initializing the caches:
        conv_cache = torch.zeros(2, 1, 16, 6, 33)
        tra_cache = torch.zeros(2, 3, 1, 8, 2)
        tcn_cache = [
            [torch.zeros(1, 16, 2 * d, 33) for d in [1, 2, 4, 8]],
            [torch.zeros(1, 16, 2 * d, 33) for d in [1, 2, 4, 8]],
        ]

        # appending the tensor stft to data list
        stft_np = wav_2_tensor(i)

        # iterating through each time frame
        for frame in tqdm(range(stft_np.shape[2])):
            # permuted: (T, F, 2)
            # audio_frame = stft_np[frame : frame + 1, :, :]
            audio_frame = stft_np[:, :, frame : frame + 1]
            # for frame in range(max_frames):
            #     # increment and get current frame
            #     audio_frame = stft_np[frame : frame + 1, :, :]
            # print("\nAudio frame: ", audio_frame)
            # print("\nAudio shape: ", audio_frame.shape)

            # only generate until n_samples
            if frame >= warmup and len(audio_samples) < n_samples:
                audio_samples.append(audio_frame.numpy())
                conv_samples.append(conv_cache.numpy())
                tra_samples.append(tra_cache.numpy())
                # flattening both the two GTCN blocks to iter
                full_tcn = tcn_cache[0] + tcn_cache[1]

                for k in range(8):
                    # need to transpose for TF (B, C, kT, F) -> (B, kT, F, C)
                    tcn_samples[k].append(full_tcn[k].permute(0, 2, 3, 1).numpy())

            with torch.no_grad():
                y, conv_cache, tra_cache, tcn_cache = stream_model(
                    audio_frame, conv_cache, tra_cache, tcn_cache
                )

    # NOTE: CHECK SHAPES
    # stacking and saving the data to write
    audio_data = np.concatenate(audio_samples, axis=0).astype(np.float32)
    scale_write(data=audio_data, output_path=output_path, file_name="audio")
    conv_data = np.concatenate(conv_samples, axis=0).astype(np.float32)
    scale_write(data=conv_data, output_path=output_path, file_name="conv_cache")
    tra_data = np.concatenate(tra_samples, axis=0).astype(np.float32)
    scale_write(data=tra_data, output_path=output_path, file_name="tra_cache")
    for c in range(8):
        scale_write(
            data=np.concatenate(tcn_samples[c], axis=0).astype(np.float32),
            output_path=output_path,
            file_name=f"tcn_cache_{c}",
        )


if __name__ == "__main__":
    # starting with basic data calibration
    calib_data = Path(
        "./gtcrn_micro/data/DNS3/noisy_blind_testset_v3_challenge_withSNR_16k/"
    )
    ouput_path = Path("./gtcrn_micro/streaming/tflite/calibration_data/")
    warmup = 63
    # max_frames = 973
    n = 30
    # convert model to streaming
    device = torch.device("cpu")
    model = GTCRNMicro().to(device).eval()
    model.load_state_dict(
        torch.load("./gtcrn_micro/ckpts/best_model_dns3.tar", map_location=device)[
            "model"
        ]
    )
    stream_model = StreamGTCRNMicro().to(device).eval()
    convert_to_stream(stream_model, model)
    gen_calib_data(
        stream_model=stream_model,
        warmup=warmup,
        n_samples=n,
        calib_data=calib_data,
        output_path=ouput_path,
    )
