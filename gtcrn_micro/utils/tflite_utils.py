from pathlib import Path
from typing import Any, Tuple, Union

import numpy as np
import numpy.typing as npt
import soundfile as sf
import tensorflow as tf
import torch
from tqdm import tqdm


def _pick(details: dict[str, Any], key: str, default_idx: int = None):
    """Search through tensor list and return the tensor whose name matches the key.

    Args:
        details (dict[str, Any]): Input (or output) details of tensors
        key (str): Key name of input to match to a tensor index
        default_idx (int): Default index to fall back on if tensor isn't found

    Returns:
        Index

    Raises:
        KeyError: Unable to find key
    """
    key = key.lower()
    for d in details:
        if key in d["name"].lower():
            return d

    if default_idx is None:
        raise KeyError(f"Couldnt find tensor conatining: {key}")
    return details[default_idx]


def _get_qparams(detail: dict[str, Any]):
    """Return (scales, zero_points, qdim).

    Args:
        detail (dict[str, Any]): Tensor details

    Returns:
        npt.NDArray: array of scales, zeropoints, and qdim
    """
    qp = detail.get("quantization_parameters", {})
    scales = qp.get("scales", None)
    zero_points = qp.get("zero_points", None)
    qdim = qp.get("quantized_dimension", 0)

    if scales is None or zero_points is None:
        # fallback to deprecated version of doing this
        s, zp = detail.get("quantization", (0.0, 0))
        return np.array([s], dtype=np.float32), np.array([zp], dtype=np.int32), 0

    return np.asarray(scales), np.asarray(zero_points), qdim


def _get_tensor_qparams(detail: dict[str, Any]):
    """Returns scale, zeropoint for per-tensor quant, else raise ValueError.

    Args:
        detail (dict[str, Any]): Tensor details

    Returns:
        Any, Any: Scale and Zeropoint

    Raises:
        ValueError: If passed not per-tensor quantization
    """
    scales, zps, qdim = _get_qparams(detail)

    if scales.size == 0:
        return None, None  # not quantized

    if scales.size != 1 or zps.size != 1:
        raise ValueError(
            f"Expected per-tensor quantization, got per-axis: "
            f"scales={scales.shape}, zps={zps.shape}, qdim={_get_qparams(detail)[2]}"
        )

    return float(scales[0]), int(zps[0])


def _quantize(
    x_f32: Union[np.float32, np.int8], detail: dict[str, Any]
) -> Union[np.float32, np.int8]:
    """Quantize inputs to int8 if needed for int8 model.

    Args:
        x_f32 (Union[np.float32, np.int8]): Input in np float32 format
        detail (dict[str, Any]): Tensor details

    Returns:
        Union[np.float32, np.int8]: Either pass-through float32 input or quantized int8 input
    """
    # pass float32 if that's what is expected
    if detail["dtype"] == np.float32:
        return x_f32.astype(np.float32)

    # quantize if we want int8
    if detail["dtype"] == np.int8:
        scale, zero_point = _get_tensor_qparams(detail)
        if scale is None:
            return x_f32.astype(np.int8)
        # q = round(x / scale + zero point)
        q = np.round(x_f32 / scale + zero_point)
        q = np.clip(q, -128, 127).astype(np.int8)
        return q


def _dequantize(y: Union[np.float32, np.int8], detail: dict[str, Any]) -> np.float32:
    """De-quantize with output tensor values.

    Args:
        y (Union[np.float32, np.int8]): Quantized output
        detail (dict[str, Any]): Tensor details

    Returns:
        np.float32: Dequantized output.
    """
    if detail["dtype"] == np.float32:
        return y.astype(np.float32)

    # reverse quantization: x = (q - zero_point) * scale
    if detail["dtype"] == np.int8:
        scale, zero_point = _get_tensor_qparams(detail)
        if scale is None:
            return y.astype(np.float32)
        return (y.astype(np.float32) - zero_point) * scale

    return y.astype(np.float32)


def _zero_like_input(detail: dict[str, Any]) -> npt.NDArray:
    """Fills caches with either zeros or the tensor's zero-point.

    Args:
        detail (dict[str, Any]): Tensor details

    Returns:
        npt.NDArray: Array of zeros or zero-point
    """
    shape = tuple(detail["shape"])
    dt = detail["dtype"]
    # if int8 cache, zero is the zero-point
    if dt == np.int8:
        scale, zero_point = _get_tensor_qparams(detail)
        if scale is None:
            return np.zeros(shape, dtype=np.int8)
        return np.full(shape, int(zero_point), dtype=np.int8)

    return np.zeros(shape, dtype=dt)


def _frame_tflite_layout(input_frame: npt.NDArray, expected_shape: npt.NDArray):
    """Adjust layout to fit expected TF shape, typically NCHW.

    Args:
        input_frame (npt.NDArray): Input frame array
        expected_shape (npt.NDArray): Expected shape to match

    Returns:
        (npt.NDArray): Re-shaped frame for inference.

    Raises:
        ValueError: Cannot map frame to expected shape.
    """
    # expected shape: (B, F, T, C) -> (B, C, F, T)
    es = tuple(expected_shape)
    if es == tuple(input_frame.shape):
        return input_frame
    if len(es) == 4:
        if (
            es[1] == 1
            and es[2] == input_frame.shape[1]
            and es[3] == input_frame.shape[3]
        ):
            # (B, 1, F, 2)
            return np.transpose(input_frame, (0, 2, 1, 3))

        if (
            es[1] == 1
            and es[2] == input_frame.shape[3]
            and es[3] == input_frame.shape[1]
        ):
            # (B, 1, 2, F)
            return np.transpose(input_frame, (0, 2, 3, 1))

    if np.prod(es) == input_frame.size:
        return input_frame.reshape(es)
    raise ValueError(f"Can't map frame of {input_frame.shape} -> expected {es}")


def _y_from_tflite(
    y_tflite: np.float32, expected_shape: npt.NDArray, canonical_shape: Tuple
):
    """Returns to original audio shape for comparison.

    Args:
        y_tflite (np.float32): Y output that has be de-quantized
        expected_shape (npt.NDArray): Expected shape from TFLite (not final one)
        canonical_shape (Tuple): Shape to return to for output comparison (PyTorch order)

    Returns:
        (npt.NDArray): Reshaped output to match PyTorch ordering for comparison

    Raises:
        ValueError: Unable to map the tflite shape to the canonical shape.
    """
    es = tuple(expected_shape)
    cs = tuple(canonical_shape)
    if tuple(y_tflite.shape) == cs:
        return y_tflite
    if tuple(y_tflite.shape) == es:
        if es == (cs[0], 1, cs[1], cs[3]):
            return np.transpose(y_tflite, (0, 2, 1, 3))
        if es == (cs[0], 1, cs[3], cs[1]):
            return np.transpose(y_tflite, (0, 3, 1, 2))

    if y_tflite.size == np.prod(cs):
        return y_tflite.reshape(cs)
    raise ValueError(f"Unable to map y {y_tflite.shape} -> canon {cs}")


def _tcn_cache_out_read(cache_out: npt.NDArray, in_shape: npt.NDArray) -> npt.NDArray:
    """Fix tcn permutation caused my tflite inference.

    Args:
        cache_out (npt.NDArray): cache ouput
        in_shape (npt.NDArray): input shape for cache

    Returns:
        (npt.NDArray): Transposed output shape for next pass of interence.

    Raises:
        ValueError: Incompatible input shapes that can't be transposed correctly.
    """
    in_shape = tuple(in_shape)

    if cache_out.shape == in_shape:
        return cache_out

    # fixing bug to make next in (B, C, T, F)
    if cache_out.ndim == 4:
        B, C, T, F = in_shape
        if cache_out.shape == (B, T, F, C):
            return np.transpose(cache_out, (0, 3, 1, 2))

    if cache_out.size == int(np.prod(in_shape)):
        return cache_out.reshape(in_shape)

    raise ValueError(f"TCN cache shape issue: out={cache_out.shape} - in={in_shape}")


def _to_tcn_layout(tcn_cache_raw: npt.NDArray) -> npt.NDArray:
    """Convert TCN cache output to the cache input layout: (1, 16, T, 33)

    Args:
        tcn_cache_raw (npt.NDArray): Raw output caches

    Returns:
        (npt.NDArray): Correct ordered cache

    Raises:
        ValueError: Wrong cache layout
    """
    if tcn_cache_raw.ndim != 4:
        raise ValueError(f"Expected 4d TCN cache, got {tcn_cache_raw.shape}")

    if tcn_cache_raw.shape[1] == 16 and tcn_cache_raw.shape[-1] == 33:
        return tcn_cache_raw

    if tcn_cache_raw.shape[-1] == 16 and tcn_cache_raw.shape[2] == 33:
        return np.transpose(tcn_cache_raw, (0, 3, 1, 2))

    raise ValueError(f"Unknown TCN Cache layout: {tcn_cache_raw.shape}")


def _reorder_tcn_caches(
    new_outs: list[npt.NDArray], old_caches: list[npt.NDArray], tcn_ins: list[dict]
) -> list[npt.NDArray]:
    """Reorder TCN cache outputs to match the next inputs.

    Args:
        new_outs (list[npt.NDArray]): list of new output caches
        old_caches (list[npt.NDArray]): list of old output caches
        tcn_ins (list[dict]): tcn in details

    Returns:
        (list[npt.NDArray]): Correctly ordered TCN outputs

    Raises:
        ValueError: Issue in number of cache slots
    """
    # order by diff time amounts per tcn (2, 4, 8, 16)
    time_slots: dict[int, list[int]] = {}
    for idx, d in enumerate(tcn_ins):
        T = int(d["shape"][2])
        time_slots.setdefault(T, []).append(idx)

    # building candidates from outputs
    time_candidates: dict[int, list[np.ndarray]] = {}
    for arr in new_outs:
        T = int(arr.shape[2])
        time_candidates.setdefault(T, []).append(arr)

    out = [None] * 8
    for T, slots in time_slots.items():
        if len(slots) != 2:
            raise ValueError(f"Expected 2 cache sltos for T={T}, got {slots}")
        cands = time_candidates.get(T, [])
        if len(cands) != 2:
            raise ValueError(f"Expected 2 cache candidates for T={T}, got {len(cands)}")

        a, b = slots
        new0, new1 = cands

        # compare whether to swap TCN positions or not
        noswap = np.mean(np.abs(new0 - old_caches[a])) + np.mean(
            np.abs(new1 - old_caches[b])
        )
        swap = np.mean(np.abs(new0 - old_caches[b])) + np.mean(
            np.abs(new1 - old_caches[a])
        )

        # swapping if necessary to align tcn caches
        # assign by continuity compared to old caches
        if swap < noswap:
            out[a] = new1
            out[b] = new0
        else:
            out[a] = new0
            out[b] = new1

    return out


def tflite_stream_infer(x: torch.Tensor, model_path: Path):
    """Run streaming tflite input for comparison and evaluation.

    Args:
        x (torch.Tensor): STFT input
        model_path (Path): Path to Tflite Model

    Returns:
        Frequency-domain tflite ouptut concatenated at time-axis
    """
    # grabbing the tensor metadata from TFLite
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    in_details = interpreter.get_input_details()
    out_details = interpreter.get_output_details()

    # uncomment for checking interpreter details
    # print("-" * 20)
    # print("INTERPRETER DETAILS\n")
    # print("\nIN DETAILS:\n")
    # for i in in_details:
    #     print(f"{i}\n")
    # print("*" * 20)
    # print("*" * 20)
    # print("\nOUT DETAILS:\n")
    # for o in out_details:
    #     print(f"{o}\n")
    # print("-" * 20)

    # getting the input tensor indexes
    audio_in = _pick(in_details, "audio", default_idx=0)
    conv_in = _pick(in_details, "conv_cache", default_idx=1)
    tra_in = _pick(in_details, "tra_cache", default_idx=2)
    tcn_ins = []
    for k in range(8):
        tcn_ins.append(_pick(in_details, f"tcn_cache_{k}", default_idx=3 + k))

    # similarly, getting the output tensors
    audio_out = _pick(out_details, "Identity", default_idx=0)
    conv_out = _pick(out_details, "Identity_1", default_idx=1)
    tra_out = _pick(out_details, "Identity_2", default_idx=2)
    tcn_outs = []
    for k in range(8):
        tcn_outs.append(_pick(out_details, f"Identity_{k + 3}", default_idx=3 + k))

    # getting cache inputs
    conv_cache = _zero_like_input(conv_in)
    tra_cache = _zero_like_input(tra_in)
    tcn_cache_list = [_zero_like_input(d) for d in tcn_ins]

    # set the input for streaming
    x_np = x.numpy()
    T = x_np.shape[2]
    y_frames = []
    print("\nTFLite STREAMING inference...\n")
    for i in tqdm(range(T)):
        frame = x_np[:, :, i : i + 1, :]  # (1, 257, 1, 2)
        # switching to tflite shape
        frame_tf = _frame_tflite_layout(
            input_frame=frame, expected_shape=audio_in["shape"]
        )
        frame_q = _quantize(frame_tf, detail=audio_in)

        # set the input tensors and values
        interpreter.set_tensor(audio_in["index"], frame_q)
        interpreter.set_tensor(conv_in["index"], conv_cache)
        interpreter.set_tensor(tra_in["index"], tra_cache)
        for k in range(8):
            interpreter.set_tensor(tcn_ins[k]["index"], tcn_cache_list[k])

        interpreter.invoke()

        y_q = interpreter.get_tensor(audio_out["index"])
        conv_cache = interpreter.get_tensor(conv_out["index"])
        tra_cache = interpreter.get_tensor(tra_out["index"])
        # tcn_cache_list = [interpreter.get_tensor(d["index"]) for d in tcn_outs]
        tcn_cache_raw = [
            _tcn_cache_out_read(
                interpreter.get_tensor(tcn_outs[k]["index"]),
                tcn_ins[k]["shape"],
            )
            for k in range(8)
        ]

        # convert and reodering next input
        tcn_new_layout = [_to_tcn_layout(arr) for arr in tcn_cache_raw]

        tcn_cache_list = _reorder_tcn_caches(
            new_outs=tcn_new_layout, old_caches=tcn_cache_list, tcn_ins=tcn_ins
        )

        y_float = _dequantize(y_q, audio_out)

        # switch back to (B, F, T, C)
        y_can = _y_from_tflite(y_float, audio_out["shape"], (1, 257, 1, 2))
        y_frames.append(y_can)

    # return on time axis
    return np.concatenate(y_frames, axis=2)


if __name__ == "__main__":
    # testing infer
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
    tflite_stream_infer(
        x,
        model_path="gtcrn_micro/streaming/tflite/gtcrn_micro_stream_simple_float32.tflite",
        # model_path="gtcrn_micro/streaming/tflite/gtcrn_micro_stream_simple_int8.tflite",
    )
