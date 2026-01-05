#!/usr/bin/bash

set -euo pipefail

# run from root
ONNX_INPUT="gtcrn_micro/streaming/onnx/"
ONNX_FILE=gtcrn_micro_stream_simple.onnx
OUTPUT_PATH="gtcrn_micro/streaming/tflite/"
CALIB_PATH="gtcrn_micro/streaming/tflite/calibration_data/"
JSON_FILE=replace_gtcrn_micro.json

# functions to help organize the file
die() {
	echo "ERROR: $*" >&2
	exit 1
}
need_file() { [[ -f "$1" ]] || die "Missing file: $1"; }
trim() { tr -d ' \n\r\t'; }
inv_scale() {
	python - <<PY
scale = float("$1")
print(1.0/scale)
PY
}

# convert the model from PyTorch ->> ONNX
if [ -e "$ONNX_INPUT$ONNX_FILE" ]; then
	echo "$ONNX_INPUT$ONNX_FILE exists..."
else
	echo "$ONNX_INPUT$ONNX_FILE doesn't exist..."
	echo "Running Streaming Torch -> ONNX conversion"
	uv run -m gtcrn_micro.streaming.conversion.stream_onnx
	echo "$ONNX_FILE created in $ONNX_INPUT"
fi
# check that it exists now:
need_file "$ONNX_INPUT$ONNX_FILE"

# check for json file
if [[ -f "$OUTPUT_PATH$JSON_FILE" ]]; then
	echo "Json file $JSON_FILE exists..."
else
	echo "Json file for replacement doesn't exist"
fi

# checking if calibration directory exists
if [[ ! -d "${CALIB_PATH}" ]]; then
	echo "Missing calibration data..."
	uv run -m gtcrn_micro.utils.calibration_data
fi
# double check the directory exists now
[[ -d "${CALIB_PATH}" ]] || die "Calibration directory missing: ${CALIB_PATH}"

# getting the names of required inputs
# INPUTS=(audio conv_cache tra_cache)
# for k in {0..7}; do INPUTS+=("tcn_cache_${k}"); done
# Converter jumbles up the inputs, so arranging explicitly
INPUTS=(
	tcn_cache_0
	tcn_cache_4
	tcn_cache_1
	tcn_cache_2
	tra_cache
	tcn_cache_5
	conv_cache
	tcn_cache_7
	tcn_cache_3
	audio
	tcn_cache_6
)

AUDIO_MEAN="[[[[0.5, 0.5]]]]"
AUDIO_STD="[[[[%s, %s]]]]"
# AUDIO_MEAN="[0.5]"
# AUDIO_STD="[%s]"

CACHE_MEAN="[0.5]"
CACHE_STD="[%s]"

# getting the non-variable args:
args=(
	-i "${ONNX_INPUT}${ONNX_FILE}"
	-o "${OUTPUT_PATH}"
	-prf ${OUTPUT_PATH}${JSON_FILE}
	-cotof
	-coion
	# -oiqt
	-qt per-channel
	-agj
	-rtpo PReLU
	-osd
	-b 1
	# -ois "audio:1,257,1,2" "conv_cache:2,1,16,6,33" "tra_cache:2,3,1,8,2" "tcn_cache_0:1,16,2,33" "tcn_cache_1:1,16,4,33" "tcn_cache_2:1,16,8,33" "tcn_cache_3:1,16,16,33" "tcn_cache_4:1,16,2,33" "tcn_cache_5:1,16,4,33" "tcn_cache_6:1,16,8,33" "tcn_cache_7:1,16,16,33"
	-v debug
	-ofgd
	-nodaftc 6
)

# -kat list
args+=(-kat "${INPUTS[@]}")

# making the -cind list with the std calc
for file in "${INPUTS[@]}"; do

	# name the file
	npy="${CALIB_PATH}${file}.npy"
	txt="${CALIB_PATH}${file}.txt"

	# check the file exists
	need_file "$npy"
	need_file "$txt"

	printf '\nNPY File: %s\n' "$npy"
	printf 'TXT File: %s\n' "$txt"

	# getting the scale for STD
	scale="$(trim <"$txt")"
	[[ -n "$scale" ]] || die "Empty scale in $txt"
	std="$(inv_scale "$scale")"
	# printf 'STD: %f\n' "$std"

	if [[ "$file" == "audio" ]]; then
		mean="$AUDIO_MEAN"
		cind_std="$(printf "$AUDIO_STD" "$std" "$std")"
	else
		mean="$CACHE_MEAN"
		cind_std="$(printf "$CACHE_STD" "$std")"
	fi

	echo "Using $file: scale=$scale std=$std npy=$npy"
	args+=(-cind "$file" "$npy" "$mean" "$cind_std")

done

# old non-streaming variant
# uv run onnx2tf \
# 	\
# 	-i "${ONNX_INPUT}${ONNX_FILE}" \
# 	-o "${OUTPUT_PATH}" \
# 	\
# 	-prf ${OUTPUT_PATH}${JSON_FILE} \
# 	-cotof \
# 	-oiqt \
# 	-qt per-channel \
# 	-cind "audio" "$CALIB_DATA" "[[[[$STD], [$STD]]]]" \
# 	-rtpo PReLU \
# 	-osd \
# 	-b 1 \
# 	-v debug \
# 	-kat audio conv_cache tra_cache tcn_cache \
# 	-ofgd

# run onnx conversion
echo "Running ONNX -> TF..."
uv run onnx2tf "${args[@]}"
