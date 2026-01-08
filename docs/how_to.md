# How-To guide for various items in project

- - -

## Using the streaming variants of the model

### PyTorch Streaming model:
   - Essentially, you'll want to load up the non-streaming model weights as normal, then run convert_to_stream function passing in the streaming model in eval. See the "how_to" guide for more details: [how_to.md](./docs/how_to.md)
   - **Note**: This comes from the `output_tests.py` file in [gtcrn_micro/utils/](../gtcrn_micro/utils/) - To test running this, as well as the `ONNX` and `TFLite` (or now `LiteRT`) inference, feel free to use that file. 
```py
from gtcrn_micro.models.gtcrn_micro import GTCRNMicro
from gtcrn_micro.streaming.conversion.convert import convert_to_stream
from gtcrn_micro.streaming.gtcrn_micro_stream import StreamGTCRNMicro

# loading up weights for trained offline model
model = GTCRNMicro()
ckpt = torch.load(
    "./gtcrn_micro/ckpts/best_model_dns3.tar",
    map_location="cpu",
)

# gettin the state dict
state = (
    ckpt.get("state_dict", None)
    or ckpt.get("model_state_dict", None)
    or ckpt.get("model", None)
    or ckpt
)
# Handling if ckpt was saved from DDP and has module prefixes
if any(k.startswith("module.") for k in state.keys()):
    state = {k.removeprefix("module."): v for k, v in state.items()}

# Double checking we've loaded and setup everything correctly here
missing, unexpected = model.load_state_dict(state, strict=False)
print("-" * 20)
print(f"\tmissing keys: {missing}")
print(f"\tunexpected keys: {unexpected}")

# explicitly setting model to eval in function
device = torch.device("cpu")
model.eval()
model.to("cpu")

# converting the model weights to the streaming model
stream_model = StreamGTCRNMicro().to(device).eval()
convert_to_stream(stream_model, model)

# running inference: 
mix, fs = sf.read(
    "./gtcrn_micro/streaming/sample/noisy1.wav",
    dtype="float32",
)
x = torch.from_numpy(mix)
# running stft
x = torch.stft(
    x, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False
)[None]

# setting up the streaming caches, can be init to zeros
# NOTE: Just make sure the shapes are correct
conv_cache = torch.zeros(2, 1, 16, 6, 33).to(device)
tra_cache = torch.zeros(2, 3, 1, 8, 2).to(device)
tcn_cache = [
    [torch.zeros(1, 16, 2 * d, 33, device=device) for d in [1, 2, 4, 8]],
    [torch.zeros(1, 16, 2 * d, 33, device=device) for d in [1, 2, 4, 8]],
]

# our outputs
ys = []
# Time dimension
for i in tqdm(range(x.shape[2])):
    xi = x[:, :, i : i + 1]
    # running frame-by-frame
    with torch.no_grad():
        yi, conv_cache, tra_cache, tcn_cache = stream_model(
            xi, conv_cache, tra_cache, tcn_cache
        )
    ys.append(yi)
# concatenating over time
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
sf.write(
    "gtcrn_micro/streaming/sample/enh_torch1.wav",
    enhanced_pytorch,
    16000,
)
```