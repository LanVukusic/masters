import torch
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder
from heartlib.models import HeartCodec  # From the official HeartMuLa repo

# Setup device execution
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. LOAD RAW AUDIO (Using PyTorch's native torchcodec instead of torchaudio)
# HeartCodec natively processes audio sampled at 48,000 Hz (48 kHz)
input_file = "music.mp3"
decoder = AudioDecoder(input_file)

# Extract all samples and retrieve data as a standard PyTorch Tensor
# torchcodec naturally outputs shapes as [channels, samples]
audio_samples = decoder.get_all_samples()
waveform = audio_samples.data.to(device)
src_sample_rate = audio_samples.sample_rate

# HeartCodec expects a mono (1 channel) or explicit batch channel format.
# Let's mix down to mono if it's stereo, and resample to 48kHz if required
if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)

if src_sample_rate != 48000:
    # Use PyTorch functional transformations if you need to resample
    # Ensure tensor shape format is [Batch, Channels, Time] for HeartCodec
    breakpoint()
    pass 

print(waveform.shape)

# Reshape waveform to [Batch, Channels, Time_Samples] -> [1, 1, Num_Samples]
waveform = waveform.unsqueeze(0) 
print(waveform.shape)


# 2. INITIALIZE HEARTCODEC
# Load the pre-trained weights from the official HeartMuLa repository
codec_model = HeartCodec.from_pretrained(
    "HeartMuLa/HeartCodec-oss-20260123", 
    device_map=device,
    ignore_mismatched_sizes=True
).to(device)
codec_model.eval()

# 3. ENCODE: Waveform -> Discrete Semantic Tokens
with torch.no_grad():
    # HeartCodec processes the waveform and outputs quantized token representations
    # tokens shape: [Batch, Codebooks, Frames] -> e.g., [1, 8, 12.5 * seconds]
    tokens = codec_model.encode(waveform)


print(f"Encoded Token Tensor Shape: {tokens.shape}")
print(f"Tokens generated for 1 second: {tokens.shape[-1] * tokens.shape[1] / (waveform.shape[-1]/48000)}")

# --- AT THIS POINT, YOU WOULD FEED 'TOKENS' INTO YOUR PREDICTION MODEL ---
# E.g., history_tokens = tokens[:, :, :62] (First 5 seconds at 12.5Hz)
# predicted_tokens = your_model.predict_next(history_tokens, steps=37) (Next 3 seconds)
# ------------------------------------------------------------------------

# 4. DECODE: Tokens -> Reconstructed High-Fidelity Waveform
with torch.no_grad():
    # Convert tokens back into a continuous raw waveform audio tensor
    reconstructed_waveform = codec_model.decode(tokens)

# Remove batch dimensions to prepare for saving [Channels, Time]
output_waveform = reconstructed_waveform.squeeze(0).cpu()

# 5. WRITE AUDIO BACK TO DISK (Using PyTorch's native torchcodec encoder)
output_file = "reconstructed_output.wav"
encoder = AudioEncoder()

# Match HeartCodec's strict native output sample rate
audio_stream = encoder.add_audio(
    sample_rate=48000,
    num_channels=output_waveform.shape[0]
)

with encoder.open_file(output_file):
    audio_stream.add_samples(output_waveform)

print(f"Successfully processed and saved audio to {output_file}!")
