import streamlit as st
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
import librosa
import tempfile
import soundfile as sf

# ── ffmpeg/ffprobe のパス指定 ──
AudioSegment.converter = "/usr/bin/ffmpeg"
AudioSegment.ffprobe   = "/usr/bin/ffprobe"

def load_mp3(uploaded_file):
    """
    Load MP3 via temp file and return normalized numpy array and sampling rate.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    audio = AudioSegment.from_file(tmp_path, format="mp3")
    sr = audio.frame_rate
    data = np.array(audio.get_array_of_samples(), dtype=np.float32)
    if audio.channels == 2:
        data = data.reshape((-1, 2)).mean(axis=1)  # convert to mono
    data /= np.abs(data).max()  # normalize to [-1,1]
    return data, sr

st.title("🎧 MP3 Resampler & Quantizer")

# File upload
uploaded = st.file_uploader("Upload MP3 file", type="mp3")
if not uploaded:
    st.info("Please upload an MP3 file to continue.")
    st.stop()

# Load audio
data, orig_sr = load_mp3(uploaded)

# Settings
st.write("### Settings")
target_sr = st.slider("Sampling Rate (Hz)", 8000, 48000, orig_sr, step=1000)
bit_depth = st.slider("Quantization Bits", 8, 24, 16, step=1)
st.write(f"Original SR: {orig_sr} Hz → Target SR: {target_sr} Hz | Quantization: {bit_depth}-bit")

# Resample and quantize
data_rs = librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr)
max_int = 2**(bit_depth - 1) - 1
quantized = np.round(data_rs * max_int) / max_int

# Waveform comparison
st.write("### Waveform Comparison")
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 9), constrained_layout=True)

# Fixed axes
max_time = len(data) / orig_sr
ax1.set_xlim(0, max_time)
ax2.set_xlim(0, max_time)
ax1.set_ylim(-1, 1)
ax2.set_ylim(-1, 1)

# Original waveform
t_orig = np.linspace(0, max_time, num=len(data))
ax1.plot(t_orig, data)
ax1.set_title("Original Waveform")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Amplitude")

# Processed waveform
proc_len = min(len(quantized), int(max_time * target_sr))
t_proc = np.linspace(0, max_time, num=proc_len)
ax2.plot(t_proc, quantized[:proc_len])
ax2.set_title(f"Processed Waveform ({target_sr} Hz, {bit_depth}-bit)")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Amplitude")

# Zoom view (first 20 ms)
zoom_duration = 0.02
zoom_len = int(target_sr * zoom_duration)
time_zoom = np.linspace(0, zoom_duration, num=zoom_len)
zoom_resampled = data_rs[:zoom_len]
zoom_quant = quantized[:zoom_len]
ax3.plot(time_zoom, zoom_resampled, label="Resampled", linestyle='-')
ax3.step(time_zoom, zoom_quant, where='mid', label="Quantized", linewidth=1.0)
ax3.plot(time_zoom, zoom_quant, marker='o', linestyle='None', label="Quantized Samples")
ax3.set_title("Zoomed Waveform (First 20 ms)")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Amplitude")
ax3.legend()

st.pyplot(fig, use_container_width=False)

# Write out WAV and play
subtype_map = {8: 'PCM_U8', 16: 'PCM_16', 24: 'PCM_24'}
selected_subtype = subtype_map.get(bit_depth, 'PCM_16')
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out:
    sf.write(out.name, quantized, target_sr, subtype=selected_subtype)
    st.audio(out.name, format="audio/wav")
