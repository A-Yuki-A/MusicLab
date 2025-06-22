import streamlit as st
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
import librosa
import tempfile
import soundfile as sf

# ── ページ設定 ──
st.set_page_config(
    page_title="MP3 Resampler & Quantizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── ffmpeg/ffprobe のパス指定 ──
AudioSegment.converter = "/usr/bin/ffmpeg"
AudioSegment.ffprobe   = "/usr/bin/ffprobe"

def load_mp3(uploaded_file):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    audio = AudioSegment.from_file(tmp_path, format="mp3")
    sr = audio.frame_rate
    data = np.array(audio.get_array_of_samples(), dtype=np.float32)
    if audio.channels == 2:
        data = data.reshape((-1, 2)).mean(axis=1)  # モノラル化
    data /= np.abs(data).max()
    return data, sr

st.title("🎧 MP3 Resampler & Quantizer")

# サイドバー設定（必ず展開されます）
with st.sidebar:
    st.header("Settings")
    target_sr = st.slider("Sampling Rate (Hz)", 8000, 48000, 44100, 1000)
    bit_depth = st.slider("Bit Depth (bits)", 8, 24, 16, 1)

# メインにもスライダーを置きたい場合はこちらをコメントアウト解除
# target_sr = st.slider("Sampling Rate (Hz)", 8000, 48000, 44100, 1000)
# bit_depth = st.slider("Bit Depth (bits)", 8, 24, 16, 1)

uploaded = st.file_uploader("Upload MP3 file", type="mp3")
if not uploaded:
    st.info("Please upload an MP3 file.")
    st.stop()

# 音声読み込み
data, orig_sr = load_mp3(uploaded)
st.write(f"**Original SR:** {orig_sr} Hz → **Target SR:** {target_sr} Hz   **Quantization:** {bit_depth}-bit")

# リサンプル・量子化
data_rs = librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr)
max_int = 2**(bit_depth - 1) - 1
quantized = np.round(data_rs * max_int) / max_int

# 波形表示
st.write("### Waveform after Resampling & Quantization")
fig, ax = plt.subplots()
t = np.linspace(0, len(quantized) / target_sr, num=len(quantized))
ax.plot(t, quantized)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
st.pyplot(fig)

# 再生
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out:
    sf.write(out.name, quantized, target_sr, subtype=f"PCM_{bit_depth}")
    st.audio(out.name, format="audio/wav")
