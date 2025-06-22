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
    アップロードされた MP3 を一時ファイル経由で読み込み、
    正規化した NumPy 配列とサンプリング周波数を返す
    """
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    audio = AudioSegment.from_file(tmp_path, format="mp3")
    sr = audio.frame_rate
    data = np.array(audio.get_array_of_samples(), dtype=np.float32)
    if audio.channels == 2:
        data = data.reshape((-1, 2)).mean(axis=1)  # モノラル化
    data /= np.abs(data).max()  # 正規化
    return data, sr

# ── Streamlit アプリ本体 ──
st.title("🎧 MP3 Resampler & Quantizer")

uploaded = st.file_uploader("Upload MP3 file", type="mp3")
if not uploaded:
    st.info("Please upload an MP3 file.")
    st.stop()

# 元データ読み込み
data, orig_sr = load_mp3(uploaded)
st.write(f"**Original SR:** {orig_sr} Hz")

# 元波形表示
st.write("### Waveform")
fig, ax = plt.subplots()
t = np.linspace(0, len(data) / orig_sr, num=len(data))
ax.plot(t, data)
ax.set_xlabel("Time (s)
")
ax.set_ylabel("Amplitude")
st.pyplot(fig, use_container_width=False)

# スライダー設定（波形の下）
st.write("### Settings")
target_sr = st.slider("Sampling Rate (Hz)", 8000, 48000, orig_sr, step=1000)
bit_depth = st.slider("Bit Depth (bits)", 8, 24, 16, step=1)
st.write(f"**Resample to:** {target_sr} Hz, **Quantize to:** {bit_depth}-bit")

# リサンプル & 量子化
rs_data = librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr)
max_int = 2**(bit_depth - 1) - 1
quantized = np.round(rs_data * max_int) / max_int

# 処理後波形表示
st.write("### Waveform after Processing")
fig2, ax2 = plt.subplots()
t2 = np.linspace(0, len(quantized) / target_sr, num=len(quantized))
ax2.plot(t2, quantized)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Amplitude")
st.pyplot(fig2, use_container_width=False)

# 再生
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out:
    sf.write(out.name, quantized, target_sr, subtype=f"PCM_{bit_depth}")
    st.audio(out.name, format="audio/wav")
