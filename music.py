import streamlit as st
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
import librosa
import tempfile
import soundfile as sf  # librosa の依存で入っているはず

# ── ffmpeg/ffprobe のパス指定 ──
AudioSegment.converter = "/usr/bin/ffmpeg"
AudioSegment.ffprobe   = "/usr/bin/ffprobe"

def load_mp3(uploaded_file):
    # 一時ファイルに書き出して読み込む
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    audio = AudioSegment.from_file(tmp_path, format="mp3")
    sr = audio.frame_rate
    data = np.array(audio.get_array_of_samples(), dtype=np.float32)
    if audio.channels == 2:
        data = data.reshape((-1, 2))
        data = data.mean(axis=1)  # モノラル化
    # 正規化
    data /= np.abs(data).max()
    return data, sr

st.title("🎧 MP3 Resampler & Quantizer")

# サイドバーに設定コントロール
st.sidebar.header("Settings")
target_sr = st.sidebar.selectbox(
    "Sampling Rate (Hz)",
    [8000, 16000, 22050, 44100, 48000],
    index=3
)
bit_depth = st.sidebar.selectbox(
    "Bit Depth (bits)",
    [8, 16, 24],
    index=1
)

uploaded = st.file_uploader("Upload MP3 file", type="mp3")
if not uploaded:
    st.info("Please upload an MP3 file.")
    st.stop()

# 元データの読み込み
data, orig_sr = load_mp3(uploaded)
st.write(f"**Original SR:** {orig_sr} Hz → **Target SR:** {target_sr} Hz")
st.write(f"**Quantization:** {bit_depth}-bit")

# リサンプル
data_rs = librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr)

# 量子化
max_int = 2**(bit_depth - 1) - 1
quantized = np.round(data_rs * max_int) / max_int

# 波形表示
st.write("### Waveform")
fig, ax = plt.subplots()
t = np.linspace(0, len(quantized) / target_sr, num=len(quantized))
ax.plot(t, quantized)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
st.pyplot(fig)

# 出力用 WAV を一時ファイルに保存して再生
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out:
    sf.write(out.name, quantized, target_sr, subtype=f"PCM_{bit_depth}")
    st.audio(out.name, format="audio/wav")
