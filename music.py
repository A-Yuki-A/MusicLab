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

# ファイルアップロード# ファイルアップロード\uploaded = st.file_uploader("Upload MP3 file", type="mp3")
if not uploaded:
    st.info("Please upload an MP3 file.")
    st.stop()

# 音声読み込み
data, orig_sr = load_mp3(uploaded)

# 処理パラメータ設定（波形表示の下に配置）
st.write("### Settings")
target_sr = st.slider("Sampling Rate (Hz)", 8000, 48000, orig_sr, step=1000)
bit_depth = st.slider("Bit Depth (bits)", 8, 24, 16, step=1)
st.write(f"**Original SR:** {orig_sr} Hz → **Target SR:** {target_sr} Hz | **Quantize:** {bit_depth}-bit")

# リサンプル & 量子化
rs_data = librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr)
max_int = 2**(bit_depth - 1) - 1
quantized = np.round(rs_data * max_int) / max_int

# 波形の比較表示
st.write("### Waveform Comparison")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)

# 軸範囲を固定
# 最大時間はオリジナルの長さを基準にする
max_time = len(data) / orig_sr
ax1.set_xlim(0, max_time)
ax2.set_xlim(0, max_time)
# 振幅は-1から1の範囲
ax1.set_ylim(-1, 1)
ax2.set_ylim(-1, 1)

# 元の波形
t_orig = np.linspace(0, max_time, num=len(data))
ax1.plot(t_orig, data)
ax1.set_title("Original Waveform")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Amplitude")

# 処理後の波形
# プロットのX軸に合わせてデータを切り詰め
proc_len = min(len(quantized), int(max_time * target_sr))
quant_trim = quantized[:proc_len]
t_proc = np.linspace(0, max_time, num=proc_len)
ax2.plot(t_proc, quant_trim)
ax2.set_title(f"Processed Waveform ({target_sr} Hz, {bit_depth}-bit)")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Amplitude")

st.pyplot(fig, use_container_width=False)

# サポートするビット深度とサブタイプのマッピング
subtype_map = {8: 'PCM_U8', 16: 'PCM_16', 24: 'PCM_24'}
selected_subtype = subtype_map.get(bit_depth, 'PCM_16')

# 再生用 WAV を一時ファイルに保存して再生
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out:
    sf.write(out.name, quantized, target_sr, subtype=selected_subtype)
    st.audio(out.name, format="audio/wav")
