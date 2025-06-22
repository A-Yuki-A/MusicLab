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

# ファイルアップロード
df = st.file_uploader("MP3ファイルをアップロード", type="mp3")
if not df:
    st.info("MP3ファイルをアップロードしてください。")
    st.stop()

# 音声読み込み
data, orig_sr = load_mp3(df)

# 設定変更
st.write("### 設定変更")
target_sr = st.slider("標本化周波数 (Hz)", 8000, 48000, orig_sr, step=1000)
bit_depth = st.slider("量子化ビット数 (bits)", 8, 24, 16, step=1)
st.write(f"**Original SR:** {orig_sr} Hz → **Target SR:** {target_sr} Hz | **Quantize:** {bit_depth}-bit")

# リサンプル & 量子化
rs_data = librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr)
max_int = 2**(bit_depth - 1) - 1
quantized = np.round(rs_data * max_int) / max_int

# 波形比較表示
st.write("### 波形比較")
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 9), constrained_layout=True)

# 軸範囲を固定
max_time = len(data) / orig_sr
ax1.set_xlim(0, max_time)
ax2.set_xlim(0, max_time)
ax3.set_xlim(0, max_time * 0.05)  # 最初の5%をズーム
ax1.set_ylim(-1, 1)
ax2.set_ylim(-1, 1)
ax3.set_ylim(-1, 1)

# 元の波形
t_orig = np.linspace(0, max_time, num=len(data))
ax1.plot(t_orig := np.linspace(0, max_time, num=len(data)), data)
ax1.set_title("元の波形")
ax1.set_xlabel("時間 (秒)")
ax1.set_ylabel("振幅")

# 処理後の波形
proc_len_full = min(len(quantized), int(max_time * target_sr))
ax2.plot(np.linspace(0, max_time, num=proc_len_full), quantized[:proc_len_full])
ax2.set_title(f"処理後の波形 ({target_sr} Hz, {bit_depth}-bit)")
ax2.set_xlabel("時間 (秒)")
ax2.set_ylabel("振幅")

# ズーム表示 (最初の 50 ms 相当)
zoom_len = int(target_sr * 0.05)
zoom_orig = librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr)[:zoom_len]
zoom_proc = quantized[:zoom_len]
time_zoom = np.linspace(0, zoom_len/target_sr, num=zoom_len)
ax3.plot(time_zoom, zoom_orig, label="リサンプル後")
ax3.plot(time_zoom, zoom_proc, linestyle='--', label="量子化後")
ax3.set_title("波形ズーム (最初の50ms)")
ax3.set_xlabel("時間 (秒)")
ax3.set_ylabel("振幅")
ax3.legend()

st.pyplot(fig, use_container_width=False)

# WAV サブタイプマップ
subtype_map = {8: 'PCM_U8', 16: 'PCM_16', 24: 'PCM_24'}
selected_subtype = subtype_map.get(bit_depth, 'PCM_16')

# 再生用 WAV を一時ファイルに保存して再生
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out:
    sf.write(out.name, quantized, target_sr, subtype=selected_subtype)
    st.audio(out.name, format="audio/wav")
