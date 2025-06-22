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
    MP3を一時ファイル経由で読み込み、正規化したNumPy配列とサンプリングレートを返す
    """
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    audio = AudioSegment.from_file(tmp_path, format="mp3")
    sr = audio.frame_rate
    data = np.array(audio.get_array_of_samples(), dtype=np.float32)
    if audio.channels == 2:
        data = data.reshape((-1, 2)).mean(axis=1)  # モノラル化
    data /= np.abs(data).max()  # [-1,1]に正規化
    return data, sr

# ── アプリ本体 ──
# タイトル（絵文字削除）
st.title("MP3 Resampler & Quantizer")

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
bit_depth = st.slider("量子化ビット数", 8, 24, 16, step=1)
st.write(f"元のサンプリングレート: {orig_sr} Hz → 目標サンプリングレート: {target_sr} Hz | 量子化: {bit_depth} ビット")

# データサイズ計算
# サンプル数 × ビット数 ÷ 8 をKB/MBに換算
rs_data = librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr)
total_samples = len(rs_data)
total_bytes = total_samples * (bit_depth / 8)
kb = total_bytes / 1024
mb = kb / 1024
st.write(f"データサイズ: {int(total_bytes)} バイト = {kb:.2f} KB ({mb:.2f} MB)")

# リサンプルと量子化
max_int = 2**(bit_depth - 1) - 1
quantized = np.round(rs_data * max_int) / max_int

# 波形比較
st.write("### 波形比較")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)

# 軸を固定: 時間0〜元の長さ、振幅-1〜1
max_time = len(data) / orig_sr
ax1.set_xlim(0, max_time)
ax2.set_xlim(0, max_time)
ax1.set_ylim(-1, 1)
ax2.set_ylim(-1, 1)

# 元波形
t_orig = np.linspace(0, max_time, num=len(data))
ax1.plot(t_orig := t_orig if 't_orig' in locals() else t_orig, data)
ax1.set_title("元の波形")
ax1.set_xlabel("時間 (秒)")
ax1.set_ylabel("振幅")

# 処理後波形
proc_len = min(len(quantized), int(max_time * target_sr))
t_proc = np.linspace(0, max_time, num=proc_len)
ax2.plot(t_proc, quantized[:proc_len])
ax2.set_title(f"処理後の波形 ({target_sr} Hz, {bit_depth} ビット)")
ax2.set_xlabel("時間 (秒)")
ax2.set_ylabel("振幅")

st.pyplot(fig, use_container_width=False)

# WAV書き出しと再生
subtype_map = {8: 'PCM_U8', 16: 'PCM_16', 24: 'PCM_24'}
selected_subtype = subtype_map.get(bit_depth, 'PCM_16')
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out:
    sf.write(out.name, quantized, target_sr, subtype=selected_subtype)
    st.audio(out.name, format="audio/wav")
