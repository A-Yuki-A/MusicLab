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
        data = data.reshape((-1, 2)).mean(axis=1)
    data /= np.abs(data).max()
    return data, sr

# ── アプリ本体 ──
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

# データ量の計算式表示（固定）
# 標本化周波数 × 量子化ビット数 × 再生時間 ÷ 8 をバイト、KB、MBで表示
duration = len(data) / orig_sr  # 秒数
bytes_size = target_sr * bit_depth * duration / 8
kb_size = bytes_size / 1024
mb_size = kb_size / 1024
# 3桁区切りでフォーマット
bytes_str = f"{int(bytes_size):,}"
kb_str = f"{kb_size:,.2f}"
mb_str = f"{mb_size:,.2f}"
# 太文字で表示
st.markdown(f"**データ量の計算式: {target_sr:,} Hz × {bit_depth:,} ビット × {duration:.2f} 秒 ÷ 8 = {bytes_str} バイト ({kb_str} KB / {mb_str} MB)**")

# 問いを追加
st.write("---")
st.write("**問い1: 標本化周波数を変えると音がどのように変化しますか？**")
st.write("**問い2: 量子化ビット数を変えると音がどのように変化しますか？**")
st.write("---")

# リサンプルと量子化
rs_data = librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr)
max_int = 2**(bit_depth - 1) - 1
quantized = np.round(rs_data * max_int) / max_int

# 波形比較
st.write("### 波形比較")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)

# 軸を固定: 時間0〜再生時間、振幅-1〜1
duration = len(data) / orig_sr
ax1.set_xlim(0, duration)
ax2.set_xlim(0, duration)
ax1.set_ylim(-1, 1)
ax2.set_ylim(-1, 1)

# 元波形
t_orig = np.linspace(0, duration, num=len(data))
ax1.plot(t_orig, data)
ax1.set_title("Original Waveform")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Amplitude")

# 処理後波形
proc_len = min(len(quantized), int(duration * target_sr))
t_proc = np.linspace(0, duration, num=proc_len)
ax2.plot(t_proc, quantized[:proc_len])
ax2.set_title(f"Processed Waveform ({target_sr:,} Hz, {bit_depth:,} ビット)")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Amplitude")

st.pyplot(fig, use_container_width=False)

# WAV書き出しと再生
subtype_map = {8: 'PCM_U8', 16: 'PCM_16', 24: 'PCM_24'}
selected_subtype = subtype_map.get(bit_depth, 'PCM_16')
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out:
    sf.write(out.name, quantized, target_sr, subtype=selected_subtype)
    st.audio(out.name, format="audio/wav")
