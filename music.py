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

# ── 音声ロード関数 ──
def load_mp3(uploaded_file):
    """
    MP3を一時ファイル経由で読み込み、
    正規化したNumPy配列とサンプリングレートを返す
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
st.title("WaveForge")  # おすすめタイトル

# ファイルアップロード
df = st.file_uploader("MP3ファイルをアップロード", type="mp3")
if not df:
    st.info("MP3ファイルをアップロードしてください。")
    st.stop()

# 音声読み込み
data, orig_sr = load_mp3(df)
duration = len(data) / orig_sr

# ── 設定変更 ──
st.write("### 設定変更")
st.write("**標本化周波数(サンプリング周波数)**: 1秒間に何回の標本点として音の大きさを取り込むかを示します。高いほど細かい音を再現できます。")
st.write("**量子化ビット数**: 各標本点の電圧を何段階に分けて記録するかを示します。ビット数が多いほど音の強弱を滑らかに表現できます。")

# 太文字かつオレンジカラーのラベルを追加
st.markdown("<span style='font-weight:bold; color:orange;'>標本化周波数 (Hz)</span>", unsafe_allow_html=True)
target_sr = st.slider("", 1000, 48000, orig_sr, step=1000)
st.markdown("<span style='font-weight:bold; color:orange;'>量子化ビット数</span>", unsafe_allow_html=True)
bit_depth = st.slider("", 3, 24, 16, step=1)

# ── 再サンプリングと量子化 ──
rs_data = librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr)
max_int = 2**(bit_depth - 1) - 1
quantized = np.round(rs_data * max_int) / max_int

# ── 波形比較 ──
st.write("### 波形比較")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)

t_orig = np.linspace(0, duration, num=len(data))
ax1.plot(t_orig, data)
ax1.set_title("Original Waveform")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Amplitude")
ax1.set_xlim(0, duration)
ax1.set_ylim(-1, 1)

proc_len = min(len(quantized), int(duration * target_sr))
t_proc = np.linspace(0, duration, num=proc_len)
ax2.plot(t_proc, quantized[:proc_len])
ax2.set_title(f"Processed Waveform ({target_sr:,} Hz, {bit_depth:,} ビット)")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Amplitude")
ax2.set_xlim(0, duration)
ax2.set_ylim(-1, 1)
st.pyplot(fig)

# ── オーディオ再生 ──
subtype_map = {8: 'PCM_U8', 16: 'PCM_16', 24: 'PCM_24'}
# 指定ビット数がマップ外の場合は16ビットを使う
selected_subtype = subtype_map.get(bit_depth, 'PCM_16')
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out:
    sf.write(out.name, quantized, target_sr, subtype=selected_subtype)
    st.audio(out.name, format="audio/wav")

# ── データ量計算 ──
st.write("### データ量計算")
bytes_size = target_sr * bit_depth * 2 * duration / 8
kb_size = bytes_size / 1024
mb_size = kb_size / 1024
st.markdown("**アップロードして設定を変更したファイルのデータ量**")
# バイト数
st.markdown(
    f"{target_sr:,} Hz × {bit_depth:,} bit × 2 ch × {duration:.2f} 秒 ÷ 8 = {int(bytes_size):,} バイト"
)
# KB/MB 表記を KB=, MB=
st.markdown(
    f"KB＝{kb_size:,.2f}  MB＝{mb_size:,.2f}"
)

# チャンネル説明
st.write("- ステレオ(2ch): 左右2つの音声信号を同時に再生します。音に広がりがあります。")
st.write("- モノラル(1ch): 1つの音声信号で再生します。音の定位は中央になります。  ")
