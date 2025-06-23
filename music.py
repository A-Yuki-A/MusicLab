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
st.title("MP3 Resampler & Quantizer")

# ファイルアップロード
df = st.file_uploader("MP3ファイルをアップロード", type="mp3")
if not df:
    st.info("MP3ファイルをアップロードしてください。")
    st.stop()

# 音声読み込み
data, orig_sr = load_mp3(df)
duration = len(data) / orig_sr

# ── 設定変更 ──
st.markdown(
    """
    <div style='background-color:#f0f0f0; padding:10px; border-radius:5px;'>
    <h3>設定変更</h3>
    <p><strong>標本化周波数(サンプリング周波数)</strong>: 1秒間に何回の標本点として音の大きさを取り込むかを示します。高いほど細かい音を再現できます。</p>
    <p><strong>量子化ビット数</strong>: 各標本点の電圧を何段階に分けて記録するかを示します。ビット数が多いほど音の強弱を滑らかに表現できます。</p>
    </div>
    """, unsafe_allow_html=True
)
target_sr = st.slider("標本化周波数 (Hz)", 8000, 48000, orig_sr, step=1000)
bit_depth = st.slider("量子化ビット数", 8, 24, 16, step=1)

# ── 再サンプリングと量子化 ──
rs_data = librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr)
max_int = 2**(bit_depth - 1) - 1
quantized = np.round(rs_data * max_int) / max_int

# ── 波形比較 ──
st.markdown(
    """
    <div style='background-color:#f0f0f0; padding:10px; border-radius:5px;'>
    <h3>波形比較</h3>
    </div>
    """, unsafe_allow_html=True
)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)
# 元波形描画
t_orig = np.linspace(0, duration, num=len(data))
ax1.plot(t_orig, data)
...

# ── オーディオ再生 ──
subtype_map = {8: 'PCM_U8', 16: 'PCM_16', 24: 'PCM_24'}
selected_subtype = subtype_map[bit_depth]
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out:
    sf.write(out.name, quantized, target_sr, subtype=selected_subtype)
    st.audio(out.name, format="audio/wav")

# ── データ量計算 ──
st.markdown(
    """
    <div style='background-color:#f0f0f0; padding:10px; border-radius:5px;'>
    <h3>データ量計算</h3>
    </div>
    """, unsafe_allow_html=True
)
# 計算用変数定義
bytes_size = target_sr * bit_depth * 2 * duration / 8
kb_size = bytes_size / 1024
mb_size = kb_size / 1024
# 表示
st.markdown("**アップロードして設定を変更したファイルのデータ量**")
st.markdown(
    f"{target_sr:,} Hz × {bit_depth:,} bit × 2 ch × {duration:.2f} 秒 ÷ 8 = {int(bytes_size):,} バイト"
)
st.markdown(
    f"({kb_size:,.2f} KB / {mb_size:,.2f} MB)"
)

# チャンネル説明
st.write("- ステレオ(2ch): 左右2つの音声信号を同時に再生します。音に広がりがあります。")
st.write("- モノラル(1ch): 1つの音声信号で再生します。音の定位は中央になります。")
