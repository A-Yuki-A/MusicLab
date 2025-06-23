import streamlit as st
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
import librosa
import tempfile
import soundfile as sf

# ── ページ設定と淡いグレー基調スタイル ──
st.set_page_config(
    page_title="MP3 Resampler & Quantizer",
    layout="wide"
)
st.markdown(
    """
    <style>
    /* 背景を淡いグレーに設定 */
    .stApp {
        background-color: #f5f5f5;
    }
    /* セクション見出し */
    h2, h3, h4 {
        color: #333333;
        border-bottom: 1px solid #dddddd;
        padding-bottom: 4px;
    }
    /* スライダーやボタンの背景 */
    .stSlider > div {
        background-color: #e0e0e0;
    }
    /* マークダウンのテキスト */
    .markdown-text-container p {
        color: #444444;
    }
    /* 波形図とオーディオセクション */
    .stPlotlyChart, .stAudio {
        background-color: white;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 0 5px rgba(0,0,0,0.1);
    }
    /* データ量セクション */
    .stMarkdown {
        background-color: #ffffff;
        padding: 8px;
        border-left: 4px solid #cccccc;
    }
    </style>
    """, unsafe_allow_html=True
)

# ── ffmpeg/ffprobe のパス指定 ──
AudioSegment.converter = "/usr/bin/ffmpeg"
AudioSegment.ffprobe   = "/usr/bin/ffprobe"

# ── 音声ロード関数 ──
def load_mp3(uploaded_file):
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

data, orig_sr = load_mp3(df)

# ── 設定変更 ──
st.write("### 設定変更")
st.write("**標本化周波数(サンプリング周波数)**: 1秒間に何回の標本点として音の大きさを取り込むかを示します。高いほど細かい音を再現できます。")
st.write("**量子化ビット数**: 各標本点の電圧を何段階に分けて記録するかを示します。ビット数が多いほど音の強弱を滑らかに表現できます。")

target_sr = st.slider("標本化周波数 (Hz)", 100, 48000, orig_sr, step=1000)
bit_depth = st.slider("量子化ビット数", 3, 24, 16, step=1)

# 再サンプリングと量子化
rs_data = librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr)
max_int = 2**(bit_depth - 1) - 1
quantized = np.round(rs_data * max_int) / max_int

# ── 波形比較 ──
st.write("### 波形比較")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)

duration = len(data) / orig_sr
ax1.plot(np.linspace(0, duration, num=len(data)), data)
ax1.set(title="Original Waveform", xlabel="Time (s)", ylabel="Amplitude")
ax1.set(xlim=(0, duration), ylim=(-1, 1))

proc_len = min(len(quantized), int(duration * target_sr))
ax2.plot(np.linspace(0, duration, num=proc_len), quantized[:proc_len])
ax2.set(title=f"Processed Waveform ({target_sr:,} Hz, {bit_depth:,} ビット)", xlabel="Time (s)", ylabel="Amplitude")
ax2.set(xlim=(0, duration), ylim=(-1, 1))

st.pyplot(fig, use_container_width=False)

# ── オーディオ再生 ──
subtype_map = {8: 'PCM_U8', 16: 'PCM_16', 24: 'PCM_24'}
selected_subtype = subtype_map.get(bit_depth, 'PCM_16')
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out:
    sf.write(out.name, quantized, target_sr, subtype=selected_subtype)
    st.audio(out.name, format="audio/wav")

# ── データ量計算 ──
st.write("### データ量計算")
st.markdown("**アップロードして設定を変更したファイルのデータ量**")
bytes_size = target_sr * bit_depth * 2 * duration / 8
kb_size = bytes_size / 1024
mb_size = kb_size / 1024
st.markdown(f"{target_sr:,} Hz × {bit_depth:,} bit × 2 ch × {duration:.2f} 秒 ÷ 8 = {int(bytes_size):,} バイト\n({kb_size:,.2f} KB / {mb_size:,.2f} MB)")

# チャンネル説明
st.write("- ステレオ(2ch): 左右2つの音声信号を同時に再生します。音に広がりがあります。")
st.write("- モノラル(1ch): 1つの音声信号で再生します。音の定位は中央になります。")
