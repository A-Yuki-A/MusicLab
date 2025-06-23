import streamlit as st
try:
    from pydub import AudioSegment
except ModuleNotFoundError:
    st.error("pydub モジュールがインストールされていません。requirements.txt に 'pydub' を追加してください。")
    st.stop()
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

# ── ページ設定 ──
st.set_page_config(
    page_title="WaveForge",
    layout="centered"
)

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
# 指示文
st.markdown(
    "**標本化周波数と量子化ビット数を変えて、音の違いを聴き比べしなさい。**"
)
# ラベルと説明を1行で表示（オレンジラベル＋黒説明）
st.markdown(
    "<span style='font-weight:bold; color:orange;'>標本化周波数 (Hz)：1秒間に何回の標本点として音の大きさを取り込むかを示します。高いほど細かい音を再現できます。</span>",
    unsafe_allow_html=True
)
# スライダー
 target_sr = st.slider("", 1000, 48000, orig_sr, step=1000)
# ラベルと説明を1行で表示（オレンジラベル＋黒説明）
st.markdown(
    "<span style='font-weight:bold; color:orange;'>量子化ビット数：各標本点の電圧を何段階に分けて記録するかを示します。ビット数が多いほど音の強弱を滑らかに表現できます。</span>",
    unsafe_allow_html=True
)
# スライダー
 bit_depth = st.slider("", 3, 24, 16, step=1){kb_size:,.2f}"
)
st.markdown(
    f"MB＝{mb_size:,.2f}"
)
# チャンネル説明を追加
st.write("- ステレオ(2ch): 左右2つの音声信号を同時に再生します。音に広がりがあります。")
st.write("- モノラル(1ch): 1つの音声信号で再生します。音の定位は中央になります。")
st.markdown(
    f"MB＝{mb_size:,.2f}"
)
